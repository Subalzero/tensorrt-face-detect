#include "facedetectionapp.h"

FaceDetectionApp::FaceDetectionApp(int argc, char** argv) :
    _argc(argc), _argv(argv)
{
	
}

FaceDetectionApp::~FaceDetectionApp()
{
}

int FaceDetectionApp::exec()
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, _argc, _argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        print_help_info();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        print_help_info();
        return EXIT_SUCCESS;
    }
	
    auto sampleTest = sample::gLogger.defineTest("FaceDetect", _argc, _argv);

    sample::gLogger.reportTestStart(sampleTest);

    _runner = TensorRTRunner(initializeParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!_runner.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    sample::gLogger.reportPass(sampleTest);

    cv::VideoCapture cap(0, cv::CAP_MSMF);

    if (cap.isOpened())
    {
        cv::Mat frame;
        cap >> frame;
        cv::Mat fr = preprocess(frame);
        auto input_shape = _runner.get_input_shape();

        std::vector<float> input_vector((float*)fr.data, (float*)fr.data + (input_shape[1] * input_shape[2] * input_shape[3]));
        _runner.process(input_vector);
    }
}

void FaceDetectionApp::print_help_info()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << "[-t or --timingCacheFile=<path to timing cache file]" << std::endl;
    std::cout << "--help             Display help information" << std::endl;
    std::cout << "--datadir          Specify path to a data directory, overriding the default. This option can be used "
        "multiple times to add multiple directories. If no data directories are given, the default is to use "
        "(data/samples/mnist/, data/mnist/)"
        << std::endl;
    std::cout << "--useDLACore=N     Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
        "where n is the number of DLA engines on the platform."
        << std::endl;
    std::cout << "--int8             Run in Int8 mode." << std::endl;
    std::cout << "--fp16             Run in FP16 mode." << std::endl;
    std::cout << "--bf16             Run in BF16 mode." << std::endl;
    std::cout << "--timingCacheFile  Specify path to a timing cache file. If it does not already exist, it will be "
        << "created." << std::endl;
}

RunnerParams FaceDetectionApp::initializeParams(const samplesCommon::Args& args)
{
    RunnerParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("models");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "version-RFB-640.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.bf16 = args.runInBf16;
    params.timingCacheFile = args.timingCacheFile;

    return params;
}

cv::Mat FaceDetectionApp::preprocess(const cv::Mat& frame)
{
    cv::Mat out;
    auto input_shape = _runner.get_input_shape();
    uint32_t mWidth = input_shape[3];
    uint32_t mHeight = input_shape[2];
    cv::resize(frame, out, cv::Size(mWidth, mHeight));
    out.convertTo(out, CV_32F);
    cv::Mat output = cv::dnn::blobFromImage(out, 1. / 128, {}, { 127, 127, 127 });
    return output;
}
