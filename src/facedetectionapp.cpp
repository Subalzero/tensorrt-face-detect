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

    sample::gLogInfo << "Building and running a GPU inference engine for Face Detection" << std::endl;

    if (!_runner.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    cv::VideoCapture cap(0, cv::CAP_MSMF);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    cv::Mat frame;
    cap >> frame;
    cv::Mat fr = preprocess(frame);

    auto input_shape = _runner.get_input_shape(0);

    std::vector<float> input_vector((float*)fr.data, (float*)fr.data + (input_shape[1] * input_shape[2] * input_shape[3]));
    _runner.process_async({ input_vector });

    std::deque<uint64_t> latency;
    auto start = std::chrono::system_clock::now();
    while (cap.isOpened())
    {
        cv::Mat frame;
        cap >> frame;
        cv::Mat fr = preprocess(frame);
        auto input_shape = _runner.get_input_shape(0);

        std::vector<float> input_vector((float*)fr.data, (float*)fr.data + (input_shape[1] * input_shape[2] * input_shape[3]));
        _runner.process_async({ input_vector });

        std::vector<std::vector<float>> output;
        _runner.get(output);
        auto& scores = output[0];
        auto& boxes = output[1];

        std::vector<int> indices;
        postprocess(boxes, scores, frame);

        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        latency.push_back(duration.count());
        if (latency.size() > 200)
            latency.pop_front();

        uint64_t temp_tot = 0;
        for (uint64_t i = 0; i < latency.size(); ++i)
        {
            temp_tot += latency[i];
        }

        auto processing_time = temp_tot / latency.size();
        auto fps = 1000 / processing_time;
        
        cv::putText(frame,
            cv::format("Latency: %d ms. %d frames per sec.", processing_time, fps),
            cv::Point(15, 50),
            cv::HersheyFonts::FONT_HERSHEY_COMPLEX,
            0.8,
            cv::Scalar(0, 255, 0), 2);

        cv::namedWindow("Face", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        cv::imshow("Face", frame);
        if (cv::waitKey(1) == 27) break;

        start = end;
    }

    cap.release();
    return sample::gLogger.reportPass(sampleTest);
}

void FaceDetectionApp::print_help_info()
{
    std::cout
        << "Usage: ./FaceDetection [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
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
    params.nContexts = 3;

    return params;
}

cv::Mat FaceDetectionApp::preprocess(const cv::Mat& frame)
{
    cv::Mat out;
    auto input_shape = _runner.get_input_shape(0);
    uint32_t mWidth = input_shape[3];
    uint32_t mHeight = input_shape[2];
    cv::resize(frame, out, cv::Size(mWidth, mHeight));
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    out.convertTo(out, CV_32F);
    cv::Mat output = cv::dnn::blobFromImage(out, 1. / 128, {}, { 127, 127, 127 });
    return output;
}

void FaceDetectionApp::postprocess(const std::vector<float>& boxes, const std::vector<float>& scores, const cv::Mat& frame)
{
    unsigned width = frame.cols;
    unsigned height = frame.rows;

    std::vector<cv::Rect> f_boxes;
    for (uint32_t i = 0; i < boxes.size(); i += 4)
    {
        uint x = static_cast<uint>(boxes[i] * width);
        uint y = static_cast<uint>(boxes[i + 1] * height);
        uint w = static_cast<uint>((boxes[i + 2] - boxes[i]) * width);
        uint h = static_cast<uint>((boxes[i + 3] - boxes[i + 1]) * height);

        f_boxes.emplace_back(x, y, w, h);
    }

    std::vector<std::pair<float, float>> f_scores;
    std::vector<float> label_1_scores, label_2_scores;
    for (uint i = 0; i < scores.size(); i += 2)
    {
        f_scores.emplace_back(scores[i], scores[i + 1]);
    }

    for (const auto& [score1, score2] : f_scores)
    {
        label_1_scores.push_back(score1);
        label_2_scores.push_back(score2);
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(f_boxes, label_2_scores, 0.85, 0.5, nms_result);

    for (int ind : nms_result)
    {
        cv::rectangle(frame, f_boxes[ind], cv::Scalar(255, 0, 0), 2);
    }
}
