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
	std::cout << "Hello, world!" << std::endl;
	return 0;
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
