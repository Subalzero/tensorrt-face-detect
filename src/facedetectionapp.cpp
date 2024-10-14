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
    _emotions = TensorRTRunner(initializeEmotionsParams());
    _gender = TensorRTRunner(initializeGendersParams());

    sample::gLogInfo << "Building and running a GPU inference engine for Face Detection" << std::endl;

    if (!_runner.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    bool emotions_success = _emotions.build([&](nvinfer1::IOptimizationProfile* profile) {
        nvinfer1::Dims dims = TensorRTRunner::vector_to_dims({ 1, 3, 260, 260 });
        profile->setDimensions("input", OptProfileSelector::kMIN, dims);
        profile->setDimensions("input", OptProfileSelector::kOPT, dims);
        profile->setDimensions("input", OptProfileSelector::kMAX, dims);
        return profile->isValid();
    });

    if (!emotions_success)
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    if (!_gender.build())
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
    auto input_tensor_name = _runner.get_input_tensor_name(0);
    auto input_tensor_data_type = _runner.get_tensor_data_type(input_tensor_name);
    auto data_size = _runner.get_tensor_data_size(input_tensor_data_type);

    std::vector<unsigned char> input_vector(fr.data, fr.data + (input_shape[1] * input_shape[2] * input_shape[3]) * data_size);
    Tensor input_tensor = { input_tensor_name, input_shape, std::move(input_vector), DataType::kFLOAT };
    _runner.process_async({ input_tensor });

    std::deque<uint64_t> latency;
    auto start = std::chrono::system_clock::now();
    while (cap.isOpened())
    {
        cv::Mat frame;
        cap >> frame;
        cv::Mat fr = preprocess(frame);
        auto input_shape = _runner.get_input_shape(0);

        std::vector<unsigned char> input_vector(fr.data, fr.data + (input_shape[1] * input_shape[2] * input_shape[3]) * data_size);
        Tensor input_tensor = { input_tensor_name, input_shape, std::move(input_vector), DataType::kFLOAT };
        _runner.process_async({ input_tensor });

        std::vector<Tensor> output;
        _runner.get(output);
        // Scores total size
        size_t score_tot_size = 1;
        for (int i = 0; i < output[0].shape.size(); ++i)
        {
            score_tot_size *= output[0].shape[i];
        }
        size_t box_tot_size = 1;
        for (int i = 0; i < output[1].shape.size(); ++i)
        {
            box_tot_size *= output[1].shape[i];
        }
        auto scores = std::vector<float>(reinterpret_cast<float*>(output[0].data.data()),
            reinterpret_cast<float*>(output[0].data.data()) + score_tot_size);
        auto boxes = std::vector<float>(reinterpret_cast<float*>(output[1].data.data()),
            reinterpret_cast<float*>(output[1].data.data()) + box_tot_size);

        std::vector<int> nms_result;
        std::vector<cv::Rect> f_boxes;
        std::vector<std::pair<float, float>> f_scores;
        postprocess(boxes, scores, frame, nms_result, f_boxes, f_scores);

        std::vector<byte_track::Object> objects;
        for (int ind : nms_result)
        {
            auto& box = f_boxes[ind];
            auto& score = f_scores[ind];
            objects.emplace_back(byte_track::Rect<float>(box.x, box.y, box.width, box.height), 0, score.second);
            // cv::rectangle(frame, f_boxes[ind], cv::Scalar(0, 255, 0), 2);
        }
        auto tracklets = tracker.update(objects);

        for (auto& tracklet : tracklets)
        {
            auto box = tracklet->getRect();
            auto id = tracklet->getTrackId();

            int bwidth = (box.width() + box.x() > frame.cols) ?
                box.width() - (box.width() + box.x() - frame.cols) : box.width();
            int bheight = (box.height() + box.y() > frame.rows) ?
                box.height() - (box.height() + box.y() - frame.rows) : box.height();

            cv::Rect rect = cv::Rect(std::floorf(box.x()), 
                std::floorf(box.y()), bwidth, bheight);
            Tensor emotions_input = preprocess_emotions(rect, frame);
            Tensor gender_input = preprocess_gender(rect, frame);

            _emotions.process_async({ emotions_input });
            _gender.process_async({ gender_input });

            std::vector<Tensor> emotions_outputs, gender_outputs;

            _emotions.get(emotions_outputs);
            _gender.get(gender_outputs);

            std::string emotion = postprocess_emotions(emotions_outputs[0]);
            std::string gender = postprocess_gender(gender_outputs[0]);

            cv::rectangle(frame, cv::Rect(box.x(), box.y(), box.width(), box.height()), cv::Scalar(0, 255, 0), 3);
            cv::putText(frame,
                cv::format("%d", id),
                cv::Point(box.x(), box.y()),
                cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8,
                cv::Scalar(0, 0, 255), 2);
            cv::putText(frame,
                emotion,
                cv::Point(box.x() + 10, box.y() + 20),
                cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8,
                cv::Scalar(0, 0, 255), 2);
            cv::putText(frame,
                gender,
                cv::Point(box.x() + 10, box.y() + 40),
                cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8,
                cv::Scalar(0, 0, 255), 2);
        }

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
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.bf16 = args.runInBf16;
    params.timingCacheFile = args.timingCacheFile;
    params.nContexts = 2;

    return params;
}

RunnerParams FaceDetectionApp::initializeEmotionsParams()
{
    RunnerParams params;
    params.dataDirs.push_back("models");

    params.onnxFileName = "emotion.onnx";
    params.int8 = true;
    params.fp16 = true;
    params.bf16 = false;
    params.nContexts = 1;

    return params;
}

RunnerParams FaceDetectionApp::initializeGendersParams()
{
    RunnerParams params;
    params.dataDirs.push_back("models");

    params.onnxFileName = "gender_googlenet.onnx";
    params.int8 = true;
    params.fp16 = true;
    params.bf16 = false;
    params.nContexts = 1;

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

Tensor FaceDetectionApp::preprocess_emotions(const cv::Rect& rect, const cv::Mat& frame)
{
    cv::Mat face = frame(cv::Range(rect.y, rect.y + rect.height), cv::Range(rect.x, rect.x + rect.width));
    cv::Mat resized;
    cv::Scalar mean(0.485, 0.456, 0.406), std(0.229, 0.224, 0.225);
    cv::resize(face, resized, {260, 260});
    //cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F);
    /*resized = resized / 255.;

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    for (int i = 0; i < 3; ++i)
    {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }

    cv::merge(channels, resized);*/

    resized = cv::dnn::blobFromImage(resized, 1.);
    Tensor tensor;
    tensor.name = "input";
    tensor.shape = { 1, 3, 260, 260 };
    tensor.data = std::vector<unsigned char>(resized.data, resized.data + (260 * 260 * 3) * sizeof(float));
    tensor.data_type = DataType::kFLOAT;
    return tensor;
}

Tensor FaceDetectionApp::preprocess_gender(const cv::Rect& rect, const cv::Mat& frame)
{
    cv::Mat face = frame(cv::Range(rect.y, rect.y + rect.height), cv::Range(rect.x, rect.x + rect.width));
    cv::Mat resized;
    cv::Scalar mean(104, 117, 123);
    cv::resize(face, resized, { 224, 224 });
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    for (int i = 0; i < 3; ++i)
    {
        channels[i] = (channels[i] - mean[i]);
    }

    cv::merge(channels, resized);
    resized.convertTo(resized, CV_32F);

    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.);

    std::string input_name = _gender.get_input_tensor_name(0);
    auto input_shape = _gender.get_input_shape(0);

    Tensor tensor;
    tensor.name = input_name;
    tensor.shape = input_shape;
    tensor.data = std::vector<unsigned char>(blob.data, 
        blob.data + std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>()) * sizeof(float));
    tensor.data_type = DataType::kFLOAT;
    return tensor;
}

std::string FaceDetectionApp::postprocess_emotions(Tensor& tensor)
{
    std::vector<float> data(reinterpret_cast<float*>(tensor.data.data()),
        reinterpret_cast<float*>(tensor.data.data()) + 7);
    //softmax(data.begin(), data.end());
    std::vector<std::string> emotions = { "anger", "disgust", "fear", "happy", "neutral", "sad", "surprise" };
    auto iter_max = std::max_element(data.begin(), data.end());
    int index = std::distance(data.begin(), iter_max);
    return emotions[index];
}

std::string FaceDetectionApp::postprocess_gender(Tensor& tensor)
{
    std::vector<float> data(reinterpret_cast<float*>(tensor.data.data()),
        reinterpret_cast<float*>(tensor.data.data()) + 2);
    //softmax(data.begin(), data.end());
    std::vector<std::string> genders = { "male", "female" };
    auto iter_max = std::max_element(data.begin(), data.end());
    int index = std::distance(data.begin(), iter_max);
    return genders[index];
}

void FaceDetectionApp::postprocess(
    const std::vector<float>& boxes,
    const std::vector<float>& scores,
    const cv::Mat& frame,
    std::vector<int>& nms_result,
    std::vector<cv::Rect>& boxes_result,
    std::vector<std::pair<float, float>>& score_results
)
{
    unsigned width = frame.cols;
    unsigned height = frame.rows;
    uint padding = 60;

    float ratio = (float)width / height;

    std::vector<cv::Rect> f_boxes;
    for (uint32_t i = 0; i < boxes.size(); i += 4)
    {
        uint x = static_cast<uint>(boxes[i] * width);
        uint y = static_cast<uint>(boxes[i + 1] * height);
        uint w = static_cast<uint>((boxes[i + 2] - boxes[i]) * width);
        uint h = static_cast<uint>((boxes[i + 3] - boxes[i + 1]) * height);

        x = x - padding * ratio;
        if (x < 0) x = 0;
        y = y - padding * ratio;
        if (y < 0) y = 0;

        w = w + padding * 2 * ratio;
        if (width < x + w) w = w - (x + w - width);
        h = h + padding * 2 * ratio;
        if (height < y + h) h = h - (h + y - height);

        assert(x >= 0);
        assert(y >= 0);
        assert(width >= x + w);
        assert(height >= y + h);

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

    cv::dnn::NMSBoxes(f_boxes, label_2_scores, 0.85, 0.5, nms_result);
    boxes_result = std::move(f_boxes);
    score_results = std::move(f_scores);
}
