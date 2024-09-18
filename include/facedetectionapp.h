#ifndef FACE_DETECTION_APP_H
#define FACE_DETECTION_APP_H

#include <iostream>
#include <stdexcept>

#include "opencv2/opencv.hpp"

#include "tensorrtrunner.h"

class FaceDetectionApp
{
public:
	FaceDetectionApp(int argc, char** argv);
	~FaceDetectionApp();

	int exec();
private:
	void print_help_info();
	RunnerParams initializeParams(const samplesCommon::Args& args);
	cv::Mat preprocess(const cv::Mat& frame);
	void postprocess(const std::vector<float>& boxes, const std::vector<float>& scores, const cv::Mat& frame);

	int _argc;
	char** _argv;

	TensorRTRunner _runner;
};

#endif // FACE_DETECTION_APP_H