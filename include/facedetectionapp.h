#ifndef FACE_DETECTION_APP_H
#define FACE_DETECTION_APP_H

#include <iostream>
#include <stdexcept>
#include <chrono>

#include "opencv2/opencv.hpp"

#include "tensorrtrunner.h"
#include "ByteTrack/BYTETracker.h"

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
	void postprocess(const std::vector<float>& boxes, const std::vector<float>& scores, const cv::Mat& frame, 
		std::vector<int>& nms_result, std::vector<cv::Rect>& boxes_result,
		std::vector<std::pair<float, float>>& score_results);

	int _argc;
	char** _argv;

	TensorRTRunner _runner;
	byte_track::BYTETracker tracker;
};

#endif // FACE_DETECTION_APP_H