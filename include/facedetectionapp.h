#ifndef FACE_DETECTION_APP_H
#define FACE_DETECTION_APP_H

#include <iostream>
#include <stdexcept>
#include <chrono>
#include <functional>
#include <cmath>

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
	RunnerParams initializeEmotionsParams();
	RunnerParams initializeGendersParams();
	cv::Mat preprocess(const cv::Mat& frame);
	Tensor preprocess_emotions(const cv::Rect& rect, const cv::Mat& frame);
	Tensor preprocess_gender(const cv::Rect& rect, const cv::Mat& frame);
	std::string postprocess_emotions(Tensor& tensor);
	std::string postprocess_gender(Tensor& tensor);
	void postprocess(const std::vector<float>& boxes, const std::vector<float>& scores, const cv::Mat& frame, 
		std::vector<int>& nms_result, std::vector<cv::Rect>& boxes_result,
		std::vector<std::pair<float, float>>& score_results);

	template <typename It>
	void softmax(It beg, It end)
	{
		using VType = typename std::iterator_traits<It>::value_type;
		using namespace std::placeholders;

		static_assert(std::is_floating_point<VType>::value,
			"Softmax function only applicable for floating types");

		auto max_ele{ *std::max_element(beg, end) };

		std::transform(
			beg,
			end,
			beg,
			[&](VType x) { return std::exp(x - max_ele); });

		VType exptot = std::accumulate(beg, end, 0.0);

		std::transform(
			beg,
			end,
			beg,
			std::bind(std::divides<VType>(), _1, exptot));
	}

	int _argc;
	char** _argv;

	TensorRTRunner _runner;
	TensorRTRunner _emotions;
	TensorRTRunner _gender;
	byte_track::BYTETracker tracker;
};

#endif // FACE_DETECTION_APP_H