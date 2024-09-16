#ifndef FACE_DETECTION_APP_H
#define FACE_DETECTION_APP_H

#include <iostream>
#include <stdexcept>

#include "tensorrtrunner.h"

class FaceDetectionApp
{
public:
	FaceDetectionApp(int argc, char** argv);
	~FaceDetectionApp();

	int exec();
private:
	void print_help_info();

	int _argc;
	char** _argv;
};

#endif // FACE_DETECTION_APP_H