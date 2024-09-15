#ifndef FACE_DETECTION_APP_H
#define FACE_DETECTION_APP_H

#include <iostream>

#include "tensorrtrunner.h"

class FaceDetectionApp
{
public:
	FaceDetectionApp(int argc, char** argv);
	~FaceDetectionApp();

	int exec();
};

#endif // FACE_DETECTION_APP_H