#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "facedetectionapp.h"

int main(int argc, char** argv)
{
	FaceDetectionApp app(argc, argv);
	return app.exec();
}