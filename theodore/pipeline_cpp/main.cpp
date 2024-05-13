#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "PANMobilenetV2.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc != 3) {
		cout << "the model and image paths not specified\n";
		return 1;
	}

	string modelPath(argv[1]);
	string imgPath(argv[2]);

	cv::Mat frame = cv::imread(imgPath);

	PANMobilenetV2 model(modelPath);

	Mat segMask;
	model.Detect(frame, segMask);

	cv::Mat channelScore = cv::Mat(128, 128, CV_32FC1, const_cast<float *>(segMask.ptr<float>(0, 0, 0)));
	Mat output;
	channelScore.convertTo(output, CV_8U, 255);

	cv::imshow("Inference", frame);
	cv::waitKey(-1);
	cv::imshow("Inference", channelScore);
	cv::waitKey(-1);

	return 0;
}