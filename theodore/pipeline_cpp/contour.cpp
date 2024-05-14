#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "PANMobilenetV2.h"

using namespace std;
using namespace cv;

void mask2convexHull(const Mat& segMask, vector<Point>& hull)
{
	int k_size = 7;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(k_size, k_size));

	Mat maskBlurred;
	GaussianBlur(segMask, maskBlurred, Size(k_size, k_size), 0);
	cv::imshow("Inference", maskBlurred);
	cv::waitKey(-1);

	// apply morphology
	Mat res_morphology;
	morphologyEx(maskBlurred, res_morphology, MORPH_OPEN, kernel);

	cv::imshow("Inference", res_morphology);
	cv::waitKey(-1);

	// get contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(res_morphology, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_L1);

	//convex hull
	convexHull(contours[0], hull);
}

int main(int argc, char** argv)
{
	if (argc < 3) {
		cout << "image paths not specified\n";
		return 1;
	}

	string imgPath(argv[1]);
	string segModelPath(argv[2]);

	cv::Mat frame = cv::imread(imgPath);

	PANMobilenetV2 segModel(segModelPath);

	Mat segMask;
	segModel.Detect(frame, segMask);

	vector<Point> hull;
	mask2convexHull(segMask, hull);

	vector<vector<Point>> hullContainer {hull};
	drawContours(segMask, hullContainer, 0, Scalar(255), 1, LINE_8);

	cv::imshow("Inference", segMask);
	cv::waitKey(-1);

	return 0;
}