#pragma once

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class IObjDetModel
{
public:
	virtual void Detect(const Mat& bgrImage, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes,
						vector<Point>& positions) = 0;
	virtual void DrawPredictions(Mat& image, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes,
								 vector<Point>& positions) = 0;
};

class ISegModel
{
public:
	virtual void Detect(const Mat& bgrImage, Mat& segMask) = 0;
	virtual void DrawPredictions(Mat& image, Mat& segMask) = 0;
};