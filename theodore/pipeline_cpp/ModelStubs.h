#pragma once

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "IModels.h"

using namespace std;
using namespace cv;

class ObjDetStub: public IObjDetModel
{
public:
	void Detect(const Mat& bgrImage, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes,
				vector<Point>& positions);
	void DrawPredictions(Mat& image, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes,
						 vector<Point>& positions);
};

class SegStub: public ISegModel
{
public:
	void Detect(const Mat& bgrImage, Mat& segMask);
	void DrawPredictions(Mat& image, Mat& segMask);
};
