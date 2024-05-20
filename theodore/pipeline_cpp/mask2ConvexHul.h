#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void mask2convexHull(const Mat& segMask, vector<Point>& hull)
{
	Mat maskBlurred;
	GaussianBlur(segMask, maskBlurred, Size(7, 7), 0);

	// apply morphology
	int k_size = 11;
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(k_size, k_size));
	Mat res_morphology;
	morphologyEx(maskBlurred, res_morphology, MORPH_OPEN, kernel);

	// get contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(res_morphology, contours, hierarchy, RETR_TREE, CHAIN_APPROX_TC89_L1);

	//convex hull
	vector<Point> contour;
	for (auto i: contours)
	{
		contour.insert(contour.end(), i.begin(), i.end());
	}
	convexHull(contour, hull);

}
