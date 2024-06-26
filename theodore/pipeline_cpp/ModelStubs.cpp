#include <boost/core/ignore_unused.hpp>

#include "ModelStubs.h"

void ObjDetStub::Detect(const Mat& bgrImage, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes,
						vector<Point>& positions)
{
	boost::ignore_unused(bgrImage);

	classIds.insert(classIds.end(), {0, 1, 2});
	confidences.insert(confidences.end(), {0.9, 0.95, 0.8});
	boxes.insert(boxes.end(), {
		Rect(220, 190, 200, 100),
		Rect(270, 300, 100, 100),
		Rect(320, 240, 50, 30)
	});
	positions.insert(positions.end(), {
		Point(320, 240),
		Point(320, 350),
		Point(345, 255)
	});
}

void ObjDetStub::DrawPredictions(Mat& image, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes,
								 vector<Point>& positions)
{
	boost::ignore_unused(image);
	boost::ignore_unused(classIds);
	boost::ignore_unused(confidences);
	boost::ignore_unused(boxes);
	boost::ignore_unused(positions);
}

void SegStub::Detect(const Mat& bgrImage, Mat& segMask)
{
	int w = bgrImage.cols;
	int h = bgrImage.rows;
	segMask = Mat(bgrImage.size(), CV_8U, Scalar(0));

	Point rook_points[1][4];
	rook_points[0][0] = Point(w / 2, 0);
	rook_points[0][1] = Point(w, h / 2);
	rook_points[0][2] = Point(w / 2, h);
	rook_points[0][3] = Point(0, h / 2);

	const Point* ppt[1] = {rook_points[0]};
	int npt[] = {4};
	
	fillPoly(segMask, ppt, npt, 1, Scalar(1), LINE_8);

}

void SegStub::DrawPredictions(Mat& image, Mat& segMask){
	boost::ignore_unused(image);
	boost::ignore_unused(segMask);
}