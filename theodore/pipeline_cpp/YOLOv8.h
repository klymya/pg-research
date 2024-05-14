#pragma once

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "IModels.h"

using namespace std;
using namespace cv;
using namespace dnn;

class YOLOv8: IObjDetModel
{
public:
	YOLOv8(const string& modelPath, float confThreshold=0.01, float nmsThreshold=0.5, int height=640, int width=640);
	void Detect(const Mat& bgrImage, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) override;
	void DrawPredictions(Mat& image, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes) override;

private:
	Mat formatToSquare(const Mat &source);
	void Preprocessing(const Mat& image, Mat& blob);
	void Inference(Mat& blob, vector<Mat>& output);
	void Postprocessing(
		const Mat& image, vector<Mat>& output, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes);

	Net m_net;
	float m_confThreshold;
	float m_nmsThreshold;
	int m_height;
	int m_width;
};
