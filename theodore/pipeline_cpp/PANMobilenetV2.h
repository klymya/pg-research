#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "IModels.h"

using namespace std;
using namespace cv;
using namespace dnn;


class PANMobilenetV2: ISegModel
{
public:
	PANMobilenetV2(const string& modelPath, int height=128, int width=128);
	void Detect(const Mat& bgrImage, Mat& segMask) override;
	void DrawPredictions(Mat& image, Mat& segMask) override;

private:
	void Preprocessing(const Mat& image, Mat& blob);
	void Inference(Mat& blob, vector<Mat>& output);
	void Postprocessing(const Mat& image, Mat& output);

	Net m_net;
	int m_height;
	int m_width;

	Scalar m_mean {0.485, 0.456, 0.406};
	Scalar m_std {0.229, 0.224, 0.225};

};
