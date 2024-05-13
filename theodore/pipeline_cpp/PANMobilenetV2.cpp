#include "PANMobilenetV2.h"

PANMobilenetV2::PANMobilenetV2(const string& modelPath, int height, int width): m_height(height), m_width(width)
{
	m_net = readNetFromONNX(modelPath);
}

void PANMobilenetV2::Preprocessing(const Mat& image, Mat& blob)
{	
	vector<Mat> channels;
	vector<Mat> normChannels;
    split(image, channels);
	for (int i = 0; i < 3; i++)
	{
		Mat normChannel;
		channels[i].convertTo(normChannel, CV_32F, 1 / 255.0);
		normChannel = (normChannel - m_mean[i]) / m_std[i];

		normChannels.push_back(normChannel);
	}
	Mat normImg;
    merge(normChannels, normImg);

	blob = blobFromImage(normImg, 1, Size(m_width, m_height), Scalar(0, 0, 0), true, false);

}

void PANMobilenetV2::Inference(Mat& blob, vector<Mat>& output)
{
	m_net.setInput(blob);
	m_net.forward(output, m_net.getUnconnectedOutLayersNames());
}

void PANMobilenetV2::Postprocessing(const Mat& image, Mat& output)
{

}

void PANMobilenetV2::Detect(const Mat& bgrImage, Mat& segMask)
{
	Mat blob;
	vector<Mat> output;

	Preprocessing(bgrImage, blob);
	Inference(blob, output);
	segMask = output[0];
	// Postprocessing(bgrImage, output);
}

void PANMobilenetV2::DrawPredictions(Mat& image, Mat& segMask)
{

}