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
	cv::Mat channelScore = cv::Mat(m_width, m_height, CV_32FC1, const_cast<float *>(output.ptr<float>(0, 0, 0)));
	channelScore.convertTo(output, CV_8U, 255.);
	resize(output, output, image.size(), 0, 0, INTER_LINEAR_EXACT);
}

void PANMobilenetV2::Detect(const Mat& bgrImage, Mat& segMask)
{
	Mat blob;
	vector<Mat> output;

	Preprocessing(bgrImage, blob);
	Inference(blob, output);
	segMask = output[0];
	Postprocessing(bgrImage, segMask);
}

void PANMobilenetV2::DrawPredictions(Mat& image, Mat& segMask)
{
	Mat segMaskBGR;
	cvtColor(segMask, segMaskBGR, COLOR_GRAY2BGR);
	float alpha = 0.6;
	float beta = ( 1.0 - alpha );
	addWeighted(image, alpha, segMaskBGR, beta, 0.0, image);
}