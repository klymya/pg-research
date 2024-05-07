#include "YOLOv8.h"

YOLOv8::YOLOv8(const string& modelPath, float confThreshold, float nmsThreshold, int height, int width):
	m_confThreshold(confThreshold), m_nmsThreshold(nmsThreshold), m_height(height), m_width(width)
{
	m_net = readNetFromONNX(modelPath);
}

void YOLOv8::Preprocessing(const Mat& image, Mat& blob)
{	
	Mat squaredImage = formatToSquare(image);
	blob = blobFromImage(squaredImage, 1 / 255.0, Size(m_width, m_height), Scalar(0, 0, 0), false, false);
}

void YOLOv8::Inference(Mat& blob, vector<Mat>& output)
{
	m_net.setInput(blob);
	m_net.forward(output, m_net.getUnconnectedOutLayersNames());
}

void YOLOv8::Postprocessing(
	const Mat& image, vector<Mat>& output, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes)
{
	int rows = output[0].size[2];
	int dimensions = output[0].size[1];
	int n_classes = dimensions - 4;
	output[0] = output[0].reshape(1, dimensions);
	cv::transpose(output[0], output[0]);

	float *data = (float *)output[0].data;

	float x_factor = (float)image.cols / m_width;
	float y_factor = (float)image.rows / m_height;

	std::vector<int> rawClassIds;
	std::vector<float> rawConfidences;
	std::vector<cv::Rect> rawBoxes;

	for (int i = 0; i < rows; ++i)
	{
		float *classes_scores = data+4;

		cv::Mat scores(1, n_classes, CV_32FC1, classes_scores);
		cv::Point class_id;
		double maxClassScore;

		minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

		if (maxClassScore > m_confThreshold)
		{
			rawConfidences.push_back(maxClassScore);
			rawClassIds.push_back(class_id.x);

			float x = data[0];
			float y = data[1];
			float w = data[2];
			float h = data[3];

			int left = int((x - 0.5 * w) * x_factor);
			int top = int((y - 0.5 * h) * y_factor);

			int width = int(w * x_factor);
			int height = int(h * y_factor);

			rawBoxes.push_back(cv::Rect(left, top, width, height));
		}

		data += dimensions;
}	

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	dnn::NMSBoxes(rawBoxes, rawConfidences, m_confThreshold, m_nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		boxes.push_back(rawBoxes[idx]);
		confidences.push_back(rawConfidences[idx]);
		classIds.push_back(rawClassIds[idx]);
	}
}

void YOLOv8::Detect(const Mat& bgrImage, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes)
{
	Mat blob;
	vector<Mat> output;

	Preprocessing(bgrImage, blob);
	Inference(blob, output);
	Postprocessing(bgrImage, output, classIds, confidences, boxes);
}

void YOLOv8::DrawPredictions(Mat& image, vector<int>& classIds, vector<float>& confidences, vector<Rect>& boxes)
{
	int baseLine;
	int top;
	Rect* box;
	for (auto i = 0; i < classIds.size(); ++i)
	{
		box = &boxes[i];
		rectangle(image, Point(box->x, box->y), Point(box->x + box->width, box->y + box->height), Scalar(0, 0, 255), 2);

		string label = format("%d: %.2f", classIds[i], confidences[i]);
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(box->y, labelSize.height);
		putText(image, label, Point(boxes[i].x, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

Mat YOLOv8::formatToSquare(const Mat &source)
{
	int col = source.cols;
	int row = source.rows;
	int _max = MAX(col, row);
	cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
	source.copyTo(result(cv::Rect(0, 0, col, row)));
	return result;
}
