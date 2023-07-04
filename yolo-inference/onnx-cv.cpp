#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <numeric>

using namespace cv;
using namespace std;

float INPUT_WIDTH = 640;
float INPUT_HEIGHT = 640;
float CONFIDENCE_THRESHOLD = 0.4;
float SCORE_THRESHOLD = 0.2;
float NMS_THRESHOLD = 0.45;

int N = 100;

void meanStd(vector<int64_t> v, double &mean, double &stdev)
{
	double sum = std::accumulate(v.begin(), v.end(), 0.0);
	mean = sum / v.size();

	std::vector<double> diff(v.size());
	std::transform(v.begin(), v.end(), diff.begin(),
				std::bind2nd(std::minus<double>(), mean));
	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	stdev = std::sqrt(sq_sum / v.size());
}


void preProcess(const Mat &img, Mat &blob)
{
	cv::dnn::blobFromImage(img, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);
}


void postProcess(
	Mat &input_image,
	vector<Mat> &outputs,
	const vector<string> &class_name,
	vector<int> &class_ids,
	vector<float> &confidences,
	vector<Rect> &boxes)
{
	// Initialize vectors to hold respective outputs while unwrapping detections.
	vector<int> raw_class_ids;
	vector<float> raw_confidences;
	vector<Rect> raw_boxes;
	// Resizing factor.
	float x_factor = input_image.cols / INPUT_WIDTH;
	float y_factor = input_image.rows / INPUT_HEIGHT;
	float *data = (float *)outputs[0].data;
	const int dimensions = 85;
	// 25200 for default size 640.
	const int rows = 25200;
	// Iterate through 25200 detections.
	for (int i = 0; i < rows; ++i)
	{
		float confidence = data[4];
		// Discard bad detections and continue.
		if (confidence >= CONFIDENCE_THRESHOLD)
		{
			float * classes_scores = data + 5;
			// Create a 1x85 Mat and store class scores of 80 classes.
			Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
			// Perform minMaxLoc and acquire the index of best class  score.
			Point class_id;
			double max_class_score;
			minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			// Continue if the class score is above the threshold.
			if (max_class_score > SCORE_THRESHOLD)
			{
				// Store class ID and confidence in the pre-defined respective vectors.
				raw_confidences.push_back(confidence);
				raw_class_ids.push_back(class_id.x);
				// Center.
				float cx = data[0];
				float cy = data[1];
				// Box dimension.
				float w = data[2];
				float h = data[3];
				// Bounding box coordinates.
				int left = int((cx - 0.5 * w) * x_factor);
				int top = int((cy - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);
				// Store good detections in the boxes vector.
				raw_boxes.push_back(Rect(left, top, width, height));
			}
		}
		// Jump to the next row.
		data += 85;
	}
	// Perform Non-Maximum Suppression and draw predictions.
	vector<int> indices;
	cv::dnn::NMSBoxes(raw_boxes, raw_confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		boxes.push_back(raw_boxes[idx]);
		confidences.push_back(raw_confidences[idx]);
		class_ids.push_back(raw_class_ids[idx]);
	}
}

int main(int argc, char** argv )
{
	// Load class list.
	vector<string> class_list;
	std::ifstream ifs("resources/coco.names", ifstream::in);
	string line;
	while (getline(ifs, line))
	{
		class_list.push_back(line);
	}

	Mat img = imread("resources/lena.png");
	if (img.empty())
	{
		cout << "!!! Failed imread(): image not found" << endl;
		// don't let the execution continue, else imshow() will crash.
	}

	// Load model.
    cv::dnn::Net net;
	Mat blob;
	vector<int> class_ids;
	vector<float> confidences;
	vector<Rect> boxes;
	vector<Mat> outputs;

	net = cv::dnn::readNet("resources/yolov5n.onnx");

	// Convert to blob.
	preProcess(img, blob);

	net.setInput(blob);
	// Forward propagate.
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	postProcess(img, outputs, class_list, class_ids, confidences, boxes);

	vector<int64_t> preprocess;
	vector<int64_t> inference;
	vector<int64_t> postprocess;
	for (int i = 0; i < N; ++i)
	{
		auto t0 = chrono::steady_clock::now();
		// Convert to blob.
		preProcess(img, blob);

		auto t1 = chrono::steady_clock::now();
		net.setInput(blob);
		// Forward propagate.
		net.forward(outputs, net.getUnconnectedOutLayersNames());
		auto t2 = chrono::steady_clock::now();

		postProcess(img, outputs, class_list, class_ids, confidences, boxes);
		auto t3 = chrono::steady_clock::now();

		preprocess.push_back(chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
		inference.push_back(chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
		postprocess.push_back(chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count());

		cout << i << ' ';
		cout.flush();
	}
	cout << endl;

	double mean;
	double stdev;
	meanStd(preprocess, mean, stdev);
	cout << "preprocess: " << mean << " " << stdev << endl;

	meanStd(inference, mean, stdev);
	cout << "inference: " << mean << " " << stdev << endl;

	meanStd(postprocess, mean, stdev);
	cout << "postprocess: " << mean << " " << stdev << endl;

	return 0;
}