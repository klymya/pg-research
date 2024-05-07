#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "YOLOv8.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	if (argc != 3) {
		cout << "the model and image paths not specified\n";
		return 1;
	}

	string modelPath(argv[1]);
	string imgPath(argv[2]);

	cout << modelPath << '\n';

	vector<string> classes {"person", "chair","table", "armchair", "tv","wheel walker"};
	int n_classes = classes.size();

	cv::Mat frame = cv::imread(imgPath);

	YOLOv8 model(modelPath, 0.25);

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	model.Detect(frame, classIds, confidences, boxes);

	for (auto i = 0; i < classIds.size(); i++){
		cout << classIds[i] << " " << confidences[i] << " " << boxes[i] << endl;
	}

	model.DrawPredictions(frame, classIds, confidences, boxes);

	cv::imshow("Inference", frame);
	cv::waitKey(-1);
	return 0;
}