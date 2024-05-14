#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "PANMobilenetV2.h"
#include "YOLOv8.h"
#include "mask2ConvexHul.h"

using namespace std;
using namespace cv;

const int MARGIN = 5;

int main(int argc, char** argv)
{
	if (argc < 4) {
		cout << "the models and image paths not specified\n";
		return 1;
	}

	string detModelPath(argv[1]);
	string segModelPath(argv[2]);
	string imgPath(argv[3]);

	// init
	cv::Mat frame = cv::imread(imgPath);
	
	YOLOv8 detModel(detModelPath);
	PANMobilenetV2 segModel(segModelPath);

	vector<string> classes {"person", "chair","table", "armchair", "tv","wheel walker"};
	vector<int> classesOfInterest {1, 2, 3, 4};
	int n_classes = classes.size();
	int table_idx = 2;

	// detection
	vector<int> allClassIds;
	vector<float> allConfidences;
	vector<Rect> allBoxes;
	detModel.Detect(frame, allClassIds, allConfidences, allBoxes);

	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	for (auto i = 0; i < allClassIds.size(); i++)
	{
		if (std::find(classesOfInterest.begin(), classesOfInterest.end(), allClassIds[i]) == classesOfInterest.end())
		{
			continue;
		}
		classIds.push_back(allClassIds[i]);
		confidences.push_back(allConfidences[i]);
		boxes.push_back(allBoxes[i]);
	}

	// segmentation
	Mat roiImg;
	Rect roi;
	for (auto i = 0; i < classIds.size(); i++)
	{
		if (classIds[i] != table_idx) continue;

		roi = Rect(boxes[i].x - MARGIN, boxes[i].y - MARGIN, boxes[i].width + 2 * MARGIN, boxes[i].height + 2 * MARGIN);
		roiImg = frame(roi);
		Mat segMask;
		segModel.Detect(roiImg, segMask);

		segModel.DrawPredictions(roiImg, segMask);

		vector<Point> hull;
		mask2convexHull(segMask, hull);

		vector<vector<Point>> hullContainer {hull};
		drawContours(roiImg, hullContainer, 0, Scalar(0, 255, 0), 2, LINE_8);
	}

	detModel.DrawPredictions(frame, classIds, confidences, boxes);

	cv::imshow("Inference", frame);
	cv::waitKey(-1);

	return 0;
}