#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "PANMobilenetV2.h"
#include "YOLOv8.h"
#include "mask2ConvexHul.h"

using namespace std;
using namespace cv;

const int MARGIN = 5;
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

int main(int argc, char** argv)
{
	if (argc < 4) {
		cout << "the parameters not specified\n";
		return 1;
	}

	string detModelPath(argv[1]);
	string segModelPath(argv[2]);
	string imgPath(argv[3]);

	// init
	cv::Mat frame = cv::imread(imgPath);
	
	YOLOv8 detModel(detModelPath);
	PANMobilenetV2 segModel(segModelPath);

	// detection
	vector<int> allClassIds;
	vector<float> allConfidences;
	vector<Rect> allBoxes;
	detModel.Detect(frame, allClassIds, allConfidences, allBoxes);

	Mat segMask;
	segModel.Detect(frame, segMask);

	

	vector<string> classes {"person", "chair","table", "armchair", "tv","wheel walker"};
	vector<int> classesOfInterest {1, 2, 3, 4};
	int n_classes = classes.size();
	int table_idx = 2;

	vector<int64_t> inference;
	for (auto i = 0; i < N; i++)
	{
		auto t0 = std::chrono::steady_clock::now();
		// detection
		allClassIds.clear();
		allConfidences.clear();
		allBoxes.clear();
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

			segModel.Detect(roiImg, segMask);

			vector<Point> hull;
			mask2convexHull(segMask, hull);
		}

		auto t1 = std::chrono::steady_clock::now();
		inference.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
	}

	double mean;
	double stdev;
	meanStd(inference, mean, stdev);
	cout << mean << " " << stdev << endl;

	return 0;
}