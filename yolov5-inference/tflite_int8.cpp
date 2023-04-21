/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <cstdlib>

#include "opencv2/opencv.hpp"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

using namespace std;
using namespace cv;

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

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)										\
	if (!(x)) {														\
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);	\
		exit(1);													\
	}


auto matPreprocess(cv::Mat src, uint width, uint height) -> cv::Mat
{
	// convert to float; BGR -> RGB
	cv::Mat dst;
	src.convertTo(dst, CV_8U);
	cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

	// resize image as model input
	cv::resize(dst, dst, cv::Size(width, height));
	return dst;
}

vector<vector<float>> tensorToVector2D(TfLiteTensor *pOutputTensor)
{
	const int row = pOutputTensor->dims->data[1];
	const int colum = pOutputTensor->dims->data[2];
	auto scale = pOutputTensor->params.scale;
	auto zero_point = pOutputTensor->params.zero_point;
	vector<vector<float>> v;
	for (int32_t i = 0; i < row; i++)
	{
		vector<float> _tem;
		for (int j = 0; j < colum; j++)
		{
			float val_float = (((int32_t)pOutputTensor->data.uint8[i * colum + j]) - zero_point) * scale;
			_tem.push_back(val_float);
		}
		v.push_back(_tem);
	}
	return v;
}


void nonMaximumSupprition(
	std::vector<std::vector<float>> &predV,
	const int &row,
	const int &colum,
	std::vector<cv::Rect> &boxes,
	std::vector<float> &confidences,
	std::vector<int> &classIds,
	std::vector<int> &indices,
	cv::Mat &img)

{

	// thresh hold
	float confThreshold = 0.2;
	float nmsThreshold = 0.45;

	int _img_width = img.cols;
	int _img_height = img.rows;

	std::vector<cv::Rect> boxesNMS;
	int max_wh = 40960;
	std::vector<float> scores;
	double confidence;
	cv::Point classId;

	for (int i = 0; i < row; i++)
	{
		if (predV[i][4] > confThreshold)
		{
			// height--> image.rows,  width--> image.cols;
			int left = (predV[i][0] - predV[i][2] / 2) * _img_width;
			int top = (predV[i][1] - predV[i][3] / 2) * _img_height;
			int w = predV[i][2] * _img_width;
			int h = predV[i][3] * _img_height;

			for (int j = 5; j < colum; j++)
			{
				// # conf = obj_conf * cls_conf
				scores.push_back(predV[i][j] * predV[i][4]);
			}

			cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
			scores.clear();
			int c = classId.x * max_wh;
			if (confidence > confThreshold)
			{
				boxes.push_back(cv::Rect(left, top, w, h));
				confidences.push_back(confidence);
				classIds.push_back(classId.x);
				boxesNMS.push_back(cv::Rect(left, top, w, h));
			}
		}
	}
	cv::dnn::NMSBoxes(boxesNMS, confidences, confThreshold, nmsThreshold, indices);
}


int main(int argc, char* argv[]) {
	const char* filename = "resources/yolov5n-int8.tflite";
	const char* imgpath = "resources/lena.png";
	int nthreads = 1;

	if (argc > 1)
	{
		nthreads = atoi(argv[1]);
	}
	if (nthreads == 0)
	{
		nthreads = 1;
	}
	cout << nthreads << endl;

	// Load model
	std::unique_ptr<tflite::FlatBufferModel> model =
		tflite::FlatBufferModel::BuildFromFile(filename);
	TFLITE_MINIMAL_CHECK(model != nullptr);

	// Build the interpreter with the InterpreterBuilder.
	// Note: all Interpreters should be built with the InterpreterBuilder,
	// which allocates memory for the Interpreter and does various set up
	// tasks so that the Interpreter can read the provided model.
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*model, resolver);
	std::unique_ptr<tflite::Interpreter> interpreter;
	builder(&interpreter);
	TFLITE_MINIMAL_CHECK(interpreter != nullptr);

	interpreter->SetNumThreads(nthreads);

	// Allocate tensor buffers.
	TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

	// get input & output layer
	// shape 40 640 3
	TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
	// shape 25200 85
	TfLiteTensor* output_tensor = interpreter->tensor(interpreter->outputs()[0]);

	const uint HEIGHT = input_tensor->dims->data[1];
	const uint WIDTH = input_tensor->dims->data[2];
	const uint CHANNEL = input_tensor->dims->data[3];

	// std::cout<< HEIGHT << " " << WIDTH << " " << CHANNEL << std::endl;
	// std::cout<< output_tensor->dims->data[1] << " " << output_tensor->dims->data[2] << std::endl;

	// read image file
	cv::Mat img = cv::imread(imgpath);

	vector<int64_t> preprocess;
	vector<int64_t> inference;
	vector<int64_t> postprocess;
	for (int i = 0; i < N; ++i)
	{
		auto t0 = std::chrono::steady_clock::now();
		cv::Mat inputImg = matPreprocess(img, WIDTH, HEIGHT);
		// flatten rgb image to input layer.
		uint8_t* inputImg_ptr = inputImg.ptr<uint8_t>(0);
		memcpy(input_tensor->data.f, inputImg.ptr<uint8_t>(0),
			WIDTH * HEIGHT * CHANNEL * sizeof(uint8_t));
		auto t1 = std::chrono::steady_clock::now();

		// Run inference
		TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
		auto t2 = std::chrono::steady_clock::now();

		// vector<float> output_vec = cvtTensor(output_tensor);
		vector<vector<float>> output_vec = tensorToVector2D(output_tensor);

		const int row = output_tensor->dims->data[1];
		const int colum = output_tensor->dims->data[2];

		std::vector<int> indices;
		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;

		nonMaximumSupprition(output_vec, row, colum, boxes, confidences, classIds, indices, img);

		vector<cv::Rect> filteredBoxes;
		vector<float> filteredScores;
		vector<float> filteredLabels;
		
		for (int i = 0; i < indices.size(); i++)
		{
			filteredBoxes.push_back(boxes[indices[i]]);
			filteredScores.push_back(confidences[indices[i]]);
			filteredLabels.push_back(classIds[indices[i]]);
		}
		auto t3 = std::chrono::steady_clock::now();

		preprocess.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
		inference.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
		postprocess.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count());
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