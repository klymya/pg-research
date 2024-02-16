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
#include "tensorflow/lite/version.h"

using namespace std;
using namespace cv;


#define TFLITE_MINIMAL_CHECK(x)										\
	if (!(x)) {														\
		fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__);	\
		exit(1);													\
	}

int N = 1;

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

int main(int argc, char* argv[]) {

	// Print TensorFlow Lite version
    std::cout << "TensorFlow Lite version: " << TFLITE_VERSION_STRING << std::endl;

	string model_pth;
	string head_pth;
	vector<string> model_pths {
		"yolov8/weights/quant_1_1m/model_256.tflite", "yolov8/weights/quant_1_1m/model_320.tflite",
		"yolov8/weights/quant_1_1m/model_640.tflite"
	};
	vector<string> head_pths {
		"yolov8/weights/quant_1_1m/head_640.tflite", "yolov8/weights/quant_1_1m/head_640.tflite",
		"yolov8/weights/quant_1_1m/head_640.tflite"
	};
	int nthreads = 1;

	float modelScoreThreshold = 0.25;
	float modelNMSThreshold = 0.5;

	if (argc > 1)
	{
		nthreads = atoi(argv[1]);
	}
	if (nthreads == 0)
	{
		nthreads = 1;
	}
	cout << nthreads << endl;

	for (int i = 0; i < model_pths.size(); i++)
	{
		model_pth = model_pths[i];
		head_pth = head_pths[i];

		// Load model
		std::unique_ptr<tflite::FlatBufferModel> model =
			tflite::FlatBufferModel::BuildFromFile(model_pth.c_str());
		TFLITE_MINIMAL_CHECK(model != nullptr);

		// Load model
		std::unique_ptr<tflite::FlatBufferModel> head =
			tflite::FlatBufferModel::BuildFromFile(head_pth.c_str());
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

		tflite::ops::builtin::BuiltinOpResolver resolverHead;
		tflite::InterpreterBuilder builderHead(*head, resolverHead);
		std::unique_ptr<tflite::Interpreter> interpreterHead;
		builderHead(&interpreterHead);
		TFLITE_MINIMAL_CHECK(interpreterHead != nullptr);

		interpreter->SetNumThreads(nthreads);
		interpreterHead->SetNumThreads(nthreads);

		// Allocate tensor buffers.
		TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
		TFLITE_MINIMAL_CHECK(interpreterHead->AllocateTensors() == kTfLiteOk);

		// get input & output layer
		TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);

		const uint HEIGHT = input_tensor->dims->data[1];
		const uint WIDTH = input_tensor->dims->data[2];
		const uint CHANNEL = input_tensor->dims->data[3];

		Mat img(640, 480, CV_8UC3);

		Mat imgResized;
		cv::resize(img, imgResized, cv::Size(WIDTH, HEIGHT), 0, 0, INTER_LINEAR);

		memcpy(input_tensor->data.uint8, imgResized.ptr<uint8_t>(0), WIDTH * HEIGHT * CHANNEL * sizeof(uint8_t));

		TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

		int n = 7;
		float scale;
		int zero_point;
		int count;

		TfLiteTensor* output_tensor;
		for (int i = 0; i < n; i++)
		{
			output_tensor = interpreter->tensor(interpreter->outputs()[i]);
			input_tensor = interpreterHead->tensor(interpreterHead->inputs()[i]);
			scale = output_tensor->params.scale;
			zero_point = output_tensor->params.zero_point;
			
			for(int j = 0; j < output_tensor->bytes; j++){
				input_tensor->data.f[j] = ((float)output_tensor->data.uint8[j] - zero_point) * scale;
			}

		}

		TFLITE_MINIMAL_CHECK(interpreterHead->Invoke() == kTfLiteOk);

		vector<int64_t> inference;
		for (int k = 0; k < N; ++k)
		{
			auto t0 = std::chrono::steady_clock::now();

			cv::resize(img, imgResized, cv::Size(WIDTH, HEIGHT), 0, 0, INTER_LINEAR);

			memcpy(input_tensor->data.uint8, imgResized.ptr<uint8_t>(0), WIDTH * HEIGHT * CHANNEL * sizeof(uint8_t));

			TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

			for (int i = 0; i < n; i++)
			{
				output_tensor = interpreter->tensor(interpreter->outputs()[i]);
				input_tensor = interpreterHead->tensor(interpreterHead->inputs()[i]);
				scale = output_tensor->params.scale;
				zero_point = output_tensor->params.zero_point;
				
				for(int j = 0; j < output_tensor->bytes; j++){
					input_tensor->data.f[j] = ((float)output_tensor->data.uint8[j] - zero_point) * scale;
				}

			}

			TFLITE_MINIMAL_CHECK(interpreterHead->Invoke() == kTfLiteOk);

			output_tensor = interpreterHead->tensor(interpreterHead->outputs()[0]);

			float *data = (float *)output_tensor->data.f;

			std::vector<int> class_ids;
			std::vector<float> confidences;
			std::vector<Rect> boxes;
			vector<Point> positions;

			int rows = output_tensor->dims->data[1];
			int dimensions = output_tensor->dims->data[2];

			for (int i = 0; i < rows; ++i)
			{
				vector<float> classes_scores;
				classes_scores.push_back(data[4]);

				cv::Point class_id;
				double maxClassScore;
				minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

				if (maxClassScore > modelScoreThreshold)
				{
					confidences.push_back(maxClassScore);
					class_ids.push_back(class_id.x);

					float x = data[0];
					float y = data[1];
					float w = data[2];
					float h = data[3];

					int left = int(x - 0.5 * w);
					int top = int(y - 0.5 * h);

					int width = int(w);
					int height = int(h);

					boxes.push_back(cv::Rect(left, top, width, height));
					positions.push_back(cv::Point(int(data[5]), int(data[6])));
				}

				data += dimensions;
			}
			std::vector<int> nms_result;
			cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

			auto t1 = std::chrono::steady_clock::now();
			inference.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
		}

		double mean;
		double stdev;
		meanStd(inference, mean, stdev);
		cout << model_pth << ": " << mean << " " << stdev  << " " << 1000 / mean << endl;
	}

	// cout << nms_result.size() << endl;
	// for (unsigned long i = 0; i < nms_result.size(); ++i)
	// {
	//     int idx = nms_result[i];

	//     cout << class_ids[idx] << " " << confidences[idx] << " " << boxes[idx]  << " " << positions[idx] << endl;
	// }

	return 0;
}