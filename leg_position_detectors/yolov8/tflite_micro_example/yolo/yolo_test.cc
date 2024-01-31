/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>
#include <iostream>
#include <fstream>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/yolo/models/yolov8_256_int8_model_data.h"
#include "tensorflow/lite/micro/examples/yolo/models/head_256_float32_model_data.h"
#include "tensorflow/lite/micro/examples/yolo/testdata/img_256_image_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

using namespace std;

using YOLOv8OpResolver = tflite::MicroMutableOpResolver<11>;

// constexpr int tensor_arena_size
constexpr int tensor_arena_size = 900000;
uint8_t tensor_arena[tensor_arena_size];

TfLiteStatus RegisterOps(YOLOv8OpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddResizeNearestNeighbor());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  return kTfLiteOk;
}

using HEADOpResolver = tflite::MicroMutableOpResolver<10>;

constexpr int head_tensor_arena_size = 900000;
uint8_t head_tensor_arena[head_tensor_arena_size];

TfLiteStatus RegisterOpsHead(HEADOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSub());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());

  return kTfLiteOk;
}

void save2file(int count,int8_t* data, string file_path) {
  ofstream f(file_path);
  for(int i = 0; i < count; i ++){
      f << (uint8_t)data[i];
  }
  f.close();
}

void save2file(int count,float* data, string file_path) {
  ofstream f(file_path);
  for(int i = 0; i < count; i ++){
      f << (float)data[i] << " ";
  }
  f.close();
}

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();

  const tflite::Model* model = ::tflite::GetModel(g_yolov8_256_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
  }

  YOLOv8OpResolver op_resolver;
  RegisterOps(op_resolver);

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       tensor_arena_size);
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);
  
  TFLITE_CHECK(input != nullptr);
  TFLITE_CHECK_EQ(4, input->dims->size);
  TFLITE_CHECK_EQ(1, input->dims->data[0]);
  TFLITE_CHECK_EQ(256, input->dims->data[1]);
  TFLITE_CHECK_EQ(256, input->dims->data[2]);
  TFLITE_CHECK_EQ(3, input->dims->data[3]);

  memcpy(input->data.int8, g_img_256_image_data, input->bytes);

  save2file(
    input->dims->data[0] * input->dims->data[1] * input->dims->data[2] * input->dims->data[3],
    input->data.int8,
    "/home/kyamkovyi/projects/tflite-micro/input"
  );

  // Run the model on this input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
  }

/////////////////////////////////////////////////////////////////////////////////////////////////
  const tflite::Model* model_head = ::tflite::GetModel(g_head_256_float32_model_data);
  if (model_head->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model_head->version(), TFLITE_SCHEMA_VERSION);
  }

  HEADOpResolver op_resolver_head;
  RegisterOpsHead(op_resolver_head);

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter_head(model_head, op_resolver_head, head_tensor_arena,
                                       head_tensor_arena_size);
  interpreter_head.AllocateTensors();

  int n = 7;
  float scale;
  int zero_point;
  int count;
  for (int i = 0; i < n; i++)
  {
    scale = interpreter.output(i)->params.scale;
    zero_point = interpreter.output(i)->params.zero_point;
    count = interpreter.output(i)->dims->data[0] *
      interpreter.output(i)->dims->data[1] *
      interpreter.output(i)->dims->data[2];
    for(int j = 0; j < count; j++){
      interpreter_head.input(i)->data.f[j] = 
        ((float)tflite::GetTensorData<uint8_t>(interpreter.output(i))[j] - zero_point) * scale;
    }
  }

  // Run the model on this input and make sure it succeeds.
  invoke_status = interpreter_head.Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed\n");
  }

  save2file(
    interpreter_head.output(0)->dims->data[0] *
      interpreter_head.output(0)->dims->data[1] *
      interpreter_head.output(0)->dims->data[2],
    interpreter_head.output(0)->data.f,
    "/home/kyamkovyi/projects/tflite-micro/outputs"
  );

  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
