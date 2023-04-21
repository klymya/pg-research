Yolov5n inference on C++ using tf lite and opencv+onnx.

How to downlaod and build TF lite: https://www.tensorflow.org/lite/guide/build_cmake#step_5_build_tensorflow_lite
TF lite CMakeLists.txt example: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/CMakeLists.txt

## Results

The inferences were tested on pg1b device(raspberry pi 3) device with stoped cp-core.service. Each experiment uses the same setup:
- as input Lena image
- input size 640*640
- number of runs = 100 and inference time averaged

tab. 1. Opencv dnn + onnx model
part | time(mean/std)(ms) 
-- | -- 
preprocessing | 25/1 
inference | 1187/14 
postprocessing | 3/0 


tab. 2. tf lite + fp16
time(mean/std)(ms) 
part | 1 thread | 2 threads | 4 threads 
-- | -- | -- | -- |
preprocessing | 29/9 | 29/9 | 29/9
inference | 1155/24 | 688/20 | 490/20
postprocessing | 95/4 | 94/4 | 94/4

tab. 3. tf lite + fp16
tf int8
part | 1 thread | 2 threads | 4 threads 
-- | -- | -- | -- |
preprocessing | 10/1 | 10/1 | 10/1
inference | 1246/4 | 802/4 | 581/9
postprocessing | 111/7 | 111/8 | 110/7

The tables above show that the best inference type is tf lite + fp16 with multiple threads.

## How to reproduce

Rename one of the CMakeLists file templates and build the project in release mode.