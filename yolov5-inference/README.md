YOLOv5 inference on C++ using tf lite and opencv+onnx. The experiment uses the yolov5n model pre-trained on COCO.
Code: https://dev.azure.com/pointgrab/PG/_git/researches?path=/yolov5-inference

## How to download and build TF lite: https://www.tensorflow.org/lite/guide/build_cmake#step_5_build_tensorflow_lite
TF lite CMakeLists.txt example: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/minimal/CMakeLists.txt

## Results

The inferences were tested on pg1b and pg1c devices(raspberry pi 3 and 4) with stopped cp-core.service. Each experiment uses the same setup:
- as input Lena image
- input size 640*640
- number of runs = 100 and inference time averaged

### tab. 1. Opencv dnn + onnx model. pg1b
part | time(mean/std)(ms) 
-- | -- 
preprocessing | 25/1 
inference | 1187/14 
postprocessing | 3/0

### tab. 2. Opencv dnn + onnx model, pg1c
part | time(mean/std)(ms) 
-- | -- 
preprocessing | 17/1 
inference | 547/31 
postprocessing | 0/0 


### tab. 3. tf lite fp16. pg1b
time(mean/std)(ms) 
part | 1 thread | 2 threads | 4 threads 
-- | -- | -- | -- |
preprocessing | 29/9 | 29/9 | 29/9
inference | 1155/24 | 688/20 | 490/20
postprocessing | 95/4 | 94/4 | 94/4

### tab. 4. tf lite fp16. pg1c
part | 1 thread | 2 threads | 4 threads 
-- | -- | -- | -- |
preprocessing | 12/2 | 15/4 | 15/4
inference | 692/12 | 431/10 | 335/10
postprocessing | 31/2 | 32/4 | 33/2


### tab. 5. tf lite int8. pg1b
part | 1 thread | 2 threads | 4 threads 
-- | -- | -- | -- |
preprocessing | 10/1 | 10/1 | 10/1
inference | 1246/4 | 802/4 | 581/9
postprocessing | 111/7 | 111/8 | 110/7

### tab. 6. tf lite int8. pg1c
part | 1 thread | 2 threads | 4 threads 
-- | -- | -- | -- |
preprocessing | 5/1 | 5/1 | 5/1
inference | 540/2 | 350/2 | 266/4
postprocessing | 36/4 | 35/4 | 35/4


The tables above show that the best inference type is tf lite. with multiple threads. For pg1c int8 is better. 

## How to reproduce

Rename one of the CMakeLists file templates and build the project in release mode.