# yolo-inference

C++ YOLO inference benchmarks on PG1 hardware: TFLite versus OpenCV-DNN+ONNX, fp16 versus int8,
across thread counts.

**Status:** ARCHIVED — frozen 2024-05 · 216 MB

## Purpose

Decide how to run a detector on a Raspberry Pi-class device, and whether it is fast enough at
all. The measurement is the deliverable — the tables below are the reason this project exists,
and they are what informed the move to a dedicated NPU (the Alif Ethos-U55) for PG2.

Tested on pg1b (Pi 3) and pg1c (Pi 4) with `cp-core.service` stopped, Lena at 640×640, 100 runs
averaged. Three runtimes: OpenCV-DNN with ONNX, TFLite fp16, TFLite int8.

Also includes YOLOv7 reparameterisation and ONNX export (`reparameterization_yolov7_tiny.py`)
and a `yolov8/` C++ ONNX inference variant.

## Connections

- Benchmarks the models produced by `object-detection/` (YOLOv7) and the YOLOv5n baseline.
- The PG2 successor is `people_detection_pg2`, where inference moved to the Ethos-U55 NPU and
  `alif_perf/` reads the cycle counts.
- `theodore/pipeline_cpp/` is a related C++ inference pipeline.

## Dependencies

C++ with vcpkg (`vcpkg.json`), CMake, OpenCV, ONNX Runtime and TensorFlow Lite — the README
below has the TFLite build instructions. Python side: `torch` and `pyyaml` for the
reparameterisation script.

---

## Original notes and results

YOLO inference on C++ using tf lite and opencv+onnx.

# YOLOv5 
The experiment uses the yolov5n model pre-trained on COCO.

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

# YOLOv7
The experiment uses the yolov5 model trained for object detection task(3 classes).
Here we only measure onnx+cv performance.

[How to convert yolov7 to onnx](https://github.com/WongKinYiu/yolov7/issues/49#issuecomment-1185120410). We need only set number of classes = 3 in config yaml file and code.

## Results
The inferences were tested on pg1b and pg1c devices(raspberry pi 3 and 4) with stopped cp-core.service. Each experiment uses the same setup:
- as input cam2 image
- input size 640*640
- number of runs = 100 and inference time averaged

### tab. 7. pg1b VS pg1c
part | pg1b | pg1c
-- | -- | -- |
inference | 16863/8620 | 10266/1118


# How to reproduce

Rename one of the CMakeLists file templates and build the project in release mode.