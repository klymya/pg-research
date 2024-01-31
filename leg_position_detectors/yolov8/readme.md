## Introduction

The original YOLOv8 implementation has an issue with int8 quantization. So, the issues resolved in (the fork)[https://github.com/DeGirum/ultralytics_yolov8]. The developers of the fork just remove the detection head from the network.

The code here depend on the fork, so, you need to add path to the folder to `PYTHONPATH` or move the notebooks to the folder with customized YOLO.

Also, we have multiple model sizes:
* 1.1 millions parameters (`yolov8-pose.yaml`);
* 2.2 millions parameters (`yolov8-pose-tiny.yaml`, `yolov8-pose-tiny-channels.yaml`),

and multiple input sizes for each model:
* 256;
* 320;
* 640;

## Detection head

As mentioned above, the detection head is separated from the network. Here we have two implementation of it:
* NumPy Python code (`tf_lite_inference.ipynb`); 
* torch module, that should be converted to tflite fp32 or fp16 weights (`tf_lite_inference_detection_head.ipynb`).