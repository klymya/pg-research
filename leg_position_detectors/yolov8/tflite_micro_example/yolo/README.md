The `yolo_test.cc` runs the PG yolov8 using image with size 256*256. 
And during the test the script check the input and outputs sizes, save input and output of model to binary files for 
future reading and processing via Python.

Build the test
```bash
bazel build tensorflow/lite/micro/examples/yolo:yolo_test
```

Run the test
```bash
bazel run tensorflow/lite/micro/examples/yolo:yolo_test
```

Some folders
- models - models' weights in tflite format
- testdata input images in bmp format

During the build, the files in the folders converts to *.cc and *.h.