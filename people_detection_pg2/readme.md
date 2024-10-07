# Introduction

The original YOLOv8 implementation has an issue with int8 quantization. So, the issues resolved in (the fork)[https://github.com/DeGirum/ultralytics_yolov8]. The developers of the fork just remove the detection head from the network.

# How to train YOLOv8 for microcontroller

1. Train YOLOv8 using regural `ultralytics`:
```bash
python train_yolo.py
```

2. Install `DeGirum/ultralytics_yolov8`:
```bash
bash install_dg_yolo.sh 
```

3. Quantize model:
```bash
python export.py ...
```

4. Check the model using `test_tflite_model.ipynb`.