# Introduction

The original YOLOv8 implementation has an issue with int8 quantization. So, the issues resolved in (the fork)[https://github.com/DeGirum/ultralytics_yolov8]. The developers of the fork just remove the detection head from the network.

# How to train YOLOv8 for microcontroller

1. Train YOLOv8 using regural `ultralytics`:
```bash
python train_yolo.py
```
2. Quantize the model:
    1. Install `DeGirum/ultralytics_yolov8`:
    ```bash
    bash install_dg_yolo.sh 
    ```

    2. Export model to int8:
    ```bash
    python export.py ...
    ```

    3. Check the model using `test_tflite_model.ipynb`.

2. Optimaze model for Alif:
    1. Install vela compiler:
    ```bash
    pip install -r requrements.txt
    ```

    2. Optimize the model:
    ```bash
    vela best_full_integer_quant.tflite \
        --output-dir vela_he \
        --accelerator-config ethos-u55-128 \
        --optimise Performance \
        --config vela.ini \
        --system-config Ethos_U55_High_End_Embedded \
        --memory-mode Shared_Sram
    ```

    3. Optimze the head:
    ```bash
    vela head_float32.tflite \
        --output-dir vela_he \
        --accelerator-config ethos-u55-128 \
        --optimise Performance \
        --config vela.ini \
        --system-config Ethos_U55_High_End_Embedded \
        --memory-mode Shared_Sram
    ```