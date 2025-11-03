import os
import glob

# 1) Disable XNNPACK BEFORE importing TF (debug only)
os.environ["TF_TFLITE_USE_XNNPACK"] = "0"

import cv2
import numpy as np
import tensorflow as tf
from people_detection_pg2.softmax_exp.tflite_inference_debug import TFLiteInference

print("TensorFlow version: ", tf.__version__)


# IS_UINT8 = False
# MODEL_PTH = "/home/klymy/projects/researches/people_detection_pg2/results/cam3_revB_batch1_5_margin_15_rq_bg_wb_alif_clahe_box_01_dfl_01_2nd/weights/best_non_separated_saved_model_full_npu/best_float32.tflite"
IS_UINT8 = True
MODEL_PTH = "/home/klymy/projects/researches/people_detection_pg2/results/cam3_revB_batch1_5_margin_15_rq_bg_wb_alif_clahe_box_01_dfl_01_2nd/weights/best_non_separated_saved_model_full_npu/best_full_integer_quant.tflite"

BASE_PTH = "/home/klymy/data/people_detection/filtered_v3_margin_15/images/val/"
IMGSZ = 320


# Initialize inference class with layer indices to collect
# You can specify additional layer indices to collect during inference
layer_indices = []  # Add layer indices here if needed, e.g., [100, 200, 300]
inference = TFLiteInference(MODEL_PTH, is_uint8=IS_UINT8, imgsz=IMGSZ, layer_indices=layer_indices)

# Load and run inference on first image
img_pths = glob.glob(os.path.join(BASE_PTH, "*.jpg"))
print(f"Found {len(img_pths)} images")

img = cv2.imread(img_pths[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference
results = inference.infer_image(img)

# Get logits and probs
logits_np, probs_np = inference.get_logits_probs()

print("Logits shape:", logits_np.shape)
print("Probs shape:", probs_np.shape)
print("Logits:", logits_np[0, 0])
print("Probs:", probs_np[0, 0])
print("Sum probs:", np.sum(probs_np[0, 0]))
