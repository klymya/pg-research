import os
import glob

# 1) Disable XNNPACK BEFORE importing TF (debug only)
os.environ["TF_TFLITE_USE_XNNPACK"] = "0"

import cv2
import numpy as np
import tensorflow as tf
print("TensorFlow version: ", tf.__version__)


MODEL_PTH = "/home/klymy/projects/researches/people_detection_pg2/results/cam3_revB_batch1_5_margin_15_rq_bg_wb_alif_clahe_box_01_dfl_01_2nd/weights/best_non_separated_saved_model_full_npu/best_float32.tflite"
BASE_PTH = "/home/klymy/data/people_detection/filtered_v3_margin_15/images/val/"
IMGSZ = 320
IMG_SHAPE = (IMGSZ, IMGSZ)
INPUT_SHAPE = (1, IMGSZ, IMGSZ, 3)

SOFTMAX_INPUT_IDX = 483
SOFTMAX_OUTPUT_IDX = 484


def pre_process(img: np.ndarray, input_shape: tuple = INPUT_SHAPE, img_shape: tuple = IMG_SHAPE) -> np.ndarray:
    if img.shape == input_shape:
        return img
    im = np.zeros(input_shape, dtype=np.int8)

    original_h, origial_w, _ = img.shape
    ratio = img_shape[0] / origial_w
    resized_h = int(original_h * ratio)

    resized_img = cv2.resize(img, (img_shape[0], resized_h))
    im[0, :resized_h, :img_shape[0], :] = resized_img[:, :, :]

    return im


def _qparams(d):
    qp = d.get("quantization_parameters", {})
    return qp.get("scales", None), qp.get("zero_points", None)


def _get_tensor(interp, d) -> np.ndarray:
    return interp.get_tensor(d["index"])


def read_dfl_softmax_tensors(interp, pre_d, post_d):
    """
    Convenience: locate and return arrays and details.
    Returns: (logits_np, probs_np, pre_detail, post_detail, bin_axis)
    """
    logits_np = _get_tensor(interp, pre_d)
    probs_np  = _get_tensor(interp, post_d)
    return logits_np, probs_np


img_pths = glob.glob(os.path.join(BASE_PTH, "*.jpg"))
print(f"Found {len(img_pths)} images")

img = cv2.imread(img_pths[0])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_data = pre_process(img, INPUT_SHAPE, IMG_SHAPE).astype(np.float32)


yolo_interpreter = tf.lite.Interpreter(
    model_path=MODEL_PTH,
    experimental_preserve_all_tensors=True,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    experimental_delegates=[],  # ensure no delegate silently fuses away intermediates
)
yolo_interpreter.allocate_tensors()


yolo_input_details = yolo_interpreter.get_input_details()
yolo_output_details = yolo_interpreter.get_output_details()

all_details = yolo_interpreter.get_tensor_details()
all_indices = {d["index"] for d in all_details}
if SOFTMAX_INPUT_IDX not in all_indices or SOFTMAX_OUTPUT_IDX not in all_indices:
    raise ValueError(
        f"Given indices not found in this model. "
        f"Available index range: [{min(all_indices)}..{max(all_indices)}]. "
        f"Got IN={SOFTMAX_INPUT_IDX}, OUT={SOFTMAX_OUTPUT_IDX}."
    )

# Grab detail dicts by index
logits_details = next(d for d in all_details if d["index"] == SOFTMAX_INPUT_IDX)
probs_details  = next(d for d in all_details if d["index"] == SOFTMAX_OUTPUT_IDX)

details = yolo_input_details[0]
yolo_interpreter.set_tensor(details['index'], input_data)
yolo_interpreter.invoke()

logits_np, probs_np = read_dfl_softmax_tensors(yolo_interpreter, logits_details, probs_details)
print("Logits:", logits_np[0, 0])
print("Probs:", probs_np[0, 0])
print("Sum probs:", np.sum(probs_np[0, 0]))
