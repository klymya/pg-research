# leg_position_detectors

Neural estimation of a person's ground-contact point ("leg position") from an overhead fisheye
view, targeting int8 TFLite on a microcontroller.

**Status:** ARCHIVED — frozen 2024-04 · 377 MB

## Purpose

A bounding box tells you where a person is in the image; converting that to a floor position
needs the point where they touch the ground. Estimating it directly — as a keypoint — was the
neural approach; `leg_position_calculator/` is the non-NN alternative that was tried alongside.

Two families were explored:

- **`yolov8/`** — YOLOv8 pose models with a single keypoint, at two capacities (1.1 M and
  2.2 M parameters) and three input sizes (256, 320, 640), exported to int8 TFLite.
- **`unet/`** — U-Net variants under `people_counting/` and `people_detection/`.

## What it produced

The int8 export path developed here — **splitting the detection head off the network because
upstream ultralytics does not quantise it correctly** — is the technique later reused wholesale
in `people_detection_pg2`. `yolov8/detection_head/` holds the separated heads, and
`tflite_micro_example/` shows the MCU-side inference.

`run_results/` carries the training runs, including TensorBoard logs.

| Path | What |
|---|---|
| `yolov8/train.py` | Training entry point |
| `yolov8/yolov8-pose*.yaml` | Model configs — note `-tiny` and `-tiny-channels` variants |
| `yolov8/eval_yolo_trch_pos.ipynb`, `eval_yolo_tflite_int8_pos.ipynb` | PyTorch and int8 evaluation |
| `yolov8/tf_lite_inference*.ipynb` | TFLite inference checks |
| `yolov8/detection_head/` | The separated head |
| `yolov8/tflite_micro_example/` | MCU inference example |
| `unet/` | U-Net alternatives |

`yolov8/readme.md` has the original notes, including the model-size and input-size matrix.

## Connections

- **Direct ancestor of `people_detection_pg2`** — the DeGirum fork, the separated head and the
  Vela path all originate here.
- `leg_position_calculator/` is the linear-regression alternative to the same problem.
- The ground point it estimates is what `gw_prototype`'s fisheye projection converts to
  centimetres.

## Dependencies

> **Environment:** run YOLO tasks in the `yolo-export` pyenv environment on
> `pgadmin-MS-7885` — it has ultralytics installed from the local DeGirum fork. Any other
> environment gets upstream ultralytics, whose head does not quantise to int8 correctly.
> See [`people_detection_pg2/readme.md`](../people_detection_pg2/readme.md#running-on-the-training-machine).


`torch`, `torchvision`, `torchsummary`, `ultralytics`, `onnx`, `onnxruntime`, `opencv-python`,
`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `paramiko` (for copying to a device). No
`requirements.txt`. Requires the `DeGirum/ultralytics_yolov8` fork on `PYTHONPATH` — see
`yolov8/readme.md`.
