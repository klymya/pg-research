# yolo_latest_test

Smoke test that a newer ultralytics release trains, exports and runs.

**Status:** ARCHIVED — frozen 2025-09 · 3 tracked files, 156 KB

## Purpose

Before adopting a newer ultralytics/YOLO11 version in `people_detection_pg2`, check the basics
on a throwaway dataset: does training start, does export produce a usable saved model, does
ONNX inference run. Nothing here is a result — it is a version-compatibility probe.

| File | What |
|---|---|
| `tst.py` | One-epoch YOLO11n train on `coco128` |
| `tst_quant.py` | Quantised export path |
| `inference_onnx.py` | ONNX inference check |

`pg2/`, `pg2_uint8/`, `yolo11n_saved_model/`, `yolov8n_saved_model/`, `results/` hold the
generated artefacts.

## Connections

- Probes the ultralytics version used by `people_detection_pg2`. Nothing depends on it.

## Dependencies

`ultralytics`, plus `onnxruntime`, `opencv-python`, `numpy` for the inference check.

```bash
pip install ultralytics onnxruntime opencv-python numpy
```
