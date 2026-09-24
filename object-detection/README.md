# object-detection

Second iteration of furniture detection — tables, chairs, laptops — using YOLOv7.

**Status:** ARCHIVED — frozen 2024-05. **Superseded by `object-detection-v3`.**

## Purpose

Detect office furniture so the system can reason about the space, not only the people in it: a
chair that is occupied, a desk that is in use. This round trained and evaluated YOLOv7 variants
and checked the dataset for leakage between train and previous sets.

It is volume 2 of the line. Volume 3 (`object-detection-v3/`) rebuilt the dataset from scratch,
moved to YOLOv8 and produced the model that was actually used — **start there instead.**

## Layout

| Notebook | What it does |
|---|---|
| `eval_yolov7.ipynb` | Evaluate the trained YOLOv7 variants |
| `dataset_stat.ipynb` | Dataset composition statistics |
| `check_img_prev_datasets_intersection.ipynb` | Leakage check against earlier datasets |
| `verify_exist_annotation.ipynb`, `verify_exist_annotation-1.ipynb` | Annotation sanity checks |
| `verivy-fix-data-yolov7.ipynb` | Data repair (filename typo preserved) |
| `Untitled.ipynb` | Unnamed scratch notebook, never cleaned up |

`weights/` holds `yolov7-custom7`, `yolov7-tiny-baseline`, `yolov7-tiny-pg-baseline` and
`obj3_v0.1.onnx`.

## Connections

- **Superseded by `object-detection-v3`**, which carries the dataset database and the final
  model.
- Shares the CVAT annotation schema handled by `annotation_poc/`.
- `yolo-inference/` benchmarks YOLOv7 inference in C++ on PG1 hardware — the deployment side of
  this work.

## Dependencies

Notebook-only; no scripts and no `requirements.txt`. Imports `onnx` and `onnx_opcounter` for
model inspection; the training itself used an external YOLOv7 checkout (gitignored, see the
root `.gitignore`).
