# pg1_yolo_evaluator

Batch evaluation of YOLO model variants against PG1 footage, at full and reduced image quality.

**Status:** ARCHIVED — frozen 2025-04 · 4 tracked files, 5.4 GB of results on disk (gitignored)

## Purpose

Compare detector candidates on the same recorded footage under identical settings, so a model
choice rests on measured accuracy rather than on which run someone remembers. Six result sets
are present, covering YOLOv8n, YOLOv8-tiny, YOLO11 and YOLO11n, each at native and reduced
quality — reduced quality being the standing question of whether the sensor can ship cheaper
optics or heavier compression.

## Layout

| Script | Role |
|---|---|
| `eval_pipeline.py` | Entry point — takes `--weights`, `--conf`, `--iou`, `--device`, `--output_root_dir` and drives a run |
| `eval.py` | The evaluation itself |
| `compute_avg_score.py` | Aggregates scores across a result set |
| `eval.ipynb` | Interactive comparison of the result sets |

`results_*/` hold the six output sets. They are **gitignored** — 5.4 GB that does not belong in
version control.

> **This project is the pattern the rest of the repo should follow:** four scripts tracked,
> gigabytes of output ignored. Nothing else here gets that ratio right.

## Connections

- **Defaults to weights from `people_detection_pg2`** —
  `--weights ../people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt`.
- Reads footage from the fileserver (`/mnt/pg-fs01`); those paths may no longer resolve.
- Imports `PG_PyTraining` for PG1-era helpers.
- `people_detection_pg2/extended_evaluation_metrics/` is the newer, richer evaluation harness —
  prefer it for PG2 work.

## Dependencies

`ultralytics`, `torch`, `opencv-python`, `pandas`, `tqdm`, plus `PG_PyTraining` on `sys.path`
and its `ml_training` / `simber` helpers. No `requirements.txt`.
