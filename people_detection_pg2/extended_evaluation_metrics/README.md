# Extended Evaluation Metrics

Utilities to align the **YOLO holdout labels** with **CVAT XML annotations** and export per-frame / per-detection attribute tables for slice-based analysis.

## Setup

- Copy `.env.example` to `.env` and update paths:
  - `ENV_PATH` is used by all scripts/notebook to locate the env file.
  - Common paths used across scripts: `REPORT_BASE`, `FRAMES_ATTRS`, `DETECTIONS_ATTRS`.
  - Evaluation outputs: `METRICS_PER_IMG`, `METRICS_PER_GT`, `METRICS_PER_PRED`, `GLOBAL_METRICS_PTH`.
  - Sweep outputs: `SWEEP_RESULTS_PTH`, `BEST_PARAMS_PTH`.
  - Visualization inputs/outputs: `YOLO_HOLDOUT`, `VIZ_DIR`, `VIZ_OUT_DIR`.
  - Report generation controls: `RUN_CONCLUSIONS_AGENT`, `ANTHROPIC_API_KEY`.

## Data Sources (Cam4 RevB)

- YOLO holdout labels (GT in YOLO format):  
  `/home/klymy/data/cam4_revB/yolo_dataset/batch1_3_margin_15_awb_clahe/labels/holdout/*.txt`
- CVAT annotations (GT + attributes):  
  `/home/klymy/data/cam4_revB/batch1_corrected/*.xml`  
  `/home/klymy/data/cam4_revB/batch2_corrected/*.xml`  
  `/home/klymy/data/cam4_revB/batch3_corrected/*.xml`
- Design doc (attribute list & report idea):  
  `/home/klymy/projects/PG2.wiki/PG2/Algo/Roadmaps-&-Proposals/YOLOv8-Evaluation-Report-—-Extended-Design-(Next-Iteration).md`

## Scripts

### `join_holdout_set_data.py`

Checks the hypothesis that detection ordering is consistent between CVAT XML and YOLO `.txt` for each frame and produces a **mapping** when it is not.

- Writes JSONL (mismatches only by default) with:
  - `yolo_id`, `yolo_path`, `xml_path`, `xml_image_name`
  - `mapping`: `{"yolo":[...], "xml":[...]}` where `mapping["yolo"][k]` is matched to `mapping["xml"][k]`

Example:
```bash
python people_detection_pg2/extended_evaluation_metrics/join_holdout_set_data.py \
  --yolo-glob "/home/klymy/data/cam4_revB/yolo_dataset/batch1_3_margin_15_awb_clahe/labels/holdout/*.txt" \
  --out-jsonl people_detection_pg2/extended_evaluation_metrics/holdout_data/mismatch_report.json \
  --report-format minimal
```

### `export_holdout_attributes_csv.py`

Exports two CSVs for the holdout set:

- **Frame-level** (`frames_attributes.csv`):
  - `camera_height`
  - `num_gt_detections` (from CVAT XML)
  - derived `scene_density` (sparse/medium/dense)
- **Detection-level** (`detections_attributes.csv`):
  - YOLO fields (`cls`, `x_c`, `y_c`, `w`, `h`, `x`, `y`)
  - `leg_distance_to_center` (Euclidean distance from `(x,y)` to `(0.5,0.5)`, fallback to `(x_c,y_c)` if missing)
  - `position` (center/edge based on leg distance)
  - CVAT attributes (filled via mapping where applicable):
    - `posture`
    - `occluded_by_object`
    - `occluded_by_person`

Example:
```bash
python people_detection_pg2/extended_evaluation_metrics/export_holdout_attributes_csv.py \
  --yolo-glob "/home/klymy/data/cam4_revB/yolo_dataset/batch1_3_margin_15_awb_clahe/labels/holdout/*.txt" \
  --xml "/home/klymy/data/cam4_revB/batch1_corrected/*.xml" "/home/klymy/data/cam4_revB/batch2_corrected/*.xml" "/home/klymy/data/cam4_revB/batch3_corrected/*.xml" \
  --mismatch-report people_detection_pg2/extended_evaluation_metrics/holdout_data/mismatch_report.json \
  --out-frames-csv people_detection_pg2/extended_evaluation_metrics/holdout_data/frames_attributes.csv \
  --out-dets-csv people_detection_pg2/extended_evaluation_metrics/holdout_data/detections_attributes.csv
```

### `evaluation.py`

Runs validation and exports evaluation artifacts. Supports PT and TFLite backends:
- `ultralytics_pt` (default): standard `YOLO(...).val(...)`
- `tflite_mixed`: INT8 backbone + FP32 head
- `tflite_backbone_only`: fully INT8 backbone (no separate head), optional output correction model

The evaluator now supports two modes:
- Single-run mode: one `(CONF_TH, IOU_TH)` pair
- Sweep mode: full grid search over `SWEEP_CONF_THS x SWEEP_IOU_THS`

Important behavior:
- Every invocation runs validation from scratch. Existing `predictions.json` is not reused as cache.
- In sweep mode, each parameter pair runs a fresh validation pass and the best pair is selected by `SELECTION_METRIC`.
- Supported selection metrics: `f1`, `recall`, `precision` (default: `f1`)
- Sweep selection uses validator **pose** metrics (`metrics/precision(P)`, `metrics/recall(P)`, derived pose F1) when available, so the chosen value matches the Section A global table in the report.

Outputs:
- `metrics_per_img.csv` (per-frame metrics)
- `metrics_per_gt.csv` (GT-centric detections, with TP/FN and match ids)
- `metrics_per_pred.csv` (Pred-centric detections, with TP/FP and match ids)
- `metrics_global.json` (global metrics dict from validator)
- `predictions.json` (Ultralytics-compatible schema used by downstream CSV logic)
- `sweep_results.csv` (all evaluated parameter combinations with summary metrics)
- `best_params.json` (selected thresholds and summary metrics for the winning run)

Example:
```bash
python people_detection_pg2/extended_evaluation_metrics/evaluation.py --env-file people_detection_pg2/extended_evaluation_metrics/.env
```

Required `.env` keys for backend selection:
- Common:
  - `EVAL_BACKEND=ultralytics_pt|tflite_mixed|tflite_backbone_only`
  - `DATASET_PTH`, `PROJECT`, `PROJECT_NAME`, `PREDICTIONS_PTH`, `GLOBAL_METRICS_PTH`
  - `CONF_TH`, `IOU_TH`, `INPUT_WIDTH`, `BATCH`, `DEVICE`
  - `ENABLE_SWEEP=0|1`
  - `SELECTION_METRIC=f1|recall|precision`
  - `SWEEP_RESULTS_PTH`, `BEST_PARAMS_PTH`
- Single-run thresholds:
  - `CONF_TH`, `IOU_TH`
- Sweep thresholds:
  - `SWEEP_CONF_THS` as a comma-separated list
  - `SWEEP_IOU_THS` as a comma-separated list
- Optional threshold source:
  - `USE_INFERENCE_PARAMS=0|1`
  - When `USE_INFERENCE_PARAMS=1` and sweep is disabled, the evaluator reads the default thresholds from `${BASE_EXPERIMENT_DIR}/inference_params.json`.
- PT backend:
  - `WEIGHTS_PTH`
- TFLite mixed:
  - `WEIGHTS_PTH` (used only for Ultralytics validator metadata/class names)
  - `TFLITE_BACKBONE_PTH`
  - `TFLITE_HEAD_PTH`
- TFLite backbone-only:
  - `WEIGHTS_PTH`
  - `TFLITE_BACKBONE_PTH`
  - `TFLITE_OUTPUT_CORRECTION_MODEL_PTH` (optional)

### `visualize_detections.py`

Renders per-image overlays (GT/Pred, TP/FP/FN colors, leg positions, score labels) into `VIZ_OUT_DIR`.

Example:
```bash
python people_detection_pg2/extended_evaluation_metrics/visualize_detections.py
```

## Report Notebook

`extended_report.ipynb` renders the interactive report using:
- `metrics_global.json`
- `metrics_per_img.csv`, `metrics_per_gt.csv`, `metrics_per_pred.csv`
- `best_params.json` when available
- `holdout_data/frames_attributes.csv`, `holdout_data/detections_attributes.csv`
- rendered images from `val/viz`

All paths are resolved via `.env` through `ENV_PATH`.

When launched through `run_eval_and_viz.sh`, the notebook is always executed once and exported to HTML.
If `RUN_CONCLUSIONS_AGENT=1` (default), the script then runs `generate_conclusions_claude.py`,
injects the generated markdown, and re-executes the notebook to produce the final HTML.
Set `RUN_CONCLUSIONS_AGENT=0` to skip the Claude step.

## Outputs

Generated artifacts live in `people_detection_pg2/extended_evaluation_metrics/holdout_data/`:
- `people_detection_pg2/extended_evaluation_metrics/holdout_data/mismatch_report.json` (JSONL)
- `people_detection_pg2/extended_evaluation_metrics/holdout_data/frames_attributes.csv`
- `people_detection_pg2/extended_evaluation_metrics/holdout_data/detections_attributes.csv`

Evaluation artifacts typically live in `people_detection_pg2/extended_evaluation_metrics/`:
- `metrics_per_img.csv`
- `metrics_per_gt.csv`
- `metrics_per_pred.csv`
- `metrics_global.json`
- `sweep_results.csv`
- `best_params.json`
