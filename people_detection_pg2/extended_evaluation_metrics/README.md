# Extended Evaluation Metrics

Utilities to align the **YOLO holdout labels** with **CVAT XML annotations** and export per-frame / per-detection attribute tables for slice-based analysis.

## Setup

- Copy `.env.example` to `.env` and update paths:
  - `ENV_PATH` is used by all scripts/notebook to locate the env file.
  - Common paths used across scripts: `REPORT_BASE`, `FRAMES_ATTRS`, `DETECTIONS_ATTRS`.
  - Evaluation outputs: `METRICS_PER_IMG`, `METRICS_PER_GT`, `METRICS_PER_PRED`, `GLOBAL_METRICS_PTH`.
  - Visualization inputs/outputs: `YOLO_HOLDOUT`, `VIZ_DIR`, `VIZ_OUT_DIR`.

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

Runs YOLO validation (if needed), saves:
- `metrics_per_img.csv` (per-frame metrics)
- `metrics_per_gt.csv` (GT-centric detections, with TP/FN and match ids)
- `metrics_per_pred.csv` (Pred-centric detections, with TP/FP and match ids)
- `metrics_global.json` (YOLO global metrics from `val_result.results_dict`)

Example:
```bash
python people_detection_pg2/extended_evaluation_metrics/evaluation.py --env-file people_detection_pg2/extended_evaluation_metrics/.env
```

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
- `holdout_data/frames_attributes.csv`, `holdout_data/detections_attributes.csv`
- rendered images from `val/viz`

All paths are resolved via `.env` through `ENV_PATH`.

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
