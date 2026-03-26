# H270 Frame Selection

## Description

[`select_h270_frames.py`](/home/klymy/projects/researches/people_detection_pg2/extended_evaluation_metrics/select_h270_frames.py) selects useful frames from the annotated H270 batch for future YOLO training and validation expansion.

The script:
- loads existing H270 evaluation artifacts and metadata
- validates that the inputs are internally consistent enough for scoring
- builds a canonical frame-level table
- ranks frames by usefulness using evaluation-derived signals
- selects frames under group-aware constraints
- assigns selected groups to `train` and `val` with strict anti-leakage splitting
- writes manifests, reports, and optionally materializes a new YOLO dataset

Default behavior is safe:
- it does not modify the original H270 holdout dataset
- it only writes reports and manifests unless `--materialize-dataset` is enabled

## Command To Run The Script

```bash
python /home/klymy/projects/researches/people_detection_pg2/extended_evaluation_metrics/select_h270_frames.py \
  --metrics-per-img /home/klymy/projects/researches/people_detection_pg2/extended_evaluation_metrics/val_H270_16-44-09/cam3_revB_batch1_12_margin_15_wb_alif_clahe_box_01_dfl_01_2nd__batch_H270_16-44-09_margin_10_wb_alif_clahe__tflite_backbone_only/metrics_per_img.csv \
  --metrics-per-gt /home/klymy/projects/researches/people_detection_pg2/extended_evaluation_metrics/val_H270_16-44-09/cam3_revB_batch1_12_margin_15_wb_alif_clahe_box_01_dfl_01_2nd__batch_H270_16-44-09_margin_10_wb_alif_clahe__tflite_backbone_only/metrics_per_gt.csv \
  --metrics-per-pred /home/klymy/projects/researches/people_detection_pg2/extended_evaluation_metrics/val_H270_16-44-09/cam3_revB_batch1_12_margin_15_wb_alif_clahe_box_01_dfl_01_2nd__batch_H270_16-44-09_margin_10_wb_alif_clahe__tflite_backbone_only/metrics_per_pred.csv \
  --frames-attributes /home/klymy/projects/researches/people_detection_pg2/extended_evaluation_metrics/holdout_H270_16-44-09/frames_attributes.csv \
  --detections-attributes /home/klymy/projects/researches/people_detection_pg2/extended_evaluation_metrics/holdout_H270_16-44-09/detections_attributes.csv \
  --groups-csv /home/klymy/data/cam4_revB/4.2_batch_H270_16-44-09_groups.csv \
  --subsets-json /home/klymy/data/cam4_revB/train_val_holdout_subsets_batches_H270_16-44-09.json \
  --dataset-root /home/klymy/data/cam4_revB/yolo_dataset/batch_H270_16-44-09_margin_10_wb_alif_clahe \
  --output-dir /home/klymy/projects/researches/people_detection_pg2/extended_evaluation_metrics/h270_selection_output \
  --selection-budget 90 \
  --max-per-group 15 \
  --val-group-count 2 \
  --drop-inconsistent-frames \
  --materialize-dataset \
  --materialize-output-root /home/klymy/data/cam4_revB/yolo_dataset/batch_H270_16-44-09_selected
```

## Logic

The script uses existing evaluation outputs instead of rerunning inference.

Core logic:
- use `metrics_per_img.csv` as the base frame list
- join frame grouping from `4.2_batch_H270_16-44-09_groups.csv`
- join subset membership from `train_val_holdout_subsets_batches_H270_16-44-09.json`
- join frame metadata from `frames_attributes.csv`
- aggregate GT-side error signals from `metrics_per_gt.csv`
- aggregate prediction-side error signals from `metrics_per_pred.csv`
- join detection attribute summaries from `detections_attributes.csv`

Validation logic:
- fail if required files or required columns are missing
- fail if frame groups are invalid, overlapping, or incomplete
- fail if required joins for scoring are missing
- compute derived precision and recall from `tp_count`, `fp_count`, `fn_count`
- compare derived precision/recall against `P` and `R` from `metrics_per_img.csv`
- by default, fail fast if inconsistencies exist
- if `--drop-inconsistent-frames` is enabled, drop only the inconsistent frames and continue

## Algorithm

### 1. Canonical Frame Table

Build one row per frame with:
- identifiers: `img_id`, `yolo_id`, `frame_index`, `group_id`
- subset membership
- evaluation metrics: `P`, `R`, `F1`, `AP`
- detailed counts: `tp_count`, `fp_count`, `fn_count`
- localization and confidence aggregates
- metadata and diversity attributes

### 2. Authoritative Fields

The script treats GT counts from `metrics_per_gt.csv` as authoritative for scoring:
- `gt_count_eval`
- `gt_count_authoritative`

Metadata GT counts from `frames_attributes.csv` are kept separately:
- `gt_count_metadata`
- `gt_count_mismatch`

### 3. Usefulness Score

The frame score is a weighted combination of:
- `f1_penalty = 1 - F1`
- `fn_signal = fn_count / gt_count_authoritative`
- `fp_signal = fp_count / pred_count`
- `loc_signal = 1 - matched_iou_mean`
- `conf_signal = 1 - pred_score_mean`

Default weighted sum:

```text
usefulness_score =
  weight_f1  * f1_penalty +
  weight_fn  * fn_signal +
  weight_fp  * fp_signal +
  weight_loc * loc_signal +
  weight_conf * conf_signal
```

Higher score means the frame is considered more informative for dataset expansion.

### 4. Selection

Selection happens in two stages:
- rank eligible holdout frames by `usefulness_score` descending
- select frames subject to:
  - total budget
  - maximum frames per group
  - optional diversity caps

### 5. Train/Val Anti-Leakage Split

Train/val assignment is strictly group-based:
- one `group_id` may appear in only one split
- no frame-level random split is used
- selected validation groups are chosen deterministically
- remaining selected groups go to train

## Outputs

When `--output-dir` is set, the script writes:

- `ranked_frames.csv`
  - one row per eligible frame
  - canonical joined data
  - validation fields
  - score components
  - final usefulness score
  - selected/not selected status

- `selected_train_manifest.csv`
  - selected train frames only

- `selected_val_manifest.csv`
  - selected val frames only

- `selection_summary.json`
  - config
  - input files
  - counts
  - selected groups
  - train/val groups
  - validation summary
  - exclusion statistics

- `selection_report.md`
  - human-readable summary
  - scoring setup
  - leakage safeguards
  - group allocation
  - validation outcome

If `--materialize-dataset` is enabled, the script also creates a YOLO dataset root with:
- `images/train`
- `images/val`
- `labels/train`
- `labels/val`
- `project.yaml`

## Notes

- Default mode is manifests/reports only.
- The original H270 holdout dataset is not modified in place.
- `--drop-inconsistent-frames` is optional and should only be used when you intentionally want to skip known bad evaluation rows instead of failing fast.
