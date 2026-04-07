# Regenerate YOLO Datasets v2

This document describes the cumulative dataset regeneration pipeline implemented in:

- [regenerate_yolo_datasets_v2.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/regenerate_yolo_datasets_v2.py)

The script rebuilds the `cam4_revB` cumulative datasets from the current raw batches, corrected CVAT XML files, and the current production utilities for:

- anti-leakage split generation
- CVAT-to-YOLO conversion
- margin filtering
- white balance
- white balance + CLAHE

## Outputs

Subsets are written to:

- `/home/klymy/data/cam4_revB/subsets_v2`

Datasets are written to:

- `/home/klymy/data/cam4_revB/yolo_dataset_v2`

Final retained cumulative datasets are:

- `batch1_3_margin_10_wb_alif`
- `batch1_3_margin_10_wb_alif_clahe`
- `batch1_3_bg_margin_10_wb_alif`
- `batch1_3_bg_margin_10_wb_alif_clahe`
- `batch1_5_bg_margin_10_wb_alif`
- `batch1_5_bg_margin_10_wb_alif_clahe`
- `batch1_7_bg_margin_10_wb_alif`
- `batch1_7_bg_margin_10_wb_alif_clahe`
- `batch1_10_bg_margin_10_wb_alif`
- `batch1_10_bg_margin_10_wb_alif_clahe`
- `batch1_11_bg_margin_10_wb_alif`
- `batch1_11_bg_margin_10_wb_alif_clahe`
- `batch1_12_bg_margin_10_wb_alif`
- `batch1_12_bg_margin_10_wb_alif_clahe`

The raw cumulative datasets and the `_margin_10` intermediates are temporary and are deleted after the downstream processed datasets have been created successfully.

Chunk-only processed datasets such as `batch4_5_margin_10_wb_alif` are internal merge inputs and are also deleted after their cumulative merge step succeeds.

## Pipeline

The pipeline is incremental, not full-cumulative recomputation.

The pipeline is:

1. Plan and validate stage subsets
2. Build and process the initial `batch1_3` dataset
3. Build and process only the new chunk being added
4. Merge the processed chunk into the already processed cumulative datasets
5. Validate file consistency at each stage
6. Remove chunk and raw / margin intermediates when they are no longer needed

Build order:

1. `batch1_3`
2. `batch1_3_bg`
3. `batch1_5_bg`
4. `batch1_7_bg`
5. `batch1_10_bg`
6. `batch1_11_bg`
7. `batch1_12_bg`

Incremental flow by example:

- `batch1_3` is built and processed once
- background is built as its own tiny chunk, processed once, then merged into the processed `batch1_3` datasets
- `batch4_5` is built and processed once, then merged into `batch1_3_bg_*`
- `batch6_7` is built and processed once, then merged into `batch1_5_bg_*`
- `batch8_10` is built and processed once, then merged into `batch1_7_bg_*`
- `batch11` is built and processed once, then merged into `batch1_10_bg_*`
- `batch12` is built and processed once, then merged into `batch1_11_bg_*`

This avoids rerunning margin filtering, WB, and CLAHE on images that were already processed in previous cumulative steps.

## Holdout Invariant

The holdout source of truth is:

- `/home/klymy/data/cam4_revB/train_val_holdout_subsets_batches_1_3.json`

Rules:

- the original `batch1_3` holdout membership is preserved exactly
- any sample from that original holdout remains in `holdout` when present in later cumulative datasets
- later stage splitting reuses the production anti-leakage logic instead of a separate policy

Important caveat:

- the original holdout JSON contains two malformed filenames
- they are preserved in `batch1_3` holdout exactly
- they cannot contribute to later group-based protected-holdout matching because no valid scene-group key can be parsed from them
- this is recorded as a warning in `subsets_v2/regeneration_manifest.json`

## Train / Val Rule

The script reuses the current anti-leakage split implementation in:

- [antileakage_split.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/antileakage_split.py)

Behavior:

- later batches use the shared group-based anti-leakage splitter
- `batch1_3` keeps the original holdout exact and re-splits only the working pool by group
- deterministic shuffle uses `seed=42` by default
- target validation ratio is `0.10` by default

## Background Handling

Background images come from:

- `/home/klymy/data/cam4_revB/4.2_background`

Behavior:

- background starts at `batch1_3_bg`
- background is appended only to `train`
- background images get empty YOLO label files
- background persists in all later cumulative `_bg` datasets
- background is processed as its own chunk before being merged into the processed `batch1_3` datasets

## Margin / WB / CLAHE

The script reuses:

- [filter_small_bboxes.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes.py)
- [preprocess_dataset.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py)

Margin note:

- the project implementation uses normalized coordinates
- `margin=0.10` is therefore the current production-equivalent interpretation of `margin_10`

Processing note:

- the `wb_alif_clahe` mode applies white balance first and CLAHE second within the shared preprocessing implementation

## Validation

Before writing outputs, `--dry-run` validates:

- required input paths
- cumulative batch membership
- holdout preservation
- stage split integrity
- expected processed dataset naming

During materialization, the script validates:

- processed chunk file consistency before merge
- raw dataset file consistency
- margin-filtered dataset file consistency
- WB dataset file consistency
- WB+CLAHE dataset file consistency
- image / label stem alignment across `train`, `val`, `holdout`, and `holdout_v2` when present
- merged dataset contents against the expected cumulative union of prior data plus the new chunk

## Run

Dry-run only:

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/regenerate_yolo_datasets_v2.py \
  --dry-run \
  --overwrite
```

Full run:

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/regenerate_yolo_datasets_v2.py \
  --overwrite
```

If the corrected XML files contain invalid annotations and you want the converter to keep running while still auditing them:

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/regenerate_yolo_datasets_v2.py \
  --overwrite \
  --allow-annotation-issues
```

## Dependency Note

Dry-run works without OpenCV.

WB / CLAHE execution requires `cv2`, because:

- [preprocess_dataset.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py)

imports OpenCV during execution of the preprocessing stages.

## Tests

Relevant tests:

- [test_antileakage_split.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_antileakage_split.py)
- [test_regenerate_yolo_datasets_v2.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_regenerate_yolo_datasets_v2.py)

Run:

```bash
pytest \
  /home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_antileakage_split.py \
  /home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_regenerate_yolo_datasets_v2.py
```
