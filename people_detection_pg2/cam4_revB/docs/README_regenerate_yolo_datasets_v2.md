# YOLO Dataset Regeneration v2

This document covers the reproducible regeneration pipeline implemented in:

- [regenerate_yolo_datasets_v2.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/regenerate_yolo_datasets_v2.py)

The script rebuilds the cumulative `cam4_revB` YOLO datasets from the current raw batches, corrected CVAT XML files, and the current production split / conversion / merge utilities.

## Purpose

Regenerate the following cumulative datasets under:

- `/home/klymy/data/cam4_revB/yolo_dataset_v2`

Datasets produced:

- `batch1_3`
- `batch1_3_bg`
- `batch1_5_bg`
- `batch1_7_bg`
- `batch1_10_bg`
- `batch1_11_bg`
- `batch1_12_bg`

The script also regenerates stage-local conversion datasets needed for the cumulative merges:

- `batch4_5`
- `batch6_7`
- `batch8_10`
- `batch11`
- `batch12`

Stage subset JSON files are written under:

- `/home/klymy/data/cam4_revB/subsets_v2`

## Build Sequence

The pipeline runs in this order:

1. Build `batch1_3` from batches `1, 2, 3`
2. Append background images to create `batch1_3_bg`
3. Build `batch4_5` and merge into `batch1_3_bg` to create `batch1_5_bg`
4. Build `batch6_7` and merge into `batch1_5_bg` to create `batch1_7_bg`
5. Build `batch8_10` and merge into `batch1_7_bg` to create `batch1_10_bg`
6. Build `batch11` and merge into `batch1_10_bg` to create `batch1_11_bg`
7. Build `batch12` and merge into `batch1_11_bg` to create `batch1_12_bg`

## Reused Production Components

The wrapper script reuses the current production modules instead of re-implementing them:

- [antileakage_split.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/antileakage_split.py)
- [cvat2yolo_convert.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/cvat2yolo_convert.py)
- [merge_yolo_datasets.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/merge_yolo_datasets.py)
- [pipeline_dataset_utils.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/pipeline_dataset_utils.py)

The only additional orchestration logic in `regenerate_yolo_datasets_v2.py` is:

- fixed-holdout handling for the base `batch1_3` rebuild
- deterministic background injection into `train`
- dry validation before any outputs are written
- post-write verification that each dataset contains exactly the expected sample IDs

## Holdout Rule

The original holdout source of truth is:

- `/home/klymy/data/cam4_revB/train_val_holdout_subsets_batches_1_3.json`

Mandatory invariant:

- every sample listed in that JSON's `holdout` split remains in `holdout` for every cumulative dataset where it is present

Implementation details:

- for `batch1_3`, the script preserves the original `holdout` paths exactly
- it does not expand the base holdout to full scene groups, because that would change holdout composition
- for later stages, the current anti-leakage splitter is reused against the protected holdout groups derived from the original holdout JSON

Important caveat:

- the original holdout JSON contains two malformed filenames:
  - `4.2_H_cam___11.tif`
  - `4.2_H_cam___12.tif`
- those paths are still preserved in `batch1_3` holdout exactly
- they cannot contribute to later group-based protected-holdout matching because the shared parser cannot derive a valid scene-group key from them
- this is recorded as a warning in `subsets_v2/regeneration_manifest.json`

## Train / Val Rule

The script uses the improved anti-leakage behavior already present in the project:

- train / val assignment is scene-group based, not frame based
- scene groups are derived from the raw filename stem
- group order is shuffled deterministically with `seed=42`
- validation target is `10%` of non-holdout frames by default

Special handling for `batch1_3`:

- the historical `train` / `val` assignment is not reused directly because it contains train/val scene overlap
- instead, the script keeps the original `holdout` exact and re-splits only the remaining working frames by scene group

## Background Handling

Background images come from:

- `/home/klymy/data/cam4_revB/4.2_background`

Behavior:

- background images are injected starting with `batch1_3_bg`
- all background images are copied only into `images/train`
- matching empty YOLO label files are created in `labels/train`
- background images are preserved in all later cumulative `_bg` datasets
- background images are not added to `val`, `holdout`, or `holdout_v2`

This matches the existing historical `_bg` dataset convention.

## Output Structure

`subsets_v2` contains stage-level split JSON files that preserve the existing subset schema:

- `train_val_holdout_subsets_batches_1_3.json`
- `train_val_holdout_subsets_batches_4_5.json`
- `train_val_holdout_subsets_batches_6_7.json`
- `train_val_holdout_subsets_batches_8_10.json`
- `train_val_holdout_subsets_batches_11.json`
- `train_val_holdout_subsets_batches_12.json`
- `regeneration_manifest.json`

Why stage-level subset JSONs and not cumulative subset JSONs:

- the current subset JSON contract is only `train` / `val` / `holdout`
- cumulative merged datasets also materialize `holdout_v2`
- adding `holdout_v2` to subset JSONs would change the existing schema
- the cumulative `holdout_v2` state is therefore represented in the dataset folders and in the regeneration manifest, not in a new subset JSON format

Each output dataset is validated after writing by comparing the actual dataset contents against the expected sample IDs for:

- `train`
- `val`
- `holdout`
- `holdout_v2` when applicable

## Filename Compatibility Notes

The regeneration flow extends the anti-leakage filename parser to accept valid annotated filenames that end with duplicate-style suffixes such as:

- `4.2_H_cam_262_spacesMR_324 (1).tif`

This change is backward-compatible with the existing parser behavior and is required because those files exist in the actual annotated source batches.

## Run

Dry validation only:

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/regenerate_yolo_datasets_v2.py \
  --dry-run
```

Full regeneration:

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/regenerate_yolo_datasets_v2.py \
  --overwrite
```

If you need to tolerate converter-reported image or annotation issues instead of failing the run:

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/regenerate_yolo_datasets_v2.py \
  --overwrite \
  --allow-annotation-issues
```

## Reporting

During execution the script logs, for each generated cumulative dataset:

- dataset name
- included batches
- whether background is included
- `train` / `val` / `holdout` / `holdout_v2` counts
- output path

It also writes:

- `regeneration_manifest.json` in `subsets_v2`
- `regeneration_summary.json` inside each cumulative dataset
- `background_append_summary.json` inside `batch1_3_bg`
- native `conversion_summary.json` and `merge_summary.json` from the reused production utilities

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
