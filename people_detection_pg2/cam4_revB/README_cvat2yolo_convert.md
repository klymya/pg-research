# CVAT-To-YOLO Conversion Utility

This utility replaces the notebook-only conversion step from [cvat2yolo.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/cvat2yolo.ipynb) with a reusable Python script:

- [cvat2yolo_convert.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/cvat2yolo_convert.py)

The design and migration rationale are documented in [cvat2yolo_review.md](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/cvat2yolo_review.md).

## Behavior

- reads corrected CVAT XML files
- reads a precomputed `train` / `val` / `holdout` subsets JSON
- discovers source images from one or more raw batch folders
- converts annotations into YOLO pose labels with one class and one keypoint
- copies and converts source images into `.jpg`
- writes:
  - `images/train`, `images/val`, `images/holdout`
  - `labels/train`, `labels/val`, `labels/holdout`
  - `project.yaml`
  - `project_holdout.yaml`
- writes additive audit artifacts:
  - `conversion_summary.json`
  - `image_issues.csv`
  - `invalid_annotations.csv`

## Backward-Compatible Contract

The script preserves the current `cam4_revB` dataset contract:

- output sample IDs are derived from the raw image path relative to the root path
- sample IDs keep the current pattern:
  `4.2_batch12____4.2_H_cam_248_viasCEO_5`
- labels keep the current YOLO pose format:
  `cls x_center y_center width height keypoint_x keypoint_y`
- label mapping stays:
  `{0: "person"}`
- output images are written as `.jpg`
- split assignment comes only from the existing subsets JSON

## Validation

The converter explicitly validates and reports:

- split entries that point to missing source images
- source images present in split JSON but missing from annotations
- annotation images that are missing on disk or absent from the split
- invalid image width or height
- malformed or ambiguous object IDs
- invalid bbox geometry
- non-finite coordinates
- normalized coordinates outside `[0, 1]`

By default, invalid items are skipped and reported. The script does not silently clamp invalid coordinates.

For stricter pipeline enforcement, use:

- `--fail-on-issues`

In this mode, the converter fails before writing output if it detects any image issues or invalid annotations.

## Example

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/cvat2yolo_convert.py \
  --root-pth /home/klymy/data/cam4_revB \
  --annotation-xml /home/klymy/data/cam4_revB/batch_H270_16-44-09_corrected/batch_H270_16-44-09_corrected.xml \
  --image-root /home/klymy/data/cam4_revB/4.2_batch_H270_16-44-09 \
  --subsets-json /home/klymy/data/cam4_revB/train_val_holdout_subsets_batches_H270_16-44-09.json \
  --dataset-name batch_H270_16-44-09 \
  --fail-on-issues \
  --overwrite
```

Output dataset:

```text
/home/klymy/data/cam4_revB/yolo_dataset/batch_H270_16-44-09
```

## Outputs

`conversion_summary.json` includes:

- input paths
- output dataset path
- total discovered images
- total split entries
- total converted images
- total objects written
- per-split image and object counts
- per-split `samples_with_dropped_objects`
- counts of image issues and invalid annotations by reason

`image_issues.csv` includes:

- split
- absolute path
- relative key
- reason
- details

`invalid_annotations.csv` includes:

- XML path
- image name
- object ID
- reason
- details

## Tests

Pytest coverage is provided in:

- [test_cvat2yolo_convert.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_cvat2yolo_convert.py)

Run:

```bash
pytest /home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_cvat2yolo_convert.py
```
