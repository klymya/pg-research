# Border-Box Filtering Utility

This utility replaces the notebook-only border-box filtering step from [filter_small_bboxes_v2.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes_v2.ipynb) with a parameterized Python script:

- [filter_small_bboxes.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes.py)

It follows the approved design in [filter_small_bboxes_v2_review.md](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes_v2_review.md).

## Behavior

- copies the source YOLO dataset into a new dataset directory
- removes stale `*.cache` files
- regenerates `project.yaml` and `project_holdout.yaml`
- filters label lines using the inner-box intersection rule
- writes additive audit reports:
  - `filter_report.json`
  - `removed_labels.csv`
  - `split_summary.csv`

Current rule:

- `inside_ratio = intersection_area(bbox, inner_box) / bbox_area`
- keep when `inside_ratio >= min_inside_ratio`
- remove when `inside_ratio < min_inside_ratio`
- always remove invalid boxes with `width <= 0` or `height <= 0`

Default parameters:

- `margin = 0.10`
- `min_inside_ratio = 0.01`

## Command

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes.py \
  --base-pth /home/klymy/data/cam4_revB/yolo_dataset \
  --base-dataset batch_H270_16-44-09 \
  --overwrite
```

This creates the default output dataset:

```text
/home/klymy/data/cam4_revB/yolo_dataset/batch_H270_16-44-09_margin_10
```

Optional parameters:

- `--output-dataset`
- `--margin`
- `--min-inside-ratio`
- `--max-workers`
- `--report-dir`
- `--overwrite`

## Outputs

By default, reports are written into the output dataset root.

- `filter_report.json`
  - config used for the run
  - total label counts
  - per-split counts
  - invalid bbox counts
- `removed_labels.csv`
  - one row per removed label with split, path, line number, bbox values, `inside_ratio`, and reason
- `split_summary.csv`
  - one row per split with file counts, total labels, removed labels, and removal rate

## Tests

Pytest coverage is provided in:

- [test_filter_small_bboxes.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_filter_small_bboxes.py)

Run:

```bash
pytest /home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_filter_small_bboxes.py
```
