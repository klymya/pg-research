# YOLO Dataset Merge Utility

This utility productionizes Step 6 of the `cam4_revB` batch injection flow:

- [merge_yolo_datasets.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/merge_yolo_datasets.py)

It replaces the notebook-only merge logic from [update_dataset.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/update_dataset.ipynb).

## Behavior

- copies a base dataset into a result dataset
- merges `train` and `val` images/labels from an update dataset
- preserves the current `holdout_v2` convention:
  - base `holdout` is copied into `holdout_v2`
  - update `holdout` is then appended into `holdout_v2`
- refreshes project YAML files
- writes `merge_summary.json`

Duplicate handling:

- identical duplicate files are treated as already present
- conflicting duplicate files raise an error

## Example

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/merge_yolo_datasets.py \
  --base-pth /home/klymy/data/cam4_revB/yolo_dataset \
  --base-dataset batch1_11_margin_15_bg_wb_alif \
  --update-dataset batch12_margin_15_wb_alif \
  --result-dataset batch1_12_margin_15_bg_wb_alif \
  --overwrite
```

## Outputs

Inside the result dataset:

- merged train/val contents
- merged `holdout_v2`
- refreshed `project.yaml`
- refreshed `project_holdout.yaml`
- refreshed `project_holdout_v2.yaml`
- `merge_summary.json`

## Tests

- [test_merge_yolo_datasets.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_merge_yolo_datasets.py)
