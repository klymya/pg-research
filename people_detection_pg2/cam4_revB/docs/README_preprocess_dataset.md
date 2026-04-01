# Dataset Preprocessing Utility

This utility productionizes Step 5 of the `cam4_revB` batch injection flow:

- [preprocess_dataset.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py)

It replaces the dataset-level processing logic that was previously notebook-driven in [white_balance_alif.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/white_balance_alif.ipynb).

## Behavior

- copies an existing YOLO dataset into a new dataset variant
- removes stale `*.cache` files
- refreshes `project.yaml`, `project_holdout.yaml`, and `project_holdout_v2.yaml` when present
- applies one preprocessing mode to all `.jpg` images under `images/**`
- writes `preprocess_summary.json`

Supported modes:

- `wb_alif`
- `wb_alif_clahe`

Default output naming:

- `<base_dataset>_wb_alif`
- `<base_dataset>_wb_alif_clahe`

## Example

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py \
  --base-pth /home/klymy/data/cam4_revB/yolo_dataset \
  --base-dataset batch_H270_16-44-09_margin_10 \
  --mode wb_alif_clahe \
  --overwrite
```

## Outputs

Inside the output dataset:

- refreshed project YAML files
- processed images under the same split structure
- `preprocess_summary.json`

The summary includes:

- input/output dataset paths
- preprocessing mode
- image counts
- issue counts
- removed cache count

## Tests

- [test_preprocess_dataset.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_preprocess_dataset.py)
