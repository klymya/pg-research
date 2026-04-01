# Pipeline of processing new batches injection

0. `../../annotation_poc/init_fix_batches_8_10.py` + `../../annotation_poc/annotation_viz_cam4_revB.ipynb`
1. `run_batch_injection.py` (production orchestration)
2. `antileakage_split.py` (production overlap + split)
3. `cvat2yolo_convert.py` (production conversion) or `cvat2yolo.ipynb` (reference)
4. `filter_small_bboxes.py` (production filtering) or `filter_small_bboxes_v2.ipynb` (reference)
5. `preprocess_dataset.py` (production preprocessing) or `white_balance_alif.ipynb` (reference)
6. `merge_yolo_datasets.py` (production merge) or `update_dataset.ipynb` (reference)

Notes:
- `run_batch_injection.py` is the current one-click post-review entrypoint.
- `antileakage_split.py` replaces the old overlap/split notebooks for production use.
- `cvat2yolo_convert.py` is the current production implementation of the CVAT-to-YOLO conversion step.
- `cvat2yolo.ipynb` remains as the historical notebook reference.
- `filter_small_bboxes.ipynb` is outdated and should not be used for new runs.
- Reason: the old notebook uses a `min_margin` formula that does not correctly represent the bbox overlap with the inner image region.
- `filter_small_bboxes_v2.ipynb` introduced the intersection-area logic, and [filter_small_bboxes.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes.py) is the parameterized script replacement for pipeline use.
- In the current script, a bbox is filtered out when the intersection area with the inner image box is less than `1%` of the bbox area, or when bbox width/height is non-positive.
- `preprocess_dataset.py` and `merge_yolo_datasets.py` are the production replacements for the notebook-only preprocessing and merge steps.
