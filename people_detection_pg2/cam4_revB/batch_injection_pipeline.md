# Pipeline of processing new batches injection

0. `../../annotation_poc/init_fix_batches_8_10.py` + `../../annotation_poc/annotation_viz_cam4_revB.ipynb`
1. `find_intersection_batch1_3_vs_4_5.ipynb`
2. `subsets_splitting_batch4_5.ipynb`
3. `cvat2yolo_convert.py` (production) or `cvat2yolo.ipynb` (reference)
4. `filter_small_bboxes.py` (production) or `filter_small_bboxes_v2.ipynb` (reference)
5. `white_balance_alif.ipynb`
6. `update_dataset.ipynb`

Notes:
- `cvat2yolo_convert.py` is the current production implementation of the CVAT-to-YOLO conversion step.
- `cvat2yolo.ipynb` remains as the historical notebook reference.
- `filter_small_bboxes.ipynb` is outdated and should not be used for new runs.
- Reason: the old notebook uses a `min_margin` formula that does not correctly represent the bbox overlap with the inner image region.
- `filter_small_bboxes_v2.ipynb` introduced the intersection-area logic, and [filter_small_bboxes.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes.py) is the parameterized script replacement for pipeline use.
- In the current script, a bbox is filtered out when the intersection area with the inner image box is less than `1%` of the bbox area, or when bbox width/height is non-positive.
