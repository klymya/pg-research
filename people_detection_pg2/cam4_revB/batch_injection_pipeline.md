# Pipeline of processing new batches injection

0. `../../annotation_poc/init_fix_batches_8_10.py` + `../../annotation_poc/annotation_viz_cam4_revB.ipynb`
1. `find_intersection_batch1_3_vs_4_5.ipynb`
2. `subsets_splitting_batch4_5.ipynb`
3. `cvat2yolo.ipynb`
4. `filter_small_bboxes_v2.ipynb`
5. `white_balance_alif.ipynb`
6. `update_dataset.ipynb`

Notes:
- `filter_small_bboxes.ipynb` is outdated and should not be used for new runs.
- Reason: the old notebook uses a `min_margin` formula that does not correctly represent the bbox overlap with the inner image region.
- `filter_small_bboxes_v2.ipynb` replaces that logic with an explicit intersection-area check between the bbox and the inner image box.
- In v2, a bbox is filtered out when the intersection area with the inner image box is less than or equal to `10%` of the bbox area.
