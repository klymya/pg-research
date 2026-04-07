# Anti-Leakage Split Utility

Main behaviors:

- parse `raw_scene_group_id` from raw batch paths or YOLO-style filenames
- detect historical protected holdout groups from older subsets JSON files
- assign new-batch files to `holdout` by exact `raw_scene_group_id` match
- split remaining files into `train` and `val` by group, not by frame
- reject and report malformed filename patterns
- validate that no group appears across `train`, `val`, and `holdout`

## Grouping Rule

One group is one filename prefix before the final frame index.

Examples:

- `4.2_batch10____4.2_H_cam_229_Kitchen_1045.txt`
  -> `raw_scene_group_id = 4.2_h_cam_229_kitchen`
- `4.2_batch8____4.2_H_cam_282_MenivimOS_4HO_17.txt`
  -> `raw_scene_group_id = 4.2_h_cam_282_menivimos_4ho`
- `4.2_H260_me_2.tif`
  -> `raw_scene_group_id = 4.2_h260_me`

Malformed names such as `4.2_H_cam___12.tif` are rejected and reported.

## Command

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/antileakage_split.py \
  --new-batch-root /home/klymy/data/cam4_revB/4.2_batch12 \
  --protected-subsets-json /home/klymy/data/cam4_revB/train_val_holdout_subsets_batches_1_3.json \
  --output-subsets-json /tmp/cam4_batch12_grouped_subsets.json \
  --output-overlap-json /tmp/cam4_batch12_overlap.json \
  --output-summary-json /tmp/cam4_batch12_summary.json
```

Outputs:

- subsets JSON with `train`, `val`, `holdout`
- overlap report JSON
- summary JSON with validation results

## Tests

Pytest coverage is provided in:

- [test_antileakage_split.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_antileakage_split.py)

Run:

```bash
pytest /home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_antileakage_split.py
```

Covered cases:

- valid historical naming variants
- malformed-name rejection
- malformed historical holdout reporting
- zero train/val group overlap after splitting
- validation failure when malformed paths are present
