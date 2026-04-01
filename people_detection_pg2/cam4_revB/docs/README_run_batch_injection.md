# One-Click Batch Injection Pipeline

This utility provides the post-review orchestration entrypoint for `cam4_revB`:

- [run_batch_injection.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/run_batch_injection.py)

It implements the design from [post_review_batch_injection_pipeline_review.md](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/post_review_batch_injection_pipeline_review.md).

## What It Runs

After Step 0 manual review, the script runs:

1. anti-leakage overlap and split generation
2. CVAT-to-YOLO conversion
3. border-box filtering
4. optional dataset preprocessing
5. optional dataset merge

## Review Gate

Execution requires:

- `--review-confirmed`
- `--review-note`

This is recorded in the run manifest.

## Example

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/run_batch_injection.py \
  --root-pth /home/klymy/data/cam4_revB \
  --run-name batch12_post_review \
  --new-batch-root /home/klymy/data/cam4_revB/4.2_batch12 \
  --protected-subsets-json /home/klymy/data/cam4_revB/train_val_holdout_subsets_batches_1_3.json \
  --annotation-xml /home/klymy/data/cam4_revB/batch12_corrected/batch12_corrected.xml \
  --review-confirmed \
  --review-note "Reviewed in CVAT and sample overlays checked" \
  --converted-dataset-name batch12 \
  --preprocess-mode wb_alif_clahe \
  --merge-base-dataset batch1_11_margin_15_bg_wb_alif_clahe \
  --merge-result-dataset batch1_12_margin_15_bg_wb_alif_clahe \
  --overwrite
```

## Main Outputs

Run artifacts are written under:

- `<root_pth>/batch_injection_runs/<run_name>`

Main manifest:

- `batch_injection_run_manifest.json`

The manifest records:

- review confirmation
- CLI settings
- per-step status
- per-step artifact paths
- final dataset paths
- failure details if any stage fails

## Optional Modes

Preprocessing:

- `none`
- `wb_alif`
- `wb_alif_clahe`

Merge:

- skipped when `--merge-base-dataset` is not provided
- enabled when both `--merge-base-dataset` and `--merge-result-dataset` are provided

## Tests

- [test_run_batch_injection.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_run_batch_injection.py)
