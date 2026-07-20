# Dataset Preprocessing Utility

This utility productionizes Step 5 of the `cam4_revB` batch injection flow:

- [preprocess_dataset.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py)

It replaces the dataset-level processing logic that was previously notebook-driven.

## Pipeline

Per image, in order:

```
raw -> ALIF white balance -> CLAHE -> sharpness augmentation
```

- **ALIF white balance** is implemented in [alif_awb.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/alif_awb.py) as `white_balance_vectorized`. It is a fixed 3×3 sensor correction matrix plus a per-image gray-world scaling, followed by an sRGB OETF gamma applied via a 256-entry LUT. It is deterministic and pixel-wise vectorized.
- **CLAHE** is `cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))` applied to the L channel of Lab.
- **Sharpness augmentation** — see the section below.

Which stages actually run is controlled by `--mode`:

| `--mode` | white balance | CLAHE | sharpness augmentation (when enabled) |
|---|:---:|:---:|:---:|
| `wb_alif` | ✓ | — | ✓ |
| `wb_alif_clahe` | ✓ | ✓ | ✓ |
| `clahe_only` | — | ✓ | ✓ |

`clahe_only` is a legacy mode; it assumes the input has already been white-balanced elsewhere.

## Sharpness Augmentation

After CLAHE, each image is optionally softened or sharpened by a per-image random delta drawn from `Uniform(-range, +range)` percent. The operator and its parameter come from the calibrated lookup table in [sharpness_tools.py](/home/klymy/projects/researches/people_detection_pg2/investigations/release_june_26/sharpness_research/sharpness_tools.py) (`CALIBRATION_LAPLACIAN`), which was derived against the ceiling-camera reference in the `release_june_26` investigation and covers ±15% of Laplacian-variance sharpness.

- **Enabled by default.** Turn it off with `--no-sharpness-augmentation`.
- **Range** is set with `--sharpness-range-pct` (default `10.0`, clamped to `[0, 15]`).
- **Seed** is set with `--augmentation-seed` (default `42`). Each image's delta is drawn from a `random.Random` seeded with `crc32(relative_path) ^ base_seed`, so the augmentation is:
  - deterministic across runs (same source dataset + same seed → same deltas),
  - independent per image (no ordering effects with the thread pool),
  - stable across output-dataset renames (the seed key is the path *inside* the output dataset).
- Applied to every image in every split (`train`, `val`, `holdout`). If you want the augmentation constrained to train, disable it with `--no-sharpness-augmentation` and add a separate pass over `images/train/` yourself.

## Other Behavior

- copies an existing YOLO dataset into a new dataset variant
- removes stale `*.cache` files
- refreshes `project.yaml`, `project_holdout.yaml`, and `project_holdout_v2.yaml` when present
- applies the pipeline to all `.jpg` images under `images/**`
- writes `preprocess_summary.json`

Default output naming:

- `<base_dataset>_wb_alif`
- `<base_dataset>_wb_alif_clahe`
- `<base_dataset>_clahe`

## Example

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py \
  --base-pth /home/klymy/data/cam4_revB/yolo_dataset \
  --base-dataset batch_H270_16-44-09_margin_10 \
  --mode wb_alif_clahe \
  --overwrite
```

Disable augmentation for a canonical (no-augmentation) build:

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py \
  --base-pth /home/klymy/data/cam4_revB/yolo_dataset \
  --base-dataset batch_H270_16-44-09_margin_10 \
  --mode wb_alif_clahe \
  --no-sharpness-augmentation \
  --overwrite
```

Narrow the augmentation range and switch to a different seed for an ablation study:

```bash
python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py \
  --base-pth /home/klymy/data/cam4_revB/yolo_dataset \
  --base-dataset batch_H270_16-44-09_margin_10 \
  --mode wb_alif_clahe \
  --sharpness-range-pct 5 \
  --augmentation-seed 2026 \
  --overwrite
```

## Outputs

Inside the output dataset:

- refreshed project YAML files
- processed images under the same split structure
- `preprocess_summary.json`

The summary now includes:

- input/output dataset paths
- preprocessing mode
- image counts
- issue counts
- removed cache count
- **augmentation config**: `sharpness_augmentation`, `sharpness_range_pct`, `augmentation_seed`
- **augmentation stats** (when enabled): `count`, `min_delta_pct`, `max_delta_pct`, `mean_delta_pct`, `stdev_delta_pct`

## Tests

- [test_preprocess_dataset.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/tests/test_preprocess_dataset.py)

Covers the pre-existing behavior plus:

- augmentation enabled by default and stats reported,
- `--no-sharpness-augmentation` produces null stats,
- same seed + same source → identical per-image deltas,
- `sharpness_range_pct > 15` raises `ValueError`.
