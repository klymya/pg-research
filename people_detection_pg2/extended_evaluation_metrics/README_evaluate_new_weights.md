# Evaluate New Weights

Step-by-step guide for running the extended evaluation pipeline against a **new set of trained weights** (PyTorch and/or TFLite). Intended for engineers and agents who need to produce metrics, per-frame/per-detection CSVs, visualizations, and the HTML report for a new model.

For attribute-CSV generation (holdout preprocessing) and the CVAT↔YOLO join, see [README.md](README.md).

---

## 1. What the pipeline does

`run_eval_and_viz.sh` orchestrates, in order:

1. [evaluation.py](evaluation.py) — runs Ultralytics validation (single-run or `(conf × iou)` sweep), writes `predictions.json`, `metrics_per_img.csv`, `metrics_per_gt.csv`, `metrics_per_pred.csv`, `global_metrics.json`, `sweep_results.csv`, `best_params.json`.
2. [visualize_detections.py](visualize_detections.py) — renders per-frame overlays (GT/Pred, TP/FP/FN colors, legs, scores) into `VIZ_OUT_DIR`.
3. Copies [extended_report.ipynb](extended_report.ipynb) into `PROJECT_PTH`, executes it with `nbconvert`, and exports HTML.
4. Optionally runs [generate_conclusions_claude.py](generate_conclusions_claude.py) to write Section G, then re-executes the notebook for the final HTML.

Everything is driven by a single `.env` file passed to the script.

---

## 2. Required inputs

### 2.1 Trained weights (per experiment)

The evaluator resolves weight paths from `BASE_EXPERIMENT_DIR` (or from explicit `WEIGHTS_PTH`/`TFLITE_*` overrides). The expected layout under `${BASE_EXPERIMENT_DIR}/weights/` is:

| Backend (`EVAL_BACKEND`) | Files required |
| --- | --- |
| `ultralytics_pt` | `weights/best.pt` |
| `tflite_mixed` | `weights/best.pt` (validator metadata only) <br> `weights/best_saved_model/best_full_integer_quant.tflite` (INT8 backbone) <br> `weights/head_saved_model/head_float32.tflite` (FP32 head) |
| `tflite_backbone_only` | `weights/best.pt` (validator metadata only) <br> `weights/best_non_separated_saved_model/best_full_integer_quant.tflite` (fully INT8) <br> optional: `TFLITE_OUTPUT_CORRECTION_MODEL_PTH` JSON |

Optional per-experiment file: `${BASE_EXPERIMENT_DIR}/inference_params.json` with `{"conf_th": <float>, "iou_th": <float>}`. When `USE_INFERENCE_PARAMS=1` and `ENABLE_SWEEP=0`, these override `CONF_TH` / `IOU_TH`.

### 2.2 Evaluation dataset (YOLO layout)

At `${BASE_DATASET_PTH}`:

```
project_holdout.yaml          # ultralytics data yaml — must set val: images/<SUBSET_NAME>
images/<SUBSET_NAME>/         # holdout images
labels/<SUBSET_NAME>/         # YOLO .txt labels (kpt_shape: [1, 2] for pose)
```

Reference dataset: `/home/klymy/data/cam4_revB/yolo_dataset/<DATASET_NAME>/`. `project_holdout.yaml` example:

```yaml
path: /home/klymy/data/cam4_revB/yolo_dataset/<DATASET_NAME>
train: images/train
val: images/holdout
kpt_shape: [1, 2]
names:
  0: person
```

### 2.3 Holdout attribute CSVs (for Sections D–F of the report)

- `frames_attributes.csv` — `yolo_id, camera_height, num_gt_detections, scene_density, …`
- `detections_attributes.csv` — `yolo_id, det_index, x_c, y_c, w, h, x, y, posture, occluded_by_object, occluded_by_person, …`

Reuse one of the existing bundles for the corresponding holdout:

| Holdout | Directory |
| --- | --- |
| `batch1_3` (default cam4 RevB) | [holdout_data/](holdout_data/) |
| `batch_testAccuracy` (baseline test set) | [holdout_data_testAccuracy/](holdout_data_testAccuracy/) |
| `H270_16-44-09` | [holdout_H270_16-44-09/](holdout_H270_16-44-09/) |
| `PG2HardCases` | [holdout_PG2HardCases/](holdout_PG2HardCases/) |

If you evaluate on a **new holdout**, regenerate these two CSVs first with [join_holdout_set_data.py](join_holdout_set_data.py) and [export_holdout_attributes_csv.py](export_holdout_attributes_csv.py) (see [README.md](README.md)).

### 2.4 Python environment

Same env used by the training pipeline (`ultralytics`, `torch`, `tensorflow`, `python-dotenv`, `pandas`, `numpy`, `tqdm`, `matplotlib`, `opencv-python`, `Pillow`, `jupyter`, `nbconvert`). For Claude conclusions: `langchain-anthropic`.

TFLite backends optionally use the emulator inferer at `/home/klymy/projects/algo_system_emulator/src` (see [evaluation.py:283](evaluation.py#L283-L300)). If unavailable, it falls back to [../tflite_inference.py](../tflite_inference.py).

---

## 3. Create the `.env` config

Copy the template and edit only what changes for the new weights.

```bash
cp people_detection_pg2/extended_evaluation_metrics/.env.example \
   people_detection_pg2/extended_evaluation_metrics/.env.<my_new_experiment>
```

### 3.1 Minimum edits per new model

| Variable | Meaning |
| --- | --- |
| `EXP_NAME` | Experiment folder name under `people_detection_pg2/results/` |
| `DATASET_NAME` | YOLO dataset folder name under `/home/klymy/data/cam4_revB/yolo_dataset/` |
| `BASE_EXPERIMENT_DIR` | Absolute path to the experiment folder (defaults to `${REPORT_BASE}/../results/${EXP_NAME}`) |
| `BASE_DATASET_PTH` | Absolute path to the YOLO dataset root |
| `SUBSET_NAME` | Split under `images/` and `labels/` used for validation (default: `holdout`) |
| `EVAL_BACKEND` | `ultralytics_pt` / `tflite_mixed` / `tflite_backbone_only` |
| `INPUT_WIDTH`, `INPUT_HEIGHT` | Model input size (default `320`) |
| `FRAMES_ATTRS`, `DETECTIONS_ATTRS` | Point to the holdout bundle matching this dataset |
| `PROJECT`, `PROJECT_NAME` | Output subfolder under `REPORT_BASE`; `PROJECT_NAME` is used as the Ultralytics `name=` |

Full annotated template: [.env.example](.env.example). Concrete per-experiment examples: [generated_envs/dataset_v2_batch1/](generated_envs/dataset_v2_batch1/) and [.env.H270_16-44-09.batch1_10](.env.H270_16-44-09.batch1_10).

### 3.2 Threshold selection

Single run (default):

```dotenv
ENABLE_SWEEP=0
USE_INFERENCE_PARAMS=1      # use inference_params.json if present
CONF_TH=0.35                # fallback if USE_INFERENCE_PARAMS=0 or file missing
IOU_TH=0.50
```

Grid sweep (each `(conf, iou)` pair is a fresh validation pass; the best pair wins by `SELECTION_METRIC`):

```dotenv
ENABLE_SWEEP=1
SWEEP_CONF_THS=0.20,0.25,0.30
SWEEP_IOU_THS=0.35,0.40,0.45
SELECTION_METRIC=f1         # f1 | recall | precision
```

Selection uses validator pose metrics (`metrics/precision(P)`, `metrics/recall(P)`, derived F1) when available (see [evaluation.py:98](evaluation.py#L98-L116)).

### 3.3 TFLite-specific keys

TFLite paths are auto-derived from `BASE_EXPERIMENT_DIR`. Override only if the layout differs:

```dotenv
# tflite_mixed
# TFLITE_BACKBONE_PTH=${BASE_EXPERIMENT_DIR}/weights/best_saved_model/best_full_integer_quant.tflite
# TFLITE_HEAD_PTH=${BASE_EXPERIMENT_DIR}/weights/head_saved_model/head_float32.tflite

# tflite_backbone_only
# TFLITE_BACKBONE_PTH=${BASE_EXPERIMENT_DIR}/weights/best_non_separated_saved_model/best_full_integer_quant.tflite
TFLITE_OUTPUT_CORRECTION_MODEL_PTH=""   # optional; JSON with ridge-regression params
```

### 3.4 Report / Claude conclusions

```dotenv
TOP_N=6
F1_BINS=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
RUN_CONCLUSIONS_AGENT=0     # set to 1 to run Claude section-G generation
ANTHROPIC_API_KEY=""        # required only when RUN_CONCLUSIONS_AGENT=1
```

Do not commit `ANTHROPIC_API_KEY`. Load it from a shell env var or a secret store; keep the value out of tracked `.env*` files.

---

## 4. Run the evaluation

### 4.1 One experiment (recommended entry point)

```bash
./people_detection_pg2/extended_evaluation_metrics/run_eval_and_viz.sh \
    people_detection_pg2/extended_evaluation_metrics/.env.<my_new_experiment>
```

This runs steps 1–4 from Section 1. The env file path must exist and be bash-parseable (no spaces around `=`).

### 4.2 Just the metrics (skip viz + notebook)

```bash
ENV_PATH=people_detection_pg2/extended_evaluation_metrics/.env.<my_new_experiment> \
python people_detection_pg2/extended_evaluation_metrics/evaluation.py
# or
python people_detection_pg2/extended_evaluation_metrics/evaluation.py \
    --env-file people_detection_pg2/extended_evaluation_metrics/.env.<my_new_experiment>
```

### 4.3 Batch across many experiments

Use [../run_batch_evaluation_v2.py](../run_batch_evaluation_v2.py) to discover experiments under `people_detection_pg2/results/`, generate `.env` files from a template, and invoke `run_eval_and_viz.sh` per experiment.

```bash
# TFLite backbone-only (default backend)
python people_detection_pg2/run_batch_evaluation_v2.py \
    --pattern "people_detection_pg2/results/dataset_v2_cam4_revB_batch1_*" \
    --template-env people_detection_pg2/extended_evaluation_metrics/.env.H270_16-44-09.batch1_10 \
    --project val_dataset_v2_batch1 \
    --generated-env-dir people_detection_pg2/extended_evaluation_metrics/generated_envs/dataset_v2_batch1 \
    --log-dir people_detection_pg2/extended_evaluation_metrics/logs/dataset_v2_batch1

# FP32 baseline (ultralytics_pt)
python people_detection_pg2/run_batch_evaluation_v2.py --evaluate-fp32 ...

# Mixed INT8 backbone + FP32 head
python people_detection_pg2/run_batch_evaluation_v2.py --evaluate-tflite-mixed ...

# Dry run — print planned env files without launching evaluations
python people_detection_pg2/run_batch_evaluation_v2.py --dry-run ...
```

Generated per-experiment envs are written to `--generated-env-dir`; logs to `--log-dir`.

---

## 5. Outputs

All artifacts land under `${PROJECT_PTH}` = `${REPORT_BASE}/${PROJECT}/${PROJECT_NAME}`.

```
${PROJECT_PTH}/
├── predictions.json          # best-run predictions (Ultralytics schema)
├── global_metrics.json       # winning summary (validator metrics + local detection metrics + selection/*)
├── best_params.json          # winning conf/iou, run_dir, num_candidates, mode
├── sweep_results.csv         # all evaluated (conf, iou) rows (sweep or single-run)
├── metrics_per_img.csv       # per-frame P/R/F1/AP
├── metrics_per_gt.csv        # GT-centric TP/FN with match_id, gt_*, matched_pred_index, attribute joins
├── metrics_per_pred.csv      # Pred-centric TP/FP with match_id, score, pred_*, matched_gt_index
├── runs/<run_name>/          # per (conf, iou) run: predictions.json + Ultralytics validator outputs
├── viz/                      # per-frame overlays (VIZ_OUT_DIR)
├── extended_report.ipynb     # executed copy of the notebook
└── extended_report.html      # exported HTML (final version when RUN_CONCLUSIONS_AGENT=1)
```

The winning run's `predictions.json` is copied up to `${PROJECT_PTH}/predictions.json`. The best summary is written to both `global_metrics.json` and `best_params.json`. `sweep_results.csv` is sorted by the selection metric so the top row is the winner.

---

## 6. Verification checklist (before shipping numbers)

- `${BASE_EXPERIMENT_DIR}/weights/best.pt` exists (required as validator metadata even for TFLite backends).
- For TFLite: the specific `.tflite` files listed in Section 2.1 exist.
- `${BASE_DATASET_PTH}/project_holdout.yaml` resolves; `images/<SUBSET_NAME>/` and `labels/<SUBSET_NAME>/` are non-empty.
- `FRAMES_ATTRS` and `DETECTIONS_ATTRS` point to CSVs that match the holdout being evaluated (same `yolo_id`s).
- After the run: `metrics_per_img.csv` row count equals the number of images in `images/<SUBSET_NAME>/`; the log does not report a large `Warning: N/M images have no entries in predictions.json`.
- `best_params.json` `selection_source` is `validator_pose_metrics` (preferred) or `local_detection_metrics` (fallback — investigate if unexpected).
