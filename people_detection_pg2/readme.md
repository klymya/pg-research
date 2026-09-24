# people_detection_pg2

YOLOv8/YOLO11 people detection for the Alif Ensemble MCU: training, int8 quantisation and Vela
compilation, plus the dataset pipeline, evaluation harness and the tool that ships weights to
the firmware repo. This is the repo's main active project.

**Status:** ACTIVE · last change 2026-09 · 12 GB on disk

## Purpose

Produce a people detector that runs on an Ethos-U55 microcontroller within the sensor's memory
and latency budget, prove its accuracy before release, and deliver it to the firmware.

The awkward part is quantisation: upstream ultralytics' detection head does not quantise to
int8 correctly. The `DeGirum/ultralytics_yolov8` fork removes the head from the network, so the
**backbone is exported int8** and the **head is exported fp32 separately**, then both are
Vela-compiled and shipped as a pair.

---

## Model configs (custom, not upstream)

The `*.yaml` files here are **modified ultralytics configs**, and the important change is a
non-standard **`t` (tiny) scale that does not exist upstream**. Each file leaves exactly one
scale uncommented, so the filename picks the size:

| Config | Scale | depth | **width** | max_ch | Use |
|---|---|---:|---:|---:|---|
| `yolov8-pose-tiny.yaml` | `t` | 0.33 | **0.21** | 1024 | **Default.** Smallest — the size targeted for the MCU |
| `yolov8-pose.yaml` | `n` | 0.33 | **0.25** | 1024 | **Second option.** Standard nano; larger and more accurate |
| `yolov8-pose-xs.yaml` | `xs` | 0.33 | 0.29 | 1024 | Between nano and small; rarely used |
| `yolo11-pose-tiny.yaml` | `t` | 0.50 | 0.21 | 1024 | YOLO11 equivalent of tiny |
| `yolo11-pose.yaml` | `n` | 0.50 | 0.25 | 1024 | YOLO11 nano |

**Width is the lever.** `t` is 0.21 against nano's 0.25 — ~16% narrower, which is what brings
the model inside the Ethos-U55 budget. Depth is unchanged.

Both YOLOv8 configs are adapted for this task: `nc: 1` (person only, not COCO's 80) and
`kpt_shape: [1, 2]` (one keypoint, x/y — the leg/ground point used for px→cm projection).

Which gets used in practice, counted from the committed experiment definitions:

| Config | `run_training_pipeline_example.sh` | `experiments.json` |
|---|---:|---:|
| `yolov8-pose-tiny.yaml` | 13 | 6 |
| `yolov8-pose.yaml` | 10 | 6 |

Experiment names encode the choice: runs built from `yolov8-pose.yaml` carry `yolov8n` in the
name (`..._batch1_10_margin_15_yolov8n_wb_alif_clahe_...`); tiny runs omit it.

> **Known inconsistency:** `yolo11-pose-tiny.yaml` and `yolo11-pose.yaml` still declare
> `nc: 80` — COCO's class count — while the YOLOv8 configs correctly use `nc: 1`. The YOLO11
> configs were copied from upstream and only the scales were edited. Fix `nc` before training
> with them, or the model will carry 80 output classes it does not need.

---

## Working pipeline

The whole chain is automated. Four scripts do the work end to end.

### 1. Train + convert — `run_training_pipeline.py`

**One command takes a dataset to Vela-compiled artefacts.** It orchestrates three sub-steps
via subprocess, so you do not run them by hand:

```
run_training_pipeline.py
   ├─ train_yolo.py            ultralytics training  -> results/<exp_name>/weights/best.pt
   ├─ export.py                int8 backbone + fp32 head (separated)  -> weights/best_saved_model/
   ├─ optimize_vela.py         Vela compile for Ethos-U55             -> weights/vela_he/
   ├─ export_non_separated.py  combined-head variant                  -> weights/<--converted-output-folder-name>/
   └─ optimize_vela.py         Vela compile that variant (--skip-head)
```

```bash
python run_training_pipeline.py \
    --dataset  /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_10_margin_15_yolov8n_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights  results/filtered_v3_margin_15_yolov8n/weights/best.pt \
    --cfg      yolov8-pose.yaml \
    --device   0 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model
```

`--cfg` selects the model **architecture** — `yolov8-pose-tiny.yaml` by default,
`yolov8-pose.yaml` for nano. See [Model configs](#model-configs-custom-not-upstream).

`--weights` is only the **starting checkpoint for transfer learning**, and does *not* have to
match the config's size: ultralytics loads whatever layers line up and initialises the rest.
In the committed invocations most runs start from `filtered_v3_margin_15` regardless of the
target size — including nano runs, which start from a tiny checkpoint 8 times out of 10.

Real invocations are kept in `python run_training_pipeline_example.sh` (the filename has a
stray `python ` prefix — it is a reference file, not executable).

### 2. Many experiments at once — `run_training_experiments_batch.py`

Reads experiment definitions from JSON (`experiments.json`, 12 entries;
`experiments_minibatch*.json` for smaller sweeps), assigns **one long-lived worker per GPU**,
and runs each experiment through `run_training_pipeline.py` with `--device` set automatically.

```bash
python run_training_experiments_batch.py --experiments-json experiments.json --gpus 0,1,2
python check_training_batch_status.py            # classify each run from its log
```

### 3. Evaluate — `extended_evaluation_metrics/run_eval_and_viz.sh`

Driven by a single `.env` file. In order:

1. `evaluation.py` — Ultralytics validation, single run or a **conf × iou sweep**. Writes
   `predictions.json`, `metrics_per_img.csv`, `metrics_per_gt.csv`, `metrics_per_pred.csv`,
   `global_metrics.json`, `sweep_results.csv`, `best_params.json`.
2. `visualize_detections.py` — per-frame overlays with TP/FP/FN colouring.
3. Executes `extended_report.ipynb` via nbconvert and exports HTML.
4. Optionally `generate_conclusions_claude.py` writes the conclusions section, then
   re-executes for the final HTML.

`EVAL_BACKEND` selects `ultralytics_pt` (`weights/best.pt`) or the TFLite backends, so **the
same harness scores the float model and the quantised one**. Full instructions:
`extended_evaluation_metrics/README_evaluate_new_weights.md`.

Other evaluators:

| Tool | Use |
|---|---|
| `eval_obj_det.py`, `ultralytics_validator_wrapper.py` | Direct PyTorch-vs-TFLite comparison on the same images |
| `run_batch_evaluation_v2.py` | Runs the extended pipeline across a whole batch of dataset_v2 experiments |
| `../pg1_yolo_evaluator/` | Same weights against PG1 footage (sibling project) |

### 4. Ship to the firmware — `cp_weights.py`

Copies the six release artefacts out of an experiment into the Alif firmware tree and records
provenance:

| From `results/<exp>/weights/` | Copied as |
|---|---|
| `<folder>/best_full_integer_quant.tflite` | `backbone.tflite` |
| `<folder>/vela_he/best_full_integer_quant_vela.tflite` | `backbone_H128.tflite` |
| `<folder>/vela_he/..._summary_Ethos_U55_High_End_Embedded.csv` | `backbone_summary.csv` |
| `head_saved_model/head_float32.tflite` | `head.tflite` |
| `head_saved_model/vela_he/head_float32_vela.tflite` | `head_H128.tflite` |
| `head_saved_model/vela_he/head_float32_summary_...csv` | `head_summary.csv` |

```bash
DST=../../rtos_sensor/projects/alif_ml-embedded-evaluation-kit_pg/resources_downloaded

python cp_weights.py \
    --src results/<exp_name> \
    --dst "$DST/pg_object_detection/int8/yolov8-nano/"
```

It also writes **`source_paths.txt`** into the destination, recording which experiment each
file came from — that file is how you trace a deployed model back to its training run. Use
`--ignore-head` to ship the backbone only.

---

## Scripts reference

| Script | Role |
|---|---|
| `run_training_pipeline.py` | **Main entry point** — train → export → Vela, both head variants |
| `run_training_experiments_batch.py` | Parallel experiment scheduler, one worker per GPU |
| `check_training_batch_status.py` | Classify batch-run status from logs |
| `train_yolo.py` | Ultralytics training wrapper |
| `export.py` | int8 backbone + fp32 head, **separated** |
| `export_non_separated.py` | Combined-head export variant |
| `export_preprocessing.py` | Exports the input adapter |
| `optimize_vela.py` / `optimize_vela_raw.py` / `optimize_vela.sh` | Ethos-U55 compilation |
| `generate_quantized_weights.py` | Search quantisation settings for a target size in MB |
| `merger.py` | Merge detector and head ONNX graphs by tensor shape |
| `detection_head.py`, `test_heads.py` | The separated head; DFL-equivalence tests |
| `tflite_inference.py`, `tflite_inference_input_adapter.py` | `TFLiteInferer` / `TFLiteMerged` runtime wrappers |
| `cp_weights.py` | **Ship artefacts to the firmware repo** |
| `prebuild_ultralytics_dataset_cache.py` | Prebuild and lock dataset caches read-only |
| `reduce_quality.py`, `pixelize.py`, `noising.py` | Image-degradation studies (blur, contrast, gamma, pixelisation, gaussian / salt-pepper / uniform noise) |
| `eval_obj_det.py`, `ultralytics_validator_wrapper.py`, `run_batch_evaluation_v2.py` | Evaluation |
| `install_dg_yolo.sh` | Install the DeGirum ultralytics fork (required for int8 export) |

## Layout

| Path | What |
|---|---|
| `cam4_revB/` | Dataset pipeline: CVAT→YOLO, AWB/CLAHE, anti-leakage splitting, batch injection |
| `extended_evaluation_metrics/` | Evaluation harness, holdout sets, HTML reports |
| `investigations/` | `release_june_26` (release regression), `rotated_cam_color_dfiff` (colour/CCM), `minibatch12` |
| `quality_reduce_comparison/`, `softmax_exp/`, `testset_review/` | Focused experiments |
| `results/` | Training runs, weights and Vela output (LFS-backed) |
| `experiments*.json` | Experiment definitions for the batch scheduler |

## Connections

**Downstream — where this project's output goes:**
- **`rtos_sensor`** (15 references) — `cp_weights.py` delivers the `.tflite` pair into
  `projects/alif_ml-embedded-evaluation-kit_pg/resources_downloaded/pg_object_detection/`.
  Same layout exists in `ALIF_BAT` and `alif_ml-embedded-evaluation-kit_pg`.
- **`PG2_GW`** (10 references) — the gateway consuming detections.
- **`algo_system_emulator`** (4 references) — offline pipeline replaying the same model.

**Within this repo:**
- `annotation_poc/` — annotation QA upstream of `cam4_revB/` dataset builds.
- `camera_calibration/` — lens parameters used for px→cm.
- `bbox_filter/` — filters this detector's boxes; ships in `PG2_GW`.
- `pg1_yolo_evaluator/` — defaults to weights from this project's `results/`.
- `camera_qc/` — ported from `investigations/rotated_cam_color_dfiff/sensor_metrics.py`.
- Imports `../../data-exploration/video_splitting`.

**Documentation:** experiment reports belong in `PG2.wiki` under
`PG2/Algo/Experiments-&-Reports/`.

## Dependencies

`requirements.txt` is present. Core: `numpy`, `opencv-python`, `torch`, `ultralytics`,
`pillow`, `tqdm`, `pandas`. Export adds `tensorflow`, `onnx`, `onnx2tf`, `onnxslim` and the
Vela compiler.

```bash
pip install -r requirements.txt
bash install_dg_yolo.sh      # DeGirum ultralytics fork — required for int8 export
```

Several exotic imports (`ncnn`, `tensorrt`, `openvino`, `coremltools`, `x2paddle`) come from
ultralytics' own export module and are **not** needed here.

---

## Original notes — how to train and quantise

The original YOLOv8 implementation has an issue with int8 quantization. So, the issues resolved in (the fork)[https://github.com/DeGirum/ultralytics_yolov8]. The developers of the fork just remove the detection head from the network.

### How to train YOLOv8 for microcontroller

1. Train YOLOv8 using regural `ultralytics`:
```bash
python train_yolo.py
```
2. Quantize the model:
    1. Install `DeGirum/ultralytics_yolov8`:
    ```bash
    bash install_dg_yolo.sh 
    ```

    2. Export model to int8:
    ```bash
    python export.py ...
    ```

    3. Check the model using `test_tflite_model.ipynb`.

2. Optimaze model for Alif:
    1. Install vela compiler:
    ```bash
    pip install -r requrements.txt
    ```

    2. Optimize the model:
    ```bash
    vela best_full_integer_quant.tflite \
        --output-dir vela_he \
        --accelerator-config ethos-u55-128 \
        --optimise Performance \
        --config vela.ini \
        --system-config Ethos_U55_High_End_Embedded \
        --memory-mode Shared_Sram
    ```

    3. Optimze the head:
    ```bash
    vela head_float32.tflite \
        --output-dir vela_he \
        --accelerator-config ethos-u55-128 \
        --optimise Performance \
        --config vela.ini \
        --system-config Ethos_U55_High_End_Embedded \
        --memory-mode Shared_Sram
    ```