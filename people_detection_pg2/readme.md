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

## Running on the training machine

All training, export and evaluation here runs on **`pgadmin-MS-7885`** — the shared training
box. Its hostname is embedded in the committed TensorBoard files
(`events.out.tfevents.*.pgadmin-MS-7885.*`), so past runs can be traced to it.

| | |
|---|---|
| GPUs | **3 × NVIDIA GeForce GTX 980 Ti**, 6 GB VRAM each |
| Driver / CUDA | 535.309.01 / CUDA 11.5 |
| CPU / RAM | 12 cores / 31 GB |

### Use the `yolo-export` pyenv environment — always

**Every YOLO-related task must run in the `yolo-export` environment.** This is not a
convenience: the environment has ultralytics installed **editable from the local DeGirum
fork**, not from PyPI —

```
ultralytics  8.3.1  ->  people_detection_pg2/ultralytics_yolov8
```

Any other environment gets upstream ultralytics, whose detection head **does not quantise to
int8 correctly**. Training may appear to work and the exported model will be wrong.

```bash
pyenv shell yolo-export           # or: export PYENV_VERSION=yolo-export
python -c "import ultralytics, torch; print(ultralytics.__file__); print(torch.cuda.device_count())"
# expect a path under people_detection_pg2/ultralytics_yolov8, and 3
```

Contents: Python 3.10.12, torch 2.4.1 + torchvision 0.19.1, tensorflow 2.17.0, onnx 1.17.0,
onnx2tf 1.22.3, onnxruntime 1.19.2, onnxslim 0.1.34, numpy 1.26.4, nbstripout 0.9.1.

The environment is selected by `PYENV_VERSION`; there is **no `.python-version` file** in the
repo, so nothing selects it for you. If you open a fresh shell and forget, you get the system
Python.

Other envs on this machine — `pgpytrain`, `pgpytrain-qt5`, `yolo-latest` — belong to other
projects. `yolo-latest` is the dangerous one to confuse: it holds upstream **ultralytics
8.3.203 + torch 2.8.0** from PyPI, exists for `yolo_latest_test/`, and will silently give
you the unquantisable head. Do not use it here.

### Using the three GPUs

One run, one GPU:

```bash
python run_training_pipeline.py --device 0 ...
```

All three in parallel — one long-lived worker per GPU, experiments pulled from a queue:

```bash
python run_training_experiments_batch.py --experiments-json experiments.json --gpus 0,1,2
python check_training_batch_status.py
```

> **6 GB of VRAM per card is the binding constraint.** These are 2015-era cards; the pipeline
> defaults to `--batch 32`, and `experiments.json` does not override it. A larger model, a
> larger `imgsz`, or another process on the same GPU will OOM. Check with `nvidia-smi` before
> starting a batch — the scheduler does not.

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

Validation is two steps: **(1) select thresholds**, then **(2) run `run_eval_and_viz.sh`**,
which is the **main validation script** — every model-evaluation number in `PG2.wiki`
(leaderboards, slice reports, baseline-set tables) comes from it.

#### 3.1 Select the inference thresholds (manual step)

Open
`cam4_revB/validate_model.ipynb`, set `RESULTS_PTH` to the experiment, sweep the confidence
threshold, pick `TUNED_CONF_TH` and `TUNED_IOU_TH` by hand, and run the cell that writes them
to `results/<exp_name>/inference_params.json`:

```json
{"conf_th": 0.25, "iou_th": 0.5}
```

`evaluation.py` reads this file only when `ENABLE_SWEEP=0` and `USE_INFERENCE_PARAMS=1`; in
sweep mode it is ignored. The file is not copied by `cp_weights.py` — the firmware uses its
own thresholds (`DetectorPostProcessing.hpp`: conf 0.2, NMS IoU 0.45).

> **To be improved.** The notebook currently tunes on the `batch1_3` **holdout** (the same
> frames later reported as a test set), on the **fp32** `best.pt` rather than the shipped
> int8 model, with `BASE_PTH` hard-coded rather than taken from the experiment's dataset. The
> IoU sweep is commented out, and the conf choice relies on validator P/R, which Ultralytics
> reports at the max-F1 point of the curve rather than at the passed `conf`. Thresholds
> should instead be selected on the experiment's `val` split, on the int8 model.

#### 3.2 Validate — `extended_evaluation_metrics/run_eval_and_viz.sh` (main validation script)

Use this for every model you evaluate or compare, fp32 or int8. Driven by a single `.env`
file. In order:

1. `evaluation.py` — Ultralytics validation, single run or a **conf × iou sweep**. Writes
   `predictions.json`, `metrics_per_img.csv`, `metrics_per_gt.csv`, `metrics_per_pred.csv`,
   `global_metrics.json`, `sweep_results.csv`, `best_params.json`.
2. `visualize_detections.py` — per-frame overlays with TP/FP/FN colouring.
3. Executes `extended_report.ipynb` via nbconvert and exports HTML (slice-based report:
   posture, occlusion, camera height, radial position, scene density).
4. Optionally `generate_conclusions_claude.py` writes the conclusions section, then
   re-executes for the final HTML.

`EVAL_BACKEND` selects `ultralytics_pt` (`weights/best.pt`) or the TFLite backends, so **the
same harness scores the float model and the quantised one**. For the shipped int8 model use
`tflite_backbone_only` (`weights/best_non_separated_saved_model/best_full_integer_quant.tflite`
— the file `rtos_sensor` deploys).

Two threshold modes — pick per purpose:

| Mode | `.env` | Use it for |
|---|---|---|
| **Sweep** (wiki protocol) | `ENABLE_SWEEP=1`, `SWEEP_CONF_THS=0.20,0.25,0.30`, `SWEEP_IOU_THS=0.35,0.40,0.45`, `SELECTION_METRIC=f1` | Numbers comparable with the existing wiki tables. `inference_params.json` is ignored; the pair is chosen on the evaluated set itself, so absolute numbers are optimistic |
| **Single run** | `ENABLE_SWEEP=0`, `USE_INFERENCE_PARAMS=1` | Evaluating at the thresholds selected in 3.1 (or set `CONF_TH=0.20`, `IOU_TH=0.45` to match the firmware) |

Run one experiment on one test set:

```bash
pyenv shell yolo-export
./extended_evaluation_metrics/run_eval_and_viz.sh extended_evaluation_metrics/.env.<my_experiment>
```

Run the int8 model of one or more experiments on the four standard test sets (`batch1_3`
holdout, `testAccuracy`, `H270_16-44-09`, `PG2HardCases`) in sweep mode, with results written
into each experiment folder:

```bash
pyenv shell yolo-export
cd extended_evaluation_metrics
nohup ./run_eval_int8_experiments.sh <exp_name> [<exp_name> ...] > ../logs/eval_int8.log 2>&1 &
# results: results/<exp_name>/eval_int8/<testset>/{best_params.json,global_metrics.json,extended_report.html,...}
```

`results/<exp>/eval_int8/` is ~150 MB per experiment (overlays, per-candidate runs) and sits
under the LFS-tracked `results/` tree — do not `git add` it; copy the numbers into the wiki.

To compare against production, use the prod runs already in
`extended_evaluation_metrics/val_baseline_<testset>/` (same protocol). Full instructions:
`extended_evaluation_metrics/README_evaluate_new_weights.md`.

Other evaluators:

| Tool | Use |
|---|---|
| `extended_evaluation_metrics/run_eval_int8_experiments.sh` | Batch wrapper over `run_eval_and_viz.sh`: int8, 4 test sets, results into `results/<exp>/eval_int8/` |
| `run_batch_evaluation_v2.py` | Runs the extended pipeline across a whole batch of dataset_v2 experiments (envs generated from a template) |
| `eval_obj_det.py`, `ultralytics_validator_wrapper.py` | Direct PyTorch-vs-TFLite comparison on the same images |
| `../pg1_yolo_evaluator/` | Same weights against PG1 footage (sibling project) |
| `../../algo_system_emulator` | Release evaluation through the sensor + gateway emulation (required by the Model Evaluation Protocol) |

> **Do not use for wiki numbers:** `cam4_revB/eval_float32_vs_int8.ipynb` scores the
> separated int8 backbone + fp32 head, not the shipped fully-int8 model, and
> `cam4_revB/eval_using_obj_det_metrics_per_frame.ipynb` is fp32-only and skips frames with
> no predictions (inflating recall). Both predate `run_eval_and_viz.sh`.

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
| `patch/` | Local patches to the DeGirum fork, applied by `install_dg_yolo.sh` — see [Fork patches](#fork-patches) |
| `ultralytics_yolov8/` | The patched fork itself (git-ignored; rebuilt by `install_dg_yolo.sh`) |

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

**On `pgadmin-MS-7885`, use the `yolo-export` environment — it already has all of this.** See
[Running on the training machine](#running-on-the-training-machine). The rest of this section
is for building the environment elsewhere.

`requirements.txt` is present. Core: `numpy`, `opencv-python`, `torch`, `ultralytics`,
`pillow`, `tqdm`, `pandas`. Export adds `tensorflow`, `onnx`, `onnx2tf`, `onnxslim` and the
Vela compiler.

```bash
pip install -r requirements.txt
bash install_dg_yolo.sh      # DeGirum ultralytics fork — required for int8 export
```

`install_dg_yolo.sh` is what makes ultralytics resolve to the local fork rather than PyPI. An
environment without it will train and export without complaint and produce a model whose head
is wrong under int8.

### Fork patches

The fork is **not used as published**. `install_dg_yolo.sh` checks out a pinned commit and then
overwrites five files with the local copies in `patch/`:

| `patch/` file | Replaces in `ultralytics_yolov8/` | Change |
|---|---|---|
| `exporter.py` | `ultralytics/engine/exporter.py` | Caps the int8 calibration set at `max_ncalib_imgs` (passed by `export.py`, default 100) instead of using the whole dataloader |
| `autobackend.py` | `ultralytics/nn/autobackend.py` | TFLite: reads `metadata.yaml` next to the weights instead of from the model zip |
| `predictor.py` | `ultralytics/engine/predictor.py` | Stops passing `end2end` to NMS |
| `predict.py` | `ultralytics/models/yolo/pose/predict.py` | Keypoints are 2 values (x, y), not 3 (x, y, visibility) |
| `plotting.py` | `ultralytics/utils/plotting.py` | Adds dark/light colour sets for drawing |

Pinned commit: `75cab2e0c68723d4344c69a3bcd85265a582ab3d` (DeGirum `ultralytics_yolov8`,
2024-09-30). The patches are **whole-file copies, not diffs**, so they only fit that commit.

- **Don't clone the fork by hand** — you get the unpatched code and nothing will complain.
- **Don't bump the commit** without re-applying each change onto the new versions of these
  files.
- `ultralytics_yolov8/` is git-ignored here but is its own git checkout, so check the patches
  from inside it:

```bash
cd ultralytics_yolov8
git rev-parse HEAD            # expect 75cab2e0...
git status --short            # expect exactly the five files above, all ` M`
cmp ../patch/exporter.py ultralytics/engine/exporter.py   # and likewise for the other four
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