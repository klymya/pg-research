# researches
/home/klymy/projects/PG2.wiki/PG2/Algo/Reports/Camera-Image-Quality-Evaluation-Tool.md
> **Repository status:** Active: a monorepo whose projects are individually ACTIVE, ARCHIVED or VENDORED (see [Current Status](#current-status))<br>
> **Primary owner/team:** Algo team<br>
> **Technical contact:** Algo team<br>
> **Last verified:** 2026-09-28<br>
> **Documentation owner:** Algo team

## 1. Overview

### Purpose

Research code and results for PointGrab's people- and object-detection work: sensor
algorithms, dataset pipelines, model training and quantisation, gateway prototypes and
hardware evaluations.

This is a **monorepo of independent research projects**, not one application. Each top-level
folder is its own project with its own dependencies; there is no shared build, no shared
virtualenv and no top-level entry point. Projects range from live production-supporting work
to investigations frozen years ago — check the status tag before assuming anything runs.

### Scope

This repository is responsible for:

- Training, int8 quantisation and Vela compilation of the PG2 people detector
  (`people_detection_pg2/`), and handing the resulting model to the firmware repo.
- Research prototypes of sensor-side algorithms (e.g. `background_subtraction/` "Sign of Life")
  and the reference implementations used for their MCU ports.
- Dataset pipelines, annotation QA, camera calibration and camera-module QC for the PG2 sensor.
- Evaluation harnesses, benchmarks and the record of finished investigations (ARCHIVED projects).

This repository is **not** responsible for:

- Firmware that ships on the sensor — `rtos_sensor`.
- The shipping gateway — `PG2_GW`. Gateway code here (`gw_prototype`, `gw_cpp_version`) is superseded.
- The offline system emulator — `algo_system_emulator`.
- Datasets and training results — they live on the fileserver (`/mnt/pg-fs01`, `/mnt/pg-fs02`).
- Conclusions and experiment reports — they belong in `PG2.wiki`, not in notebook output.

### Current Status

- Production status: **development / research.** Nothing is deployed from this repo; the
  detector trained in `people_detection_pg2/` ships inside `rtos_sensor`.
- Current release/version: none — the repo has no tags and no versioning.
- Main branch: `master`.
- Supported platforms/environments: Ubuntu Linux on the training machine `pgadmin-MS-7885`
  (see [Prerequisites](#5-prerequisites)). `footfall_report/` also targets a Windows end user.
- Major work currently in progress: `people_detection_pg2/cam4_revB` dataset builds and
  training experiments; the `background_subtraction` Sign-of-Life C port.
- Known replacement/migration plan: none. `REPO_INVENTORY.md` §7 proposes moving
  `people_detection_pg2` results out to the fileserver.

Every project carries one of three status tags:

| Tag | Meaning |
|---|---|
| **ACTIVE** | In development. |
| **ARCHIVED** | Frozen record of a finished investigation. Do not refactor, reformat or strip its notebook outputs; if it looks broken, it probably is. |
| **VENDORED** | Upstream third-party code kept in-tree. |

---

## 2. System Context

This repo sits upstream of the sensor firmware and the gateway: it produces models, parameter
files and reference implementations that other repos ship.

### Upstream Dependencies

| Dependency | Purpose | Interface | Required? | Documentation |
|---|---|---|---|---|
| Fileserver `/mnt/pg-fs01`, `/mnt/pg-fs02` | Datasets, raw captures, training results | Mounted filesystem; paths hard-coded in notebooks | Yes, for almost all data work | Paths in each project's notebooks |
| CVAT | Annotations for `cam4_revB` and earlier datasets | Exported annotation files | Yes, for dataset builds | `annotation_poc/README.md`, `people_detection_pg2/readme.md` |
| DeGirum ultralytics fork | Training/export with an int8-quantisable head | Editable Python install from `people_detection_pg2/ultralytics_yolov8` | Yes, for every YOLO task | `people_detection_pg2/readme.md` |
| `PG_PyTraining` | Shared PG1-era utilities | Python import from notebooks | Only for ARCHIVED `vayyar`, `coordinates_projection`, `kpmg_research`, `object-detection` | — |
| `alif_ensemble_data_collection` firmware | Captures the TIFF frames used for datasets and calibration | USB/Ethernet/SD transfer + host `client.py` | For new sensor datasets | `alif_ensemble_data_collection/CAMERA_TIFF_TRANSFER.md` |

### Downstream Consumers

| Consumer | What It Uses | Interface | Documentation |
|---|---|---|---|
| `rtos_sensor` (Alif firmware) | int8 backbone + fp32 head `.tflite`, Vela-compiled; Sign-of-Life C reference | `people_detection_pg2/cp_weights.py` copies six artefacts and writes `source_paths.txt`; `background_subtraction/sign_of_life_cpp/` | `people_detection_pg2/readme.md`, `background_subtraction/readme.md` |
| `PG2_GW` | Camera geometry; `bbox_filter` novelty parameters | Ported source (`PG2_GW/src`); JSON parameter dumps (`PG2_GW/gen_novelty_params`) | `bbox_filter/README.md`, `pg_camera_geometry/README.md` |
| `algo_system_emulator` | `bbox_filter`, `objectdetectionmetrics`, geometry from `gw_prototype` | Copied source | `bbox_filter/README.md`, `objectdetectionmetrics/README.md` |
| All geometry consumers | Fisheye lens parameters | `camera_calibration/params/*.json` | `camera_calibration/README.md` |
| `PG2.wiki` | Experiment and evaluation reports | Written up by hand | `PG2/Algo/Experiments-&-Reports/` |

### External / Third-Party Dependencies

| Dependency | Purpose | Version / Constraint | Account / Ownership Notes |
|---|---|---|---|
| Git LFS | ~40,000 of ~45,000 tracked files are LFS pointers | 3.x (3.5.1 on the training machine) | LFS storage on Azure DevOps, project `PG` |
| nbstripout | Strips outputs from notebooks in ACTIVE projects | 0.9.1 in `yolo-export` | Per-clone filter; not shared through Git |
| ultralytics (DeGirum fork) | YOLO training and export | 8.3.1, **pinned** — see [Technology Stack](#4-technology-stack) | Vendored under `people_detection_pg2/ultralytics_yolov8` |
| PyTorch | Training | 2.4.1 (+cu121) in `yolo-export` | — |
| Arm Vela (`ethos-u-vela`) | Compiles `.tflite` for Ethos-U55 | 4.0.0 in `yolo-export` | — |
| Azure DevOps | Git hosting | — | `dev.azure.com/pointgrab/PG`; access via the Algo team |

### Architecture

Cross-repo connection map:
```
                        ┌──────────────────────────────────────────┐
                        │  PG2.wiki — docs, protocols, reports     │
                        └───────────────▲──────────────────────────┘
                                        │ reports & protocols
   researches/people_detection_pg2 ─────┼─── .tflite (int8 + Vela) ──▶ rtos_sensor
        │  training / quantisation      │                             (Alif firmware)
        │                               │                                    ▲
        ├──▶ pg1_yolo_evaluator         │                                    │ frames
        │                               │        researches/alif_ensemble_data_collection
        ├──▶ algo_system_emulator ◀─────┼──── researches/bbox_filter
        │    (offline TFLite pipeline)  │  ◀── researches/objectdetectionmetrics (2nd copy)
        │                               │  ◀── researches/gw_prototype (geometry, AOI)
        │                               │
   researches/camera_calibration ───────┴── lens params ──▶ all of the above
        │
        └──▶ researches/camera_qc

   researches/gw_prototype ──▶ researches/gw_cpp_version ──▶ PG2_GW (shipping C gateway)
   researches/pg_camera_geometry ───────────────────────────▶ PG2_GW/src
   researches/bbox_filter ──────────────────────────────────▶ PG2_GW/gen_novelty_params
   researches/alif_beta ◀── pg2gw journal logs ───────────── PG2_GW
   researches/background_subtraction ── SoL C port ─────────▶ rtos_sensor

   Also imported by notebooks: PG_PyTraining (vayyar, coordinates_projection,
   kpmg_research, object-detection)
```

**Known duplication:** camera geometry exists in four places; `bbox_filter` in three;
`objectdetectionmetrics` in two; calibration captures are byte-duplicated between
`camera_calibration/data/` and `gw_prototype/camera_calibration/`.

Architecture diagrams: none beyond the map above. Per-project architecture lives in each
project's readme. The most detailed is `background_subtraction/`, which has 11 markdown files.

---

## 3. Repository Structure

```text
.
├── <36 project folders>/   # Independent projects, each with its own README/readme.md
├── README.md               # This file
├── CLAUDE.md               # Conventions for AI agents (and a concise rule list for humans)
├── REPO_INVENTORY.md       # Inventory, Git/LFS audit and remediation log
├── .gitattributes          # LFS routing and nbstripout scope
└── .gitignore
```

`build/`, `.pytest_cache/` and `.vscode/` at the root are local, untracked leftovers.

Important files:

| Path | Purpose |
|---|---|
| `.gitattributes` | Decides what goes to LFS and which notebooks nbstripout strips. Check any change with `git check-attr`. |
| `CLAUDE.md` | The rules that must not be broken — environments, what never to commit, archived projects. |
| `REPO_INVENTORY.md` | Per-project audit: file counts, sizes, defects, remediation history. |
| `<project>/README.md` or `readme.md` | Purpose, inputs, outputs, status and how to run that project. |

### Layout

36 projects in six clusters:

| Cluster | What it is | Projects |
|---|---|---:|
| **A — PG2 sensor algorithm** | The live core: detector training, quantisation, QC | 7 |
| **B — PG2 gateway & system** | Gateway prototypes, geometry, storage | 4 |
| **C — Detection & evaluation research** | Datasets, models, benchmarks | 12 |
| **D — Hardware evaluations & MCU tooling** | Sensor/SoC investigations; 2 still active | 7 |
| **E — One-off analyses & tools** | Reports, customer checks, utilities | 4 |
| **F — Vendored / third-party** | Upstream code kept in-tree | 2 |

Each entry below shows: status · month of last change · size on disk.

### Cluster A — PG2 sensor algorithm

#### `people_detection_pg2/` — ACTIVE · 2026-09 · 12 GB
The main project. End-to-end YOLOv8/YOLO11 people detection for the Alif Ensemble MCU:
training, int8 quantisation via the `DeGirum/ultralytics_yolov8` fork (upstream's head breaks
int8), and Vela compilation for Ethos-U55. Sub-trees are projects in their own right —
`cam4_revB/` (dataset pipeline, CVAT→YOLO, anti-leakage splitting), `extended_evaluation_metrics/`
(holdout evaluation and reports), `investigations/` (release regressions, colour/sharpness
studies). Produces the `.tflite` that ships in `rtos_sensor`.

#### `background_subtraction/` — ACTIVE · 2026-09 · 6.1 GB
"Sign of Life" detection: a 4-layer non-NN background-subtraction pipeline for the Alif MCU
that isolates newly introduced *static* objects while rejecting shadows and lighting drift.
Includes pure-NumPy ports of the OpenCV operations (bit-exact) as the C porting target, a
`sign_of_life_cpp/` reference implementation and ~350 tests. The best-documented project here —
11 markdown files covering architecture, rationale and quantisation roadmap. Only `data/` (the
six original CDNET-14 sequences) is committed; reduced-rate trees are regenerated on demand.

#### `camera_qc/` — ACTIVE · 2026-09 · 995 MB
Camera-module incoming QC. Compares a device-under-test camera against a reference and emits a
one-page PDF verdict (PASS/MARGINAL/FAIL); sharpness is the only blocking metric, colour and
noise are advisory. A proper installable package with a real CLI (its `tests/` folder is empty) — closer to a
delivered tool than research. Capture batches stay out of Git.

#### `camera_calibration/` — ACTIVE · 2026-09 · 1.3 GB
Fisheye intrinsic calibration for the cam4.2 rev-B sensor. `calibration_revB.ipynb` runs
`cv2.fisheye.calibrate` over TIFF captures and writes the lens parameters every downstream
projection depends on. `alfa/` holds pixel-density analysis and detection-area drawing. The
small JSONs in `params/` are the valuable output; the 1.3 GB of captures are re-shootable.

#### `annotation_poc/` — ACTIVE · 2026-05 · 78 MB
CVAT annotation QA: visualisation notebooks, batch/filename fixers, and validators that catch
malformed or misaligned annotations. `Pilot/` holds a comparison of an outsourced annotation
vendor against in-house labels. Upstream of the `cam4_revB` dataset builds.

#### `bbox_filter/` — ACTIVE · 2026-09 · 6.6 MB
Novelty filtering of detector bounding boxes on `[aspect_ratio, area, dist_from_center,
cam_height]`. Trains sklearn IsolationForest/GMM, then re-implements inference dependency-free
in Python and C++ (Eigen) reading JSON parameter dumps, so it can run on the gateway. Shipping
copies live in `PG2_GW` and `algo_system_emulator` — confirm which is authoritative before
changing anything.

#### `detection_area/` — ACTIVE · 2026-09 · 32 KB
Generates the A4 floor-marker sheet used for detection-area and coverage data collection: one
large circle per page, plus registration ticks and a 100 mm scale bar that catches
"fit to page" printer errors. Two files, self-contained.

---

### Cluster B — PG2 gateway & system

#### `gw_prototype/` — ARCHIVED · 2025-09 · 356 MB
Python prototype of the PG2 gateway: receives detections over MQTT (mosquitto + protobuf),
filters by detection area, converts px→cm via the fisheye model, stores to SQLite and
aggregates AoI and XY events. Ships sensor and cloud stubs plus a Raspberry Pi deployment path.
Superseded by `PG2_GW`, but its stubs and broker remain the fastest local test rig.

#### `gw_cpp_version/` — ARCHIVED · 2025-01 · 1.1 MB
The same gateway functionality in C++ (vcpkg + CMake + protobuf), reusing `gw_prototype`'s
broker and stubs. Superseded by `PG2_GW`. Its readme is a copy-paste of the Python one and is
still titled "Gateway Python Prototype".

#### `pg_camera_geometry/` — ARCHIVED · 2024-12 · 204 MB
C++ camera-geometry library (px↔cm projection, detection area) with tests. Note that geometry
now exists in four places — here, `gw_prototype`, `PG2_GW` and `PG_PyTraining`. There is no
single source of truth.

#### `gw_storage/` — ARCHIVED · 2024-11 · 68 KB
Throughput spike comparing Redis Streams / RedisTimeSeries against SQLite for storing XY
points. The result is two `.txt` files; the code has no reuse value.

---

### Cluster C — detection & evaluation research

#### `pg1_yolo_evaluator/` — ARCHIVED · 2025-04 · 5.4 GB
Batch evaluation of YOLO variants (v8n, v8-tiny, 11, 11n) on PG1 footage at full and reduced
quality. Only four driver scripts are committed; all 5.4 GB of results are gitignored — the
pattern the rest of the repo should follow.

#### `fps_influence_on_count_acuracy_research/` — ACTIVE · 2026-07 · 912 KB
Sweep of detection period against count period, evaluated with MAE / %-wrong / accuracy
matrices and an HTML report. Clean and self-contained; reuses the aggregation and evaluation
scripts rather than reimplementing them. (Folder name has a typo: "acuracy".)

#### `object-detection-v3/` — ARCHIVED · 2024-08 · 1.1 GB
Third iteration of furniture detection (tables, chairs, monitors). Holds the dataset database —
annotations, image IDs, splits, embeddings — plus the CVAT→YOLO pipeline and training results.
Final model and thresholds are documented in its README. Dataset itself lives on
`/mnt/pg-fs01/Current/DB/ObjectDetectionV3`.

#### `object-detection/` — ARCHIVED · 2024-05 · 942 MB
Volume 2 of the same furniture-detection line: YOLOv7 evaluation notebooks and weights.
Superseded by `object-detection-v3`.

#### `alif_beta/` — ACTIVE · 2026-09 · 22 MB
Beta-deployment analysis: parses `pg2gw` journal logs into structured detections and plots
per-device behaviour. `smoothing/` compares count smoothers (SMA/WMA/EMA/Gaussian/Median/
Savitzky-Golay) and recommends **Median → EMA** as the production default.

#### `leg_position_detectors/` — ARCHIVED · 2024-04 · 377 MB
Neural leg-position estimation: YOLOv8 pose models at several sizes and input resolutions, int8
TFLite export with a split detection head (the same trick later reused in
`people_detection_pg2`), and a TFLite-Micro example.

#### `leg_position_calculator/` — ARCHIVED · 2023-11 · 89 MB
The non-NN alternative: linear regression from bbox coordinates and aspect ratio to leg
position. One notebook and its training data.

#### `data-exploration/` — ARCHIVED · 2024-10 · 354 MB
PG1-era dataset archaeology: crawl `.seq`/`.avi` paths on the fileserver, deduplicate, parse
annotations into a SQLite database, then build COCO/YOLOX datasets. The README documents the
eight-step pipeline.

#### `theodore/` — ARCHIVED · 2024-06 · 289 MB
Work on the TU-Chemnitz THEODORE fisheye dataset: TFRecord extraction, YOLO and U-Net dataset
preparation, ONNX and C++ inference pipelines. Superseded by in-house cam4 data.

#### `tracker/` — ARCHIVED · 2024-05 · 31 MB
ByteTrack evaluation plus `alif/` — a C++ ByteTrack port for the MCU. Most of the file count is
vendored Eigen and a vendored ByteTrack checkout; the PG work is about 30 files.

#### `yolo-inference/` — ARCHIVED · 2024-05 · 216 MB
C++ YOLOv5/v7 inference benchmarks on PG1 hardware (Raspberry Pi 3/4): OpenCV-DNN+ONNX vs
TFLite fp16 vs int8, across thread counts. The benchmark tables in its README are the
deliverable.

#### `yolo_latest_test/` — ARCHIVED · 2025-09 · 156 KB
Smoke test that a newer ultralytics/YOLO11 trains and exports. Throwaway.

---

### Cluster D — hardware evaluations & MCU tooling

#### `alif_ensemble_data_collection/` — ACTIVE · 2026-09 · 478 MB
Alif Ensemble firmware (CMSIS + Azure RTOS) for capturing TIFF frames off the sensor and
shipping them over USB/Ethernet/SD, plus the host-side client. Forked from an Alif template, so
its README is still the upstream one. The live data-collection tool for new sensor datasets.

#### `alif_perf/` — ACTIVE · 2026-09 · 140 KB
One notebook parsing Ethos-U per-stage `CPU TOTAL: <cycles>` logs into per-run seconds. Small
but the only tool for reading MCU inference timings.

#### `gap9/` — ARCHIVED · 2023-12 · 91 MB
GreenWaves GAP9 MCU evaluation: nntool quantisation of YOLOv8/YOLOX, SDK fixes, eval notebooks.
The hardware was **not** selected (Alif won). Its README is two lines and records no outcome —
the notebook outputs are the only surviving evidence of why, which is why they are deliberately
exempt from nbstripout.

#### `alif-microphone-test/` — ARCHIVED · 2024-07 · 3.0 MB
Python prototype of a microphone production test using 5/10/15 kHz synthetic tones, with
iPhone-recorded reference samples.

#### `vayyar/` — ARCHIVED · 2023-11 · 111 MB
Vayyar 60 GHz radar evaluation: capture scripts and projection of radar points onto PG video.
Depends on `PG_PyTraining`.

#### `Socionext/` — ARCHIVED · 2023-05 · 24 KB
Socionext 60 GHz radar: FMCW chirp parameter tuning (range, bandwidth, resolution) and a C
sample app.

#### `orbbec-3d-research/` — ARCHIVED · 2023-05 · 48 KB
Orbbec Persee 3D camera: simultaneous RGB and depth capture via OpenNI2 and OpenCV.

---

### Cluster E — one-off analyses & tools

#### `footfall_report/` — ACTIVE · 2026-09 · 116 KB
Streamlit app turning a weekly hourly IN/OUT CSV into a validated daily footfall report, with
an explicit data-quality section — nothing invalid is silently dropped. Has tests, sample data
and a Windows launcher; built for a non-technical end user.

#### `coordinates_projection/` — ARCHIVED · 2024-08 · 299 MB
Undistortion and ground-plane projection demos, plus a study of how sensor tilt translates into
position error. Depends on `PG_PyTraining`.

#### `kpmg_research/` — ARCHIVED · 2026-03 · 4.1 MB
Customer-site geometry sanity check: overlays the computed detection-area polygon on a site
sketch and validates px→cm distances. Imports from both `PG_PyTraining` and `gw_prototype`.

#### `pipeline_report_parser/` — ARCHIVED · 2025-04 · 8 KB
Aggregates tracking-analysis CSVs across PG1 benchmark runs from `/mnt/pg-fs02`. Hard-coded run
IDs; PG1-era.

---

### Cluster F — vendored third-party

#### `objectdetectionmetrics/` — VENDORED · 8.2 MB
Padilla et al. *Object-Detection-Metrics* (MIT), plus the paper. Used across the detection
projects — and **copied again** into `algo_system_emulator/src/`. Two divergent copies.

#### `sagemaker/` — ARCHIVED · 2023-06 · 184 KB
AWS SageMaker tutorial follow-along (TensorFlow and PyTorch MNIST). No PointGrab content.

Also vendored in-tree: `tracker/alif/eigen-3.3.9`, `tracker/ByteTrack`,
`orbbec-3d-research/OpenNIDevEnvironment`, `nlohmann` copies under `gw_cpp_version` and
`theodore/pipeline_cpp`.

---

## 4. Technology Stack

There is no repo-wide stack; each project declares its own. The common ground:

| Area | Technology | Version |
|---|---|---|
| Language | Python | 3.10 for YOLO work (`yolo-export`: 3.10.12); 3.9+ elsewhere |
| Language | C / C++ | Per project — CMake, vcpkg (`gw_cpp_version`), CMSIS csolution (`alif_ensemble_data_collection`) |
| ML framework | PyTorch + ultralytics (DeGirum fork) | 2.4.1 + 8.3.1, **pinned** |
| Export toolchain | TensorFlow, onnx, onnx2tf, Arm Vela | 2.17.0, 1.17.0, 1.22.3, 4.0.0 |
| Notebooks | Jupyter | — |
| Package / env manager | pip + pyenv (pyenv-virtualenv) | — |
| Test framework | pytest | Per project |
| Runtime / OS | Ubuntu Linux, NVIDIA driver 535.309.01, CUDA 11.5 | Training machine |
| Storage | Git LFS | 3.x |

**Intentional pins:**

- **ultralytics 8.3.1 from the local DeGirum fork.** Upstream ultralytics' detection head does
  not quantise to int8 correctly. The fork splits the head from the network so the backbone
  exports int8 and the head fp32. Upgrading to upstream breaks the shipped model without any
  error.
- **Weights are tied to configs.** The `*.yaml` model configs in `people_detection_pg2/` are
  not edited once weights exist, because editing them would invalidate comparisons against
  trained runs.

---

## 5. Prerequisites

Required:

- **Git LFS**, installed and initialised (`git lfs install`) before cloning.
- **nbstripout**, installed into the clone before committing any notebook.
- **pyenv with the `yolo-export` environment** for any YOLO task (training, export,
  quantisation, evaluation in `people_detection_pg2`, `pg1_yolo_evaluator`,
  `leg_position_detectors`, `object-detection-v3`).
- A per-project Python environment for everything else. Each project owns its requirements.
- Access to: the Azure DevOps `PG` project, the fileserver mounts `/mnt/pg-fs01` and
  `/mnt/pg-fs02`, and for cross-repo work `rtos_sensor`, `PG2_GW`, `algo_system_emulator`
  and `PG2.wiki`.

Recommended development environment — the training machine **`pgadmin-MS-7885`**:

| | |
|---|---|
| GPUs | 3 × NVIDIA GeForce GTX 980 Ti, **6 GB VRAM each** |
| Driver / CUDA | 535.309.01 / CUDA 11.5 |
| CPU / RAM | 12 cores / 31 GB |
| Python envs | `yolo-export` (YOLO work), `yolo-latest` (only `yolo_latest_test/`), `pgpytrain`, `pgpytrain-qt5` |

IDE/editor: any. A notebook-capable editor (VS Code / JupyterLab) is assumed by most projects.

### Access Requirements

| System | Required Role / Permission | How Access Is Requested |
|---|---|---|
| Azure DevOps `pointgrab/PG`, repo `researches` | Read (clone); Contribute (push) | Ask the Algo team |
| Fileserver `/mnt/pg-fs01`, `/mnt/pg-fs02` | Read; write for new captures and results | Ask the Algo team |
| Training machine `pgadmin-MS-7885` | User account | Ask the Algo team |
| `rtos_sensor`, `PG2_GW`, `algo_system_emulator`, `PG2.wiki` | Read; Contribute to deliver models/reports | Ask the Algo team |

**Do not place passwords, private keys, API tokens, certificates, or other secrets in this
README.**

---

## 6. Getting Started

Both LFS and nbstripout are **per-clone**. Cloning does not enable them, and skipping either
corrupts what you commit.

### 6.1 Clone

```bash
# Git LFS — once per machine
sudo apt install git-lfs        # or: brew install git-lfs
git lfs install

# Full clone pulls ~15 GB of LFS content
git clone https://pointgrab@dev.azure.com/pointgrab/PG/_git/researches
cd researches
```

**Code-only clone (~0.4 GB instead of ~15 GB)**. Then fetch just the data you need:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <url>
cd researches
git lfs pull --include="background_subtraction/data/**"
```

The history was rewritten in September 2026, so a clone from before then must be re-cloned,
not pulled. See [Troubleshooting](#17-troubleshooting).

### 6.2 Select the Correct Toolchain

No `.python-version` file exists, so a fresh shell gets the system Python. For YOLO work:

```bash
pyenv shell yolo-export           # or: export PYENV_VERSION=yolo-export
python -c "import ultralytics; print(ultralytics.__file__)"
# must print a path under people_detection_pg2/ultralytics_yolov8
```

For other projects, use the environment described in that project's readme.

### 6.3 Install Dependencies

Per project. There is no repo-wide `requirements.txt`. To build the YOLO environment on a
machine other than `pgadmin-MS-7885`:

```bash
cd people_detection_pg2
pip install -r requirements.txt
bash install_dg_yolo.sh      # DeGirum ultralytics fork — required for int8 export
```

### 6.4 Configure Local Environment

Install nbstripout into the clone **before committing any notebook**:

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

No `.env` file is used.

### 6.5 Initialize Required Local Services

None at repo level. The fileserver must be mounted for data work. Individual projects may need
more — for example, ARCHIVED `gw_prototype/` runs mosquitto in Docker (see its readme).

### 6.6 Build

Nothing to build at repo level. See [Build](#8-build) for the C/C++ projects.

### 6.7 Run Locally

Per project. See the project's readme.

### 6.8 Verify the Setup

```bash
git lfs ls-files | wc -l                          # ~40,000 — not 0
git check-attr filter -- <notebook-in-active-project>   # expect: filter: nbstripout
nvidia-smi                                        # 3 GPUs visible on the training machine
```

Expected result: LFS files are real content, not 130-byte pointer files. Notebooks in ACTIVE
projects resolve to the `nbstripout` filter. Notebooks in ARCHIVED projects and
output-is-the-result notebooks resolve to `unset`, which is deliberate.

---

## 7. Configuration

### Configuration Sources

No repo-wide configuration. Configuration is per project:

1. Hard-coded paths and parameters in notebooks and scripts — including fileserver paths.
2. Project config files — e.g. model `*.yaml` configs and `experiments_*.json` batch
   definitions in `people_detection_pg2/`, `pyproject.toml`-declared CLI options in `camera_qc/`.
3. Command-line arguments of the project's scripts.

### Environment Variables

| Variable | Required | Default | Description | Secret? |
|---|---:|---|---|---:|
| `PYENV_VERSION` | Yes, for YOLO work | unset → system Python | Selects the pyenv environment; set it to `yolo-export` (or use `pyenv shell`) | No |
| `GIT_LFS_SKIP_SMUDGE` | No | unset → full LFS download | `1` for a code-only clone | No |

### Secrets

None are stored in or needed by this repo. Access to Azure DevOps and the fileserver uses each
engineer's own credentials. Access is requested through the Algo team.

---

## 8. Build

### Standard Build

No top-level build. Python projects are run directly. The C/C++ projects build independently.
Their exact commands are in their readmes:

| Project | Build system | Status |
|---|---|---|
| `alif_ensemble_data_collection/` | CMSIS csolution (`alif.csolution.yml`) | ACTIVE |
| `background_subtraction/sign_of_life_cpp/` | CMake | ACTIVE |
| `bbox_filter/` (C++ inference) | Header-only Eigen + nlohmann/json | ACTIVE |
| `pg_camera_geometry/` | CMake, with tests | ARCHIVED |
| `gw_cpp_version/` | vcpkg + CMake + protobuf | ARCHIVED |
| `tracker/alif/`, `theodore/pipeline_cpp`, `yolo-inference/` | CMake | ARCHIVED |

### Build Outputs

| Artifact | Location | Purpose |
|---|---|---|
| Trained weights, int8 backbone + fp32 head `.tflite`, Vela output | `people_detection_pg2/results/<run>/` | Delivered to `rtos_sensor` by `cp_weights.py` |
| Lens parameters | `camera_calibration/params/*.json` | Input to every px→cm projection |
| Novelty-filter parameters | JSON dumps from `bbox_filter/` | Consumed by `PG2_GW` |

### Generated Files

- `background_subtraction/data_*fps*/` are regenerated by `downsample_dataset.py` /
  `convert_to_bmp.py`. Never commit them.
- `gw_prototype/` protobuf stubs are generated by `libprotoc`.

---

## 9. Running and Development Workflow

### Development Mode

Most work is notebooks and scripts run in the project directory. For YOLO training:

```bash
pyenv shell yolo-export
nvidia-smi                        # 6 GB per card is the binding constraint; the scheduler does not check
cd people_detection_pg2
python run_training_pipeline.py ...                        # train → export → Vela, one run
python run_training_experiments_batch.py --gpus 0,1,2 ...  # one worker per GPU
```

Use `--device 0` for a single-GPU run. Full options are in `people_detection_pg2/readme.md`.

### Debugging

Standard Python/Jupyter debugging. The MCU projects (`alif_ensemble_data_collection/`) need
an Alif Ensemble board and debug probe. See that project's readme.

### Common Developer Workflow

```bash
git check-attr filter -- <notebook>   # before committing any notebook
git status                            # make sure no results/, runs/, derived datasets are staged
python -m pytest tests -q             # in projects that have tests
```

Do **not** edit by hand: derived datasets (`background_subtraction/data_*fps*/`), generated
protobuf stubs, or the vendored ultralytics fork except as a deliberate change to the
quantisation path.

---

## 10. Testing

### Test Types

No repo-wide test suite. Projects with automated tests:

| Project | Location | Command (from project dir) | Dependencies |
|---|---|---|---|
| `background_subtraction/` | `tests/` (~350 tests) | `python -m pytest tests -q` | `numpy`, `opencv-python`, `pytest` |
| `footfall_report/` | `tests/` | `python -m pytest tests -q` | `requirements.txt` |
| `bbox_filter/` | `test_*.py` | `python -m pytest -q` | `numpy`, `scikit-learn`, `pytest` |
| `pg_camera_geometry/` (ARCHIVED) | `test/` | CMake test target | C++ toolchain |
| `gw_prototype/` (ARCHIVED) | `tests/` | `python -m pytest tests -q` | mosquitto (Docker), `pytest-mock` |

Model accuracy is evaluated on holdout data by
`people_detection_pg2/extended_evaluation_metrics/run_eval_and_viz.sh`. It is not a test
suite.

### Coverage

```text
Not defined.
```

### Test Data

- `background_subtraction/data/` — six CDNET-14 sequences, committed via LFS.
- Everything else — datasets, holdout sets, captures — is on the fileserver. Paths are
  recorded in notebooks. If a path does not resolve, the data was moved or removed. Do not
  substitute another dataset silently.
- Captures from customer sites can show people. Treat them as sensitive and keep them on the
  fileserver.

---

## 11. Interfaces and APIs

This repo owns no running services. The interfaces it owns are **artefact handovers**:

| Interface | Direction | Protocol / Format | Description | Specification |
|---|---|---|---|---|
| Model delivery | Out → `rtos_sensor` | Files: `.tflite` (int8 backbone, fp32 head, Vela-compiled) + `source_paths.txt` | `cp_weights.py` copies six artefacts; `source_paths.txt` traces a deployed model back to its training run | `people_detection_pg2/readme.md` |
| Lens parameters | Out → geometry consumers | JSON | Fisheye intrinsics for cam4.2 rev-B | `camera_calibration/README.md` |
| Novelty-filter parameters | Out → `PG2_GW` | JSON | IsolationForest/GMM parameter dumps | `bbox_filter/README.md` |
| Frame capture | In ← sensor | TIFF over USB/Ethernet/SD, 640×480×3 + 176-byte header | Raw frames for datasets and calibration | `alif_ensemble_data_collection/CAMERA_TIFF_TRANSFER.md` |
| Detections (ARCHIVED) | In ← sensor | MQTT + protobuf | Gateway prototype input | `gw_prototype/readme.md` |

### Data Formats / Schemas

- YOLO labels (`.txt`) and CVAT exports — `people_detection_pg2/cam4_revB/`.
- CDNET-14 sequences and `temporalROI.txt` — `background_subtraction/data/`.
- Pose models use `nc: 1`, `kpt_shape: [1, 2]` — one keypoint, the ground point used for px→cm
  projection.

---

## 12. Data and Persistence

- **Git (with LFS)** holds code, small reference data and committed results. About 15.6 GB of
  LFS content at HEAD, 63% of it in `people_detection_pg2`.
- **The fileserver** (`/mnt/pg-fs01`, `/mnt/pg-fs02`) holds datasets, captures and training
  results. This repo does not own its backups or retention.
- **LFS routing** is automatic via `.gitattributes`. Small text files (YOLO label `.txt`,
  metric `.csv`, `temporalROI.txt`) are **deliberately excluded** from LFS. They compress
  about 10:1 in Git, and LFS defeats delta compression.
- **History rewrite (September 2026):** LFS migrations and a targeted purge changed every
  commit hash. The pack went from 1.61 GiB to 379 MiB. This cannot be undone for existing
  clones, so they must re-clone. Details are in `REPO_INVENTORY.md` §8.

There are no databases or schema migrations.

---

## 13. CI/CD

Not applicable. There are no pipelines, and no automated checks run on push or merge. The
checks in [Development Standards](#22-development-standards) are manual.

---

## 14. Deployment

Not applicable. Nothing is deployed from this repo. The deliverables in
[Interfaces](#11-interfaces-and-apis) are copied into the consuming repos. Release, deployment
and rollback belong to those repos. `source_paths.txt` in `rtos_sensor` identifies which
training run a shipped model came from.

---

## 15. Release Management

No versioning, no tags, no releases. Work happens on short-lived feature branches (e.g.
`PG2-46-improve-batch-injection-pipeline`) merged into `master`. As of 2026-09-28, only
`pg2-quality-reduction-experiments` is unmerged.

---

## 16. Observability and Operations

Not applicable — no running services. Training runs write their logs and metrics into the run
directory under `people_detection_pg2/results/`. Use `nvidia-smi` for GPU state. Use
`check_training_batch_status.py` for batch progress.

---

## 17. Troubleshooting

### Data files are ~130 bytes / images won't open

**Symptoms:** images, weights or TIFFs are tiny text files starting with
`version https://git-lfs.github.com/spec/v1`.

**Likely cause:** Git LFS was not installed before cloning, or the clone used
`GIT_LFS_SKIP_SMUDGE=1`.

**Resolution:**

```bash
git lfs install
git lfs pull                                  # or --include="<project>/**" for a subset
```

### Exported int8 model is wrong, but training and export "succeeded"

**Likely cause:** the export ran outside `yolo-export`, so upstream ultralytics was used. Its
detection head does not quantise to int8 correctly. The same happens with `yolo-latest`.

**Diagnosis:**

```bash
python -c "import ultralytics; print(ultralytics.__version__, ultralytics.__file__)"
# correct: 8.3.1 .../people_detection_pg2/ultralytics_yolov8/...
```

**Resolution:** `pyenv shell yolo-export` and re-run training and export. Do not reuse the
output of the wrong environment.

### `git pull` shows unrelated histories or a huge divergence

**Likely cause:** the clone predates the September 2026 history rewrite.

**Resolution:** re-clone. Save any local branches as patches first
(`git format-patch origin/master..<branch>`).

### A notebook commit is megabytes large

**Likely cause:** nbstripout is not installed in this clone. The filter lives in `.git/config`
and is not shared.

**Resolution:** install it (see [6.4](#64-configure-local-environment)), then unstage and
re-add the notebook. Notebooks in ARCHIVED projects, and those whose outputs are the result,
are deliberately exempt. Do not widen the rules.

### A notebook path under `/mnt/pg-fs0*` does not resolve

**Likely cause:** the fileserver is not mounted, or the data was moved or removed.

**Resolution:** check the mount and ask the Algo team. Do not substitute a different dataset.

### CUDA out of memory in a batch run

**Likely cause:** another job is already on the card. With 6 GB per GPU, one run per card is
the practical limit, and `run_training_experiments_batch.py` does not check.

**Diagnosis:** `nvidia-smi`. **Resolution:** pass only free GPUs to `--gpus`, or reduce batch
size or image size.

---

## 18. Known Limitations and Technical Debt

| Item | Impact | Workaround | Tracking |
|---|---|---|---|
| No CI; every commit rule is manual | Notebook outputs, derived datasets or results can be committed by mistake | Pre-commit checklist in §22 | — |
| `people_detection_pg2/results/` etc. hold ~9.85 GB in LFS | 63% of every full clone | Code-only clone + selective `git lfs pull` | `REPO_INVENTORY.md` §7 |
| Camera geometry in four places, `bbox_filter` in three, `objectdetectionmetrics` in two; calibration captures duplicated in `gw_prototype/camera_calibration/` | Fixes land in the wrong copy | Establish the shipping copy first — usually in `PG2_GW` or `algo_system_emulator` | `REPO_INVENTORY.md` §3 |
| `objectdetectionmetrics/` is **behind** `algo_system_emulator`'s copy, which has keypoint support | Porting fixes into the emulator loses features | Port fixes out of the emulator copy | `REPO_INVENTORY.md` §8 |
| `yolo11-pose*.yaml` declare `nc: 80` (YOLOv8 configs correctly use `nc: 1`) | YOLO11 runs train with COCO class count | Left unchanged to keep trained weights comparable | `REPO_INVENTORY.md` §8 |
| `alif_ensemble_data_collection`: USB transfer compiled out (`DC_BOARD` absent from `usb.cproject.yml`) | Checked-in build captures frames but never sends them over USB | See `CAMERA_TIFF_TRANSFER.md` | `REPO_INVENTORY.md` §8 |
| Hard-coded fileserver paths in notebooks | Notebooks break when data moves | None; report broken paths | — |
| ARCHIVED projects are not maintained | Many will not run against current dependencies | Treat them as records | — |
| `gap9` hardware decision not written down | Only the notebook outputs show why Alif was chosen | Outputs kept unstripped on purpose | — |
| Stale names: `gw_cpp_version/readme.md` titled "Gateway Python Prototype"; folder `fps_influence_on_count_acuracy_research` misspelt | Cosmetic / confusing | — | — |
| `camera_qc/tests/` is empty despite being a delivered QC tool | Metric/verdict changes are unverified | Compare against `examples/*_demo_report.pdf` by hand | — |
| Unreferenced leftovers: `alif_beta/out copy/`, `object-detection/Untitled.ipynb` | Clutter | — | `REPO_INVENTORY.md` §8 |

---

## 22. Development Standards

Before committing:

```bash
git check-attr filter -- <notebook>      # expect `nbstripout` in ACTIVE projects
git status                               # nothing from results_*/, runs/, capture batches, data_*fps*/
python -m pytest tests -q                # in projects with tests
```

- **Check the project's status first.** Do not refactor, reformat, modernise or strip
  outputs of ARCHIVED projects.
- **Never commit** derived datasets, training results, `runs/`, `results_*/` or capture
  batches. Check `.gitignore` first.
- **Never change `.gitattributes`** without checking the result with
  `git check-attr filter -- <path>`. Do not widen the nbstripout rules, and do not move small
  text files into LFS.
- Match the surrounding style within a project. They were written years apart and differ.
- New docs follow `background_subtraction/readme.md`: purpose, inputs, outputs, status.
- Results and conclusions go in `PG2.wiki`, not only into notebook output.
- Never commit credentials or local environment files.

---

## 23. Important Design Decisions

| Decision | Reason | Consequence / Constraint | Reference |
|---|---|---|---|
| Int8 backbone + separately exported fp32 detection head | Upstream head does not quantise to int8 | Requires the DeGirum fork and `yolo-export`; two artefacts shipped as a pair | `people_detection_pg2/readme.md`; originated in `leg_position_detectors` |
| Custom `t` model scale (width 0.21) as default | Fits the Ethos-U55 memory/latency budget | Not available in upstream ultralytics | `people_detection_pg2/readme.md` |
| Selective nbstripout | Some notebook outputs are the only record of a result | ARCHIVED and output-is-the-result notebooks keep outputs | `.gitattributes`, `REPO_INVENTORY.md` §8 |
| Small text excluded from LFS | ~10:1 Git delta compression; LFS defeats it | Label/metric/ROI text stays in Git | `.gitattributes` |
| Non-NN background subtraction for Sign of Life | NN compute budget already used by the detector | Bit-exact NumPy ports of OpenCV ops as the C-porting target | `background_subtraction/readme.md` |

---

## 24. Related Documentation

| Document | Purpose |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Conventions for AI agents; the short list of rules not to break |
| [`REPO_INVENTORY.md`](REPO_INVENTORY.md) | Full inventory, Git/LFS audit, remediation log |
| `<project>/README.md` or `readme.md` | What a project does in detail and how to run it |
| `PG2.wiki` → `PG2/Algo/Experiments-&-Reports/` | Experiment and evaluation reports |
