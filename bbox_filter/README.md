# bbox_filter

Novelty detection over detector bounding boxes: learns what a plausible person-box looks like
for a given camera height, then rejects boxes that fall outside that distribution. Trained in
Python with scikit-learn, but **re-implemented dependency-free in Python and C++** so the same
model can run on the gateway.

**Status:** ACTIVE · last change 2026-09 · 6.6 MB

## Purpose

The detector occasionally emits boxes that are geometrically implausible — wrong aspect ratio
for a standing person, impossible size at that distance from the image centre, and so on.
Rather than hand-tune thresholds, this project fits an unsupervised model over four features
and treats outliers as false positives:

```
[ aspect_ratio, bbox_area, dist_from_center, cam_height ]
```

The constraint that shapes everything here: **the filter must run on the gateway**, where
scikit-learn is not available. So training and inference are deliberately separated — sklearn
fits the model, the parameters are dumped to JSON, and small hand-written implementations load
that JSON at runtime.

## Working pipeline

### 1. Train — `train_gmm_detector.py` / `train_custom_detector.py`

Both read `yolo_annotations.csv` (31,252 labelled boxes with `x, y, w, h, pos_x, pos_y,
cam_height, subset`) and write two JSON files:

| Script | Output directory | Files |
|---|---|---|
| `train_gmm_detector.py` | `gmm_detector_model/` | `gmm_detector_params.json`, `gmm_detector_scaler.json` |
| `train_custom_detector.py` | `novelty_detector_model/` | `custom_detector_params.json`, `custom_detector_scaler.json` |

`train_gmm_sklearn.py` is the reference sklearn fit; `generate_models.py` emits IsolationForest
and scaler parameters in the same JSON form.

### 2. Run without sklearn — the `custom_*.py` modules

`custom_scaler.py`, `custom_gmm_detector.py`, `custom_isolation_forest.py` and
`custom_novelty_detector.py` each expose `from_json(...)` and reproduce sklearn's inference in
plain NumPy. `example_usage.py` shows the full load-and-score path.

### 3. Run on the gateway — `novelty_detectors.hpp`

Header-only C++ (Eigen + nlohmann/json) reading the **same JSON files**. `main.cpp` is a
worked example: load scaler and GMM, build the 4-feature vector, scale, score.

### 4. Verify parity — `test_all_detectors.py`, `test_isolation_forest.py`

Check that the dependency-free implementations agree with sklearn. Run these after any change
to the training code — they are what guarantees the gateway behaves like the notebook.

## Layout

| Path | What |
|---|---|
| `yolo_annotations.csv` | Training data — 31,252 boxes |
| `train_*.py` | Fitting (GMM, custom detector, sklearn reference) |
| `custom_*.py` | Dependency-free Python inference |
| `novelty_detectors.hpp`, `main.cpp` | C++ inference for the gateway |
| `*_model/` | **Output** — the JSON parameter dumps that get shipped |
| `test_*.py` | sklearn-vs-reimplementation parity tests |
| `novelty_detection_models.ipynb`, `visualize_dataset_statistics.ipynb`, `read_yolo_dataset.ipynb` | Exploration and dataset statistics |

## Connections

- **Ships into `PG2_GW`** — see its `gen_novelty_params/`, `novelty_models_params/` and
  `test_novelty_detectors` targets. That is the production copy.
- **Also copied into `algo_system_emulator/src/bbox_filter/`** for the offline pipeline.
- Consumes boxes produced by `people_detection_pg2`; `cam_height` comes from the sensor
  installation parameters that `camera_calibration` and the gateway also use.

> **Three copies exist** (here, `PG2_GW`, `algo_system_emulator`). Establish which one is
> authoritative before changing any of them — the JSON parameter format is the contract
> between them, and the C++ and Python loaders must stay in step.

## Dependencies

Core: `numpy`, `scikit-learn` (training only), `pytest` (tests).
C++ side: Eigen and nlohmann/json, both header-only. No `requirements.txt`.

```bash
pip install numpy scikit-learn pytest
```
