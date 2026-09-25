# camera_calibration

Fisheye intrinsic calibration for the PG2 cam4.2 rev-B sensor. Produces the lens parameters
that every downstream pixel→centimetre projection in the system depends on.

**Status:** ACTIVE · last change 2026-09 · 1.3 GB on disk (mostly TIFF captures in LFS)

## Purpose

A fisheye lens distorts the image, so a detection's pixel coordinates cannot be turned into a
floor position without a model of that distortion. This project fits that model.

`calibration_revB.ipynb` runs `cv2.fisheye.calibrate` over chessboard captures in
`data/calibCam4/<n>/` and writes one JSON per calibration run to `params/`:

```json
{ "IMG_W": ..., "IMG_H": ..., "LENS_PARAMS": { "CX": ..., "CY": ..., "FX": ..., "FY": ..., "K": [...] } }
```

`params/cam4_2_rev_b_main.json` is the consolidated result — **this is the file the rest of the
system consumes.** The other five are individual runs kept for comparison.

The goal is a calibration stable enough that position error stays within the PG2 accuracy
budget across the sensor's full field of view (`fovs.json`: cam4.2 is 92.9° horizontal,
122.5° vertical).

## Layout

| Path | What |
|---|---|
| `calibration_revB.ipynb` | The calibration run: load captures → `cv2.fisheye.calibrate` → write `params/*.json` |
| `utils.py` | `calibrate`, `calibrate_robust`, `undistort`, `undistort_upd` |
| `params/` | **Output.** Six lens-parameter JSONs; `cam4_2_rev_b_main.json` is the one in use |
| `fovs.json` | Nominal field of view per camera revision |
| `data/calibCam4/`, `data/calib-cam4revb/` | Chessboard captures (TIFF, LFS-backed) |
| `alfa/` | Pixel-density analysis, CLAHE tests, detection-area drawing (`viz_da_rev_b.ipynb`) |

## Connections

- **Consumed by** `gw_prototype/fisheye_camera.py`, `gw_cpp_version`, `pg_camera_geometry`,
  and `algo_system_emulator/sensor_params.json` — all read the lens parameters produced here.
- **Imports** `gw_prototype` (via `sys.path.append('../gw_prototype')`) to reuse its
  `CameraCalibrationParameters` / detection-area code when visualising.
- **Sibling data:** `gw_prototype/camera_calibration/calib-cam4revb/` is a byte-identical copy
  of `data/calib-cam4revb/`. Two copies, 344 MB each — do not add a third.
- Same rev-C hardware is measured by `camera_qc/` for incoming module QC.

## Dependencies

Core: `numpy`, `opencv-python`, `matplotlib`, `tqdm`.
No `requirements.txt` — install directly:

```bash
pip install numpy opencv-python matplotlib tqdm
```

Notebooks add `../gw_prototype` to `sys.path`, so that project must be present as a sibling.

## Running

```bash
jupyter lab calibration_revB.ipynb
```

Set `idx` to the capture set under `data/calibCam4/<idx>/`; the notebook shuffles and takes 30
images, calibrates, and writes `params/cam4_2_<idx>_rev_b.json`. The TIFFs are LFS-backed —
run `git lfs pull --include="camera_calibration/data/**"` first if they are pointer files.
