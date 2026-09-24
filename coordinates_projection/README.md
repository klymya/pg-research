# coordinates_projection

Undistortion and ground-plane projection for PG1/PG2 fisheye cameras, plus a study of how
sensor mounting orientation affects position error.

**Status:** ARCHIVED — frozen 2024-08. Kept as a record; not maintained.

## Purpose

Two related questions:

1. **How do you turn a detection box into a floor position?** `undistort_and_projection.ipynb`
   and `demo.ipynb` walk through undistorting a fisheye frame and projecting image coordinates
   onto the ground plane, using the camera parameters and the detection-area polygon.

2. **How much does sensor tilt cost you?** `sensor_orientation_vs_position_error_research/`
   measures position error against ground truth for two recorded sessions
   (`2024-08-19-11-19-00`, `2024-08-19-11-49-25`) under different assumed orientations. The
   intent was to decide how tightly installers must level the sensor.

> **The conclusion was never written down.** `exp0/1/2.ipynb` contain no markdown cells and no
> summary — only code and output plots. Anyone needing the tilt-vs-error result must read the
> notebook outputs directly, and those outputs are the only record. This is why the notebooks
> here are deliberately exempt from `nbstripout`.

## Layout

| Path | What |
|---|---|
| `undistort_and_projection.ipynb` | Fisheye undistortion → ground-plane projection walkthrough |
| `demo.ipynb` | Worked example on three recorded scenes |
| `data/` | Three scene captures: `Cam2_H256_*`, `Cam3_H242_*` (PNG + JSON pairs) |
| `sensor_orientation_vs_position_error_research/` | The tilt study: `exp0/1/2.ipynb`, `cam_params.json`, `detection_utils.py`, two dated capture sets |

Scene filenames encode the camera and mounting height — `Cam2_H256` is camera 2 at 256 cm.

## Connections

- **Depends on `PG_PyTraining`** (sibling repo, 5 references): notebooks do
  `sys.path.append("../PG_PyTraining")` and import its camera and geometry helpers. Without
  that repo checked out alongside, nothing here runs.
- Same projection problem is solved in production by `gw_prototype/fisheye_camera.py`,
  `pg_camera_geometry` and `PG2_GW`. This project is the exploratory version.
- `kpmg_research/` does a similar geometry sanity check against a customer site sketch.

## Dependencies

`numpy`, `opencv-python`, `matplotlib`, `tqdm`, plus the `PG_PyTraining` repo on `sys.path`.
No `requirements.txt`.

```bash
pip install numpy opencv-python matplotlib tqdm
```

## Note for anyone reviving this

The capture sets under `sensor_orientation_vs_position_error_research/` are LFS-backed and
committed, so the experiments are re-runnable — but the `PG_PyTraining` API has moved on since
2024-08 and the imports may no longer resolve.
