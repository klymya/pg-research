# tracker

ByteTrack multi-object tracking: evaluation in Python, plus a C++ port for the Alif MCU.

**Status:** ARCHIVED — frozen 2024-05 · 1,864 tracked files, of which **~30 are PointGrab's**

## Purpose

Detection alone gives boxes per frame; counting people entering and leaving a space needs those
boxes linked into tracks. ByteTrack was the candidate, and the question was twofold: does it
track PG footage well enough, and will it fit on the sensor?

`alif/` is the answer to the second — a C++ port (BYTETracker, STrack, Kalman filter, LAPJV
assignment) with Eigen, sized for the MCU rather than a workstation.

## What is actually PointGrab's

The file count is misleading. Of 1,864 tracked files:

| Path | Files | What |
|---|---:|---|
| `alif/eigen-3.3.9/` | 1,609 | **Vendored Eigen** |
| `ByteTrack/` | 223 | **Vendored upstream ByteTrack** checkout |
| `alif/src`, `alif/include` | 21 | **PG work** — the C++ port |
| repo root | 7 | **PG work** — the Python evaluation |
| `alif/` build files + vendored `nlohmann` | 4 | `CMakeLists.txt`, `main.cpp`, `bytetrack.cpp`, JSON header |

So the PG contribution is `alif/{src,include}` plus the seven root scripts. Everything else is
third-party code committed in place.

| Script | Role |
|---|---|
| `run_tracker.py` | Run ByteTrack over detections (`--track_thresh`, `--track_buffer`) |
| `run_model.py`, `model.py` | Produce the detections to track |
| `eval.py` | Score the tracks |
| `vis.py`, `vis_tracks.py` | Visualise detections and tracks |

## Connections

- Tracks detections of the kind `people_detection_pg2` produces.
- The C++ port targets the same Alif MCU as `background_subtraction`'s Sign-of-Life port.
- Imports `PG_PyTraining` helpers and `metricsevaluator` (see `objectdetectionmetrics/`).
- Upstream: <https://github.com/ifzhang/ByteTrack>.

## Dependencies

Python: `numpy`, `opencv-python`, `tqdm`, `tensorflow`, plus the vendored `ByteTrack` package
on `sys.path`.
C++: Eigen (vendored) and the sources under `alif/`.

> **The vendored trees are candidates for removal** — Eigen and ByteTrack should be
> dependencies, not committed source. They account for 1,832 of the 1,864 tracked files. Check
> nothing builds against the in-tree copies before deleting.
