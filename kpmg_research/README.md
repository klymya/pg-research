# kpmg_research

Geometry sanity check against a real customer installation: overlay the computed detection-area
polygon on the site's own sketch and verify that pixel→centimetre distances agree with it.

**Status:** ARCHIVED — frozen 2026-03 · 6 files, 4.1 MB

## Purpose

The detection area and px→cm conversion are derived from the lens model and the mounting
height. This checks that derivation against an independent source — a customer-supplied site
drawing with real dimensions — rather than against the model's own assumptions.

`research.ipynb` builds a `PgCamera` and `PgDetectionArea` for a Cam3 at 260 cm, gets the
detection-area polygon in both centimetres and pixels, overlays it on the site sketch
(`SKETCH_data_PG-L18-38.jpg`), and measures the distance between two marked points to compare
against the drawing.

This is the kind of validation that catches a wrong FOV constant or a mis-set mounting height
before it becomes a customer complaint.

## Layout

| File | What |
|---|---|
| `research.ipynb` | The whole analysis |
| `SKETCH_data_PG-L18-38.jpg` | Customer site drawing |
| `Screenshot 2026-03-06 at *.png` | Reference captures from the site (macOS default names) |

## Connections

**Imports from two places at once**, which is unusual and worth knowing:

- `PG_PyTraining` — `common.pg.pg_geometry` (`PgCamera`, `PgDetectionArea`) and
  `common.pg.cam_params`
- `gw_prototype` — `fisheye_camera.CameraCalibrationParameters`,
  `detection_area.CameraConfiguration` / `DetectionArea`

It therefore cross-checks **two of the four** geometry implementations against each other and
against a physical drawing. Neither repo needs to be installed — both are added to `sys.path`
as siblings.

- Same validation question as `coordinates_projection/`, but against a real site.

## Outcome

No written conclusion. The notebook's plots and printed distances are the record.

## Dependencies

`numpy`, `opencv-python`, `matplotlib`, plus `PG_PyTraining` and `gw_prototype` present as
siblings. No declared requirements.
