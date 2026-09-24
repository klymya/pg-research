# leg_position_calculator

The non-neural approach to leg position: linear regression from bounding-box geometry to the
person's ground-contact point.

**Status:** ARCHIVED — frozen 2023-11 · 2 tracked files, 89 MB

## Purpose

Test whether a keypoint network is needed at all. If the ground point can be predicted well
enough from a box's coordinates and aspect ratio, that is far cheaper than running a pose model
on a microcontroller.

`box-position-model.ipynb` fits the regression on `box_data.csv` — the box measurements and
their known ground points. Features are the box coordinates and aspect ratio; the target is the
leg position.

The neural alternative is `leg_position_detectors/`, and the line that ultimately shipped is
`people_detection_pg2`, which estimates a keypoint. The outcome recorded here is the fit
itself — the notebook's outputs are the only record of how well it worked, which is why they
are exempt from `nbstripout`.

## Layout

| File | What |
|---|---|
| `box-position-model.ipynb` | The regression fit and its evaluation |
| `box_data.csv` | Training data — 87.9 MB, LFS-backed, with its own explicit `.gitattributes` rule |

## Connections

- Alternative to `leg_position_detectors/` on the same problem.
- Superseded by the keypoint approach in `people_detection_pg2` (`kpt_shape: [1, 2]` — one
  keypoint, the ground point).

## Dependencies

Notebook-only; standard scientific Python (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).
No declared requirements.
