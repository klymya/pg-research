# detection_area

Generates the printable A4 floor-marker sheet used during detection-area and coverage data
collection.

**Status:** ACTIVE · last change 2026-09 · 32 KB

## Purpose

Coverage and detection-area measurements need known reference points on the floor. This script
produces the sheet you print and lay on a chalk grid: one black circle per A4 page, as large as
a consumer printer can reliably place.

Three details on the sheet exist to protect the measurement:

- **Edge registration ticks** at the four edge midpoints, to align the sheet with the floor grid.
- **A 100 mm scale bar.** If the printer was left on "Fit to page" the circle comes out the
  wrong size and every pixels-per-metre figure derived from it is wrong. Checking the bar with
  a ruler catches that in five seconds.
- **The nominal diameter printed on the sheet**, so stacks from different print runs cannot be
  silently mixed.

Output is vector PDF, so the printer renders at its native dpi with no rasterisation loss.

## Usage

```bash
python make_marker_sheet.py                 # 1 page, 190 mm circle
python make_marker_sheet.py --pages 150     # a full campaign in one print job
python make_marker_sheet.py --diameter 180  # override the size
```

`floor_marker_A4.pdf` is a generated sheet kept for reference.

## Connections

- Feeds the **coverage-test protocol** documented in `PG2.wiki`
  (`coverage-test-protocol-source.md`).
- The measurements it enables validate the detection-area geometry implemented in
  `gw_prototype/detection_area.py` and `pg_camera_geometry`, and calibrated by
  `camera_calibration`.

## Dependencies

`matplotlib` only (Agg backend, no display needed).

```bash
pip install matplotlib
```
