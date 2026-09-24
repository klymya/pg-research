# fps_influence_on_count_acuracy_research

How much does running the detector less often cost you in people-count accuracy? Sweeps
detection period against count-reporting period and reports the error surface.

**Status:** ACTIVE · last change 2026-07 · 912 KB

*(Folder name contains a typo — "acuracy".)*

## Purpose

The sensor's power budget depends on how often it wakes to run inference. Running at a lower
frame rate saves power but risks missing people; reporting counts less often smooths noise but
delays the signal. This project measures both axes at once so the trade can be made on numbers
rather than intuition.

The grid:

| Axis | Values |
|---|---|
| Detection period (input rate) | 10, 20, 30, 60 s |
| Count period (output cadence) | 30, 60, 90, 120 s |

16 combinations, each scored against ground truth with MAE, % of ticks reporting the wrong
count, and exact-match accuracy.

## Working pipeline

```
parse_logs_to_csv.py     gateway logs        -> detections.csv
convert_det_to_gt.py     annotations         -> gt_counts.csv
run_count_aggregation.py detections + period -> aggregated counts per tick
evaluate_counts.py       counts vs gt        -> MAE / %wrong / accuracy
run_sweep.py             ^ all 16 cells      -> artifacts/sweep/, eval_*.csv matrices
per_frame_accuracy.py    per-frame view      -> per_frame_accuracy.csv
report.ipynb             everything          -> report.html
```

`run_sweep.py` is the entry point and **reuses `run_count_aggregation.py` and
`evaluate_counts.py` rather than reimplementing them** — the sweep and a single run therefore
cannot disagree.

```bash
python run_sweep.py          # writes artifacts/sweep/ and the eval matrices
```

Outputs under `artifacts/`: `eval_results.csv` (long form), plus `eval_mae.csv`,
`eval_wrong_pct.csv`, `eval_accuracy.csv` as matrices with detection period on rows and count
period on columns.

## Layout

| Path | What |
|---|---|
| `ControlledENV/` | The source recording: `logs.txt`, `det.txt`, reference frame |
| `artifacts/` | All generated CSVs and the sweep output |
| `report.ipynb` / `report.html` | The written-up result |

## Connections

- Aggregation mirrors `gw_prototype/aggregator.py`; the reference implementation reused here
  comes from `algo_system_emulator`.
- Feeds the reporting-period configuration in `PG2_GW` (`aois.json` reporting periods).
- Consumes logs produced by the gateway, same format `alif_beta/` parses.

## Dependencies

Core: `pandas`; also `numpy`, `scikit-learn`. No `requirements.txt`.

```bash
pip install pandas numpy scikit-learn
```

Scripts expect the `pgpytrain` environment in the original author's setup; any Python 3.10+
with the above will do.
