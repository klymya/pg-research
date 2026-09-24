# alif_beta

Analysis of the PG2 beta deployment: turns `pg2gw` journal logs into structured detections,
plots what each sensor actually reported, and evaluates count smoothing.

**Status:** ACTIVE · last change 2026-09 · 22 MB

## Purpose

Once sensors were in a real building, the question became *why does this device report what it
reports* — spurious detections, counts that flicker, differences between units. The gateway
logs everything, but as prose. This project makes that machine-readable and then answers two
questions with it.

### 1. What is each sensor actually detecting?

`converter.py` parses `pg2gw` journal logs into a JSON list of detections. Its key design
decision: **events are grouped by telemetry message, not by their own line timestamp**, since
one "Telemetry from sensor" block can carry several detections, and grouping by line would
scatter them across time. Three event types are extracted:

| Type | Log line |
|---|---|
| `detection` | `Detection translated ... pixel(...) -> cm(...). conf=...` |
| `filtered_confidence` | `Detection filtered out due to low confidence` |
| `filtered_novelty` | `Detection filtered out by GMM: OUTLIER\|NOVEL` |

The novelty rejections come from `bbox_filter`, so these logs are also how that filter's
behaviour in the field is observed.

`plot_detection_boxplots_all.py`, `visualize_detections.ipynb` and `analyze_experiment.ipynb`
plot the results; `beta_v2/` repeats the exercise on a later firmware with its own telemetry
plotter and captured logs (`pg2gw_2ho.log`, `pg2gw_fp.log`, `pg2gw_outliers.log`).

### 2. How should counts be smoothed?

`smoothing/` compares SMA, WMA, EMA, Gaussian, Median and Savitzky-Golay on real count series,
scoring noise reduction, spike robustness, step responsiveness and whether each is usable
online.

**Conclusion: Median → EMA is the recommended production default** — the median removes spikes,
the EMA smooths what remains, and both are causal. Gaussian scores well but is non-causal, so
it is offline-only; Savitzky-Golay is described as often misleading for count steps. Written up
in `smoothing/smoothing_report.md`.

## Layout

| Path | What |
|---|---|
| `converter.py` | Log → JSON detection parser |
| `parse_pg2gw_logs.ipynb` | Interactive log exploration |
| `smoothing/` | The smoother comparison, `common.py`, and `smoothing_report.md` |
| `beta_v2/` | Second-round telemetry: logs, plotter, images |
| `2025-12-19_10-50-09/` | A captured session, including `pg_em/pg-em.log` |

## Connections

- **Consumes `PG2_GW` journal logs** — the log format is the interface; if the gateway's log
  strings change, `converter.py` breaks.
- Observes `bbox_filter`'s GMM rejections in production.
- The smoothing recommendation belongs in the gateway's aggregation path
  (`gw_prototype/aggregator.py`, `PG2_GW`); as of this writing it is a recommendation, not
  shipped code.
- Count-cadence questions overlap `fps_influence_on_count_acuracy_research/`.

## Dependencies

Core: `pandas`, `numpy`, `matplotlib`, `scikit-learn`; `scipy` and `statsmodels` for some
smoothers. No `requirements.txt`.

```bash
pip install pandas numpy matplotlib scikit-learn scipy statsmodels
```

> The folder `out copy/` is a stray duplicate of `out/`. Left in place rather than deleted,
> since neither is referenced by code.
