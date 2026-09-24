# alif_perf

Parses Ethos-U inference logs into per-stage timings.

**Status:** ACTIVE · last change 2026-09 · 1 file, 140 KB

## Purpose

The Alif firmware prints per-stage profiling as raw cycle counts (`CPU TOTAL: <n> cycles`).
This notebook turns that into seconds per stage per run, which is how inference timing on the
MCU actually gets measured — there is no other tool for it in the repo.

`log_parser.ipynb` contains `parse_perf_runs_from_cycles()`, which:

- reads the CPU clock from `Processor internal clock: <Hz>Hz` in the log, or takes it as an
  argument;
- segments the log into runs at each `INFO - Running inference on image`;
- for every `INFO - Profile for <Stage>:` block, scans to the next stage header and extracts
  that stage's `CPU TOTAL`;
- returns one dict per run mapping stage → seconds.

## Connections

- Consumes logs from firmware built out of `alif_ensemble_data_collection` / `rtos_sensor`
  running a model produced by `people_detection_pg2`.
- The latency half of the picture whose accuracy half lives in
  `people_detection_pg2/extended_evaluation_metrics/`.

## Dependencies

`numpy`, `pandas` (standard `re` for parsing).

```bash
pip install numpy pandas
```
