"""Sweep detection period x count period, saving each count series and eval matrices.

Grid:
    detection period (input fps)      : 10, 20, 30, 60 s
    count period     (output cadence) : 30, 60, 90, 120 s

For each of the 16 combinations this:
    1. subsamples the detections to the detection period (row-decimation by default),
    2. runs the reused aggregation pipeline at the count period,
    3. saves the per-tick count series to its own CSV, and
    4. evaluates it against the ground truth.

All heavy lifting is imported from the existing scripts -- ``run_count_aggregation``
(aggregation) and ``evaluate_counts`` (metrics) -- nothing is reimplemented.

Outputs (under ``--outdir``, default ``artifacts/sweep``):
    counts_d<dp>_p<cp>_<sub>.csv   one count series per grid cell (timestamp,count)
    eval_results.csv               long-form: detection_period,count_period,mae,pct_wrong,accuracy
    eval_mae.csv                   matrix  rows=detection period, cols=count period
    eval_wrong_pct.csv             matrix  (% of ticks with wrong count)
    eval_accuracy.csv              matrix  (exact-match accuracy %)

Run with the pgpytrain env (needs pandas):
    ~/.pyenv/versions/pgpytrain/bin/python run_sweep.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_count_aggregation import (
    DEFAULT_EMULATOR,
    LOG_TS_FORMAT,
    SUBSAMPLERS,
    _load_emulator,
    _make_catch_all_aoi,
    aggregate,
    frames_to_points,
    load_frames,
)
from evaluate_counts import evaluate

DETECTION_PERIODS = [10, 20, 30, 60]
COUNT_PERIODS = [30, 60, 90, 120]

METRIC_FILES = {
    "mae": ("eval_mae.csv", "MAE"),
    "pct_wrong": ("eval_wrong_pct.csv", "Wrong-count %"),
    "accuracy": ("eval_accuracy.csv", "Exact-match accuracy %"),
}


def write_counts(results: list[tuple[int, int]], path: Path) -> None:
    """Write an aggregated count series (``timestamp,count``) to ``path``."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "count"])
        for ts, count in results:
            stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(LOG_TS_FORMAT)
            writer.writerow([stamp, count])


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detections", type=Path, default=here / "artifacts" / "detections.csv")
    parser.add_argument("--gt", type=Path, default=here / "artifacts" / "gt_counts.csv")
    parser.add_argument("--outdir", type=Path, default=here / "artifacts" / "sweep")
    parser.add_argument("--subsample", choices=list(SUBSAMPLERS), default="row",
                        help="fps reduction method (default: row-decimation)")
    parser.add_argument("--emulator", type=Path, default=DEFAULT_EMULATOR)
    parser.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    for path in (args.detections, args.gt):
        if not path.is_file():
            raise SystemExit(f"Input not found: {path}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    Aggregator, AreaOfInterest, Point = _load_emulator(args.emulator)
    frames, ts_bounds = load_frames(args.detections, Point, args.year)

    records = []
    for dp in DETECTION_PERIODS:
        sampled = SUBSAMPLERS[args.subsample](frames, dp)
        points = frames_to_points(sampled)
        for cp in COUNT_PERIODS:
            aoi = _make_catch_all_aoi(AreaOfInterest, Point, cp)
            results = aggregate(points, ts_bounds, Aggregator, aoi, cp)

            count_path = args.outdir / f"counts_d{dp}_p{cp}_{args.subsample}.csv"
            write_counts(results, count_path)

            m = evaluate(args.gt, count_path)
            records.append({"detection_period": dp, "count_period": cp,
                            "mae": m["mae"], "pct_wrong": m["pct_wrong"], "accuracy": m["accuracy"]})
            print(f"d={dp:>3}s p={cp:>3}s  ({len(sampled):>3} frames -> {m['n']:>3} ticks)  "
                  f"MAE={m['mae']:.4f}  wrong={m['pct_wrong']:5.2f}%  acc={m['accuracy']:5.2f}%")

    df = pd.DataFrame(records)
    df.to_csv(args.outdir / "eval_results.csv", index=False)

    for metric, (fname, label) in METRIC_FILES.items():
        matrix = df.pivot(index="detection_period", columns="count_period", values=metric)
        matrix.to_csv(args.outdir / fname)
        print(f"\n{label}  (rows = detection period s, cols = count period s):")
        print(matrix.to_string(float_format=lambda v: f"{v:.3f}"))

    print(f"\nWrote {len(records)} count series + 4 eval files -> {args.outdir}")


if __name__ == "__main__":
    main()
