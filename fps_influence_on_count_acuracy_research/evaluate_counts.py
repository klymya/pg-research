"""Evaluate aggregated count reports against ground truth.

Compares the gateway's aggregated per-tick counts
(``ControlledENV/aggregated_counts.csv``, from ``run_count_aggregation.py``)
against the ground-truth room count (``gt_counts.csv``, from
``convert_det_to_gt.py``).

Ground truth is a step function: each GT row holds until the next mark, so every
aggregated tick is matched to the GT value in effect at that time via a
backward ``merge_asof``.

Metrics:
    MAE                 mean absolute error between reported and true counts
    % wrong counts      share of ticks where reported count != true count
    exact-match acc.    100 - % wrong counts

Run with the ``pgpytrain`` env (pandas + scikit-learn):
    ~/.pyenv/versions/pgpytrain/bin/python evaluate_counts.py
    ... evaluate_counts.py --gt gt_counts.csv \
        --pred ControlledENV/aggregated_counts.csv --out ControlledENV/eval_per_tick.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import mean_absolute_error
    _SKLEARN_ERR: Exception | None = None
except Exception as exc:  # env has sklearn but it may be broken (e.g. numpy ABI)
    mean_absolute_error = None
    _SKLEARN_ERR = exc

TS_FORMAT = "%b %d %H:%M:%S"  # "Jul 23 05:20:00" (shared across the pipeline)


def load_series(path: Path, value_name: str) -> pd.DataFrame:
    """Load a ``timestamp,count`` CSV into a sorted ``[ts, <value_name>]`` frame."""
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"], format=TS_FORMAT)
    df = df.rename(columns={"count": value_name})
    return df[["ts", value_name]].sort_values("ts").reset_index(drop=True)


def match_ground_truth(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Attach the GT value in effect at each prediction tick (step function)."""
    merged = pd.merge_asof(pred, gt, on="ts", direction="backward")
    missing = int(merged["gt"].isna().sum())
    if missing:
        print(f"Note: dropping {missing} tick(s) before the first GT mark.")
        merged = merged.dropna(subset=["gt"])
    merged["gt"] = merged["gt"].astype(int)
    return merged


def compute_metrics(merged: pd.DataFrame) -> dict:
    """MAE, wrong-count share and exact-match accuracy for a matched frame."""
    y_true = merged["gt"].to_numpy()
    y_pred = merged["pred"].to_numpy()
    if mean_absolute_error is not None:
        mae = float(mean_absolute_error(y_true, y_pred))
    else:
        mae = float(np.mean(np.abs(y_true - y_pred)))
    n = len(merged)
    wrong = int(np.sum(y_true != y_pred))
    pct_wrong = 100.0 * wrong / n if n else float("nan")
    return {"n": n, "mae": mae, "wrong": wrong,
            "pct_wrong": pct_wrong, "accuracy": 100.0 - pct_wrong}


def evaluate(gt_path: Path, pred_path: Path) -> dict:
    """Load a GT file and a prediction file and return the metrics dict."""
    gt = load_series(gt_path, "gt")
    pred = load_series(pred_path, "pred")
    merged = match_ground_truth(pred, gt)
    return compute_metrics(merged)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, default=here / "gt_counts.csv")
    parser.add_argument("--pred", type=Path, default=here / "ControlledENV" / "aggregated_counts.csv")
    parser.add_argument("--out", type=Path, default=None, help="optional per-tick merged CSV to write")
    args = parser.parse_args()

    for path in (args.gt, args.pred):
        if not path.is_file():
            raise SystemExit(f"Input not found: {path}")

    gt = load_series(args.gt, "gt")
    pred = load_series(args.pred, "pred")
    merged = match_ground_truth(pred, gt)

    if mean_absolute_error is None:
        print(f"Warning: scikit-learn unavailable ({type(_SKLEARN_ERR).__name__}); "
              f"computing MAE with NumPy (identical result).")
    m = compute_metrics(merged)

    print(f"\nEvaluated {m['n']} report ticks "
          f"({merged['ts'].min():%b %d %H:%M} - {merged['ts'].max():%b %d %H:%M})")
    print(f"  MAE                : {m['mae']:.4f}")
    print(f"  Wrong counts       : {m['wrong']}/{m['n']}  ({m['pct_wrong']:.2f}%)")
    print(f"  Exact-match acc.   : {m['accuracy']:.2f}%")

    # Per-true-count breakdown: where the error concentrates.
    breakdown = (
        merged.assign(abs_err=(merged["pred"] - merged["gt"]).abs())
        .groupby("gt")
        .agg(ticks=("pred", "size"),
             mae=("abs_err", "mean"),
             wrong=("abs_err", lambda s: int((s != 0).sum())))
    )
    breakdown["pct_wrong"] = 100.0 * breakdown["wrong"] / breakdown["ticks"]
    print("\nBy ground-truth count:")
    print(breakdown.to_string(float_format=lambda v: f"{v:.2f}"))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out = merged.copy()
        out["timestamp"] = out["ts"].dt.strftime(TS_FORMAT)
        out["abs_err"] = (out["pred"] - out["gt"]).abs()
        out[["timestamp", "gt", "pred", "abs_err"]].to_csv(args.out, index=False)
        print(f"\nWrote per-tick detail -> {args.out}")


if __name__ == "__main__":
    main()
