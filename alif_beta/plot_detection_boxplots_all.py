#!/usr/bin/env python3
"""
For EACH interval produce ONE figure with 4 subplots (2x2).

Each subplot is a MESSAGE-LEVEL time series:
- x = time (UTC) at TELEMETRY MESSAGE resolution (one point per message)
- y = number of detections in that message

No k-binning / grouping. Every telemetry message contributes one y value.

Subplots (per interval):
1) accepted detections per message
2) filtered(conf) detections per message
3) filtered(novelty/GMM) detections per message
4) total detections per message (accepted + conf + novelty)

Intervals (UTC):
- Empty Room (10:20–10:24)
- Single-Person (10:05–10:20 + 10:24–10:30, combined)
- Three-Person (10:30–10:37)
- Seat Switching under Sensor 3 (10:38–10:48)

Output:
- One PNG per interval: out/messages_ts_<interval>.png

Usage:
  python plot_message_timeseries_per_interval.py /path/to/pg2gw.log --date 2025-12-19 --outdir out

If telemetry boundary string differs, edit KW_TELEMETRY.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

# --- Keywords ---
KW_TELEMETRY = "Telemetry from sensor"  # telemetry message boundary
KW_ACCEPTED = "Detection translated"
KW_LOWCONF = "Detection filtered out due to low confidence"
KW_GMM = "Detection filtered out by GMM"

TS_LEN = 15  # "Dec 19 10:45:44"


@dataclass(frozen=True)
class Interval:
    name: str
    time_windows: List[Tuple[datetime, datetime]]  # (start, end) UTC, end exclusive


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_intervals(base_date: date) -> List[Interval]:
    def dt(h: int, m: int) -> datetime:
        return datetime(base_date.year, base_date.month, base_date.day, h, m, 0, tzinfo=timezone.utc)

    return [
        Interval("Empty Room (10:20–10:24)", [(dt(10, 20), dt(10, 24))]),
        Interval("Single-Person (10:05–10:20, 10:24–10:30)", [(dt(10, 5), dt(10, 20)), (dt(10, 24), dt(10, 30))]),
        Interval("Three-Person (10:30–10:37)", [(dt(10, 30), dt(10, 37))]),
        Interval("Seat Switching Sensor 3 (10:38–10:48)", [(dt(10, 38), dt(10, 48))]),
    ]


def total_duration_seconds(windows: List[Tuple[datetime, datetime]]) -> int:
    return int(sum((end - start).total_seconds() for start, end in windows))


def format_time_ranges(windows: List[Tuple[datetime, datetime]]) -> str:
    return ", ".join(f"{s.strftime('%H:%M')}–{e.strftime('%H:%M')}" for s, e in windows)


def parse_ts_prefix(line: str, year: int) -> Optional[datetime]:
    if len(line) < TS_LEN:
        return None
    ts_str = line[:TS_LEN]
    try:
        dt_naive = datetime.strptime(ts_str, "%b %d %H:%M:%S")
        return dt_naive.replace(year=year, tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_messages_from_log(log_path: str, year: int, target_date: date) -> pd.DataFrame:
    """
    Parse telemetry messages and count detections inside each message.

    One row per telemetry message:
      ts, accepted, low_conf, gmm, total
    """
    rows = []

    cur_ts: Optional[datetime] = None
    acc = lo = gmm = 0

    def flush_current():
        nonlocal cur_ts, acc, lo, gmm, rows
        if cur_ts is None:
            return
        total = acc + lo + gmm
        rows.append((cur_ts, acc, lo, gmm, total))
        cur_ts = None
        acc = lo = gmm = 0

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if (KW_TELEMETRY not in line) and (KW_ACCEPTED not in line) and (KW_LOWCONF not in line) and (KW_GMM not in line):
                continue

            ts = parse_ts_prefix(line, year=year)
            if ts is None or ts.date() != target_date:
                continue

            if KW_TELEMETRY in line:
                flush_current()
                cur_ts = ts
                continue

            if cur_ts is None:
                continue

            if KW_ACCEPTED in line:
                acc += 1
            elif KW_LOWCONF in line:
                lo += 1
            elif KW_GMM in line:
                gmm += 1

    flush_current()

    if not rows:
        return pd.DataFrame(columns=["ts", "accepted", "low_conf", "gmm", "total"])

    df = pd.DataFrame(rows, columns=["ts", "accepted", "low_conf", "gmm", "total"])
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def filter_messages_to_interval(messages: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    if messages.empty:
        return messages
    mask = pd.Series(False, index=messages.index)
    for start, end in interval.time_windows:
        mask |= (messages["ts"] >= start) & (messages["ts"] < end)
    return messages.loc[mask].copy().sort_values("ts").reset_index(drop=True)


def plot_interval_message_timeseries(interval: Interval, msgs: pd.DataFrame, out_path: str) -> None:
    """
    One figure per interval, 2x2 subplots, one point per telemetry message.
    """
    duration_s = total_duration_seconds(interval.time_windows)
    time_ranges = format_time_ranges(interval.time_windows)
    n_msgs = len(msgs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.ravel()

    series = [
        ("accepted", "Accepted detections / message"),
        ("low_conf", "Filtered(conf) / message"),
        ("gmm", "Filtered(novelty) / message"),
        ("total", "All detections / message"),
    ]

    if n_msgs == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No telemetry messages in interval", ha="center", va="center")
            ax.set_axis_off()
        fig.suptitle(f"{interval.name} ({time_ranges} UTC)\nMessages: 0 | Duration: {duration_s}s", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    # Use index-based x to avoid datetime formatter issues; show timestamps as tick labels.
    x = list(range(n_msgs))
    t_labels = msgs["ts"].dt.strftime("%H:%M:%S").tolist()
    step = max(1, n_msgs // 10)
    xticks = list(range(0, n_msgs, step))

    for ax, (col, title) in zip(axes, series):
        y = msgs[col].to_numpy()
        ax.plot(x, y, marker="o", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([t_labels[i] for i in xticks], rotation=45, ha="right")

    fig.suptitle(
        f"{interval.name} ({time_ranges} UTC)\nMessages: {n_msgs} | Duration: {duration_s}s",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", help="Path to pg2gw.log")
    ap.add_argument("--date", default="2025-12-19", help="UTC date (YYYY-MM-DD)")
    ap.add_argument("--year", type=int, default=2025, help="Year for syslog timestamps")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--save-csv", action="store_true", help="Save per-message CSV for each interval")
    args = ap.parse_args()

    exp_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    os.makedirs(args.outdir, exist_ok=True)

    messages = parse_messages_from_log(args.log_path, year=args.year, target_date=exp_date)
    if messages.empty:
        raise SystemExit(
            f"No telemetry messages/detections found for {exp_date}. "
            f"Check KW_TELEMETRY='{KW_TELEMETRY}' matches your logs."
        )

    intervals = build_intervals(exp_date)

    for interval in intervals:
        msgs_i = filter_messages_to_interval(messages, interval)
        base = safe_filename(interval.name)
        out_path = os.path.join(args.outdir, f"messages_ts_{base}.png")
        plot_interval_message_timeseries(interval, msgs_i, out_path)
        print(f"Wrote {out_path}")

        if args.save_csv:
            csv_path = os.path.join(args.outdir, f"per_message_{base}.csv")
            msgs_i.to_csv(csv_path, index=False)

    print(f"Done. Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()
