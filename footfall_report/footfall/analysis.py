"""Aggregations and dataset-level data-quality checks.

Input is always the validated frame produced by ``parsing.parse`` with columns
date | hour | floor | in_count | out_count | source_row.
"""

from __future__ import annotations

import pandas as pd

from .issues import Issue

HOURLY_COLUMNS = ["Date", "Hour", "Floor", "IN", "OUT"]
FLOOR_DAILY_COLUMNS = ["Date", "Floor", "Total IN", "Total OUT", "Total events", "Difference", "Error rate %"]
DAILY_COLUMNS = ["Date", "Total IN", "Total OUT", "Total events", "Difference", "Error rate %"]


def error_rate(in_counts, out_counts):
    """|IN - OUT| / (IN + OUT), as a percentage. NaN when there are no events at all.

    This is the IN/OUT imbalance: with a perfect sensor every entry eventually
    produces an exit, so the residual is a proxy for counting error.
    """
    in_counts = pd.Series(in_counts, dtype="float64")
    out_counts = pd.Series(out_counts, dtype="float64")
    total = in_counts + out_counts
    return ((in_counts - out_counts).abs() / total.where(total != 0)) * 100.0


def hour_label(hour: int) -> str:
    return f"{int(hour):02d}:00"


def hourly_table(clean: pd.DataFrame) -> pd.DataFrame:
    """Every hour of every floor, exactly as read from the file."""
    if clean.empty:
        return pd.DataFrame(columns=HOURLY_COLUMNS)
    out = clean.sort_values(["date", "hour", "floor"], kind="stable").copy()
    return pd.DataFrame(
        {
            "Date": out["date"].astype(str),
            "Hour": out["hour"].map(hour_label),
            "Floor": out["floor"],
            "IN": out["in_count"].astype(int),
            "OUT": out["out_count"].astype(int),
        }
    ).reset_index(drop=True)


def daily_by_floor(clean: pd.DataFrame) -> pd.DataFrame:
    """Daily IN/OUT totals per floor."""
    if clean.empty:
        return pd.DataFrame(columns=FLOOR_DAILY_COLUMNS)
    grouped = (
        clean.groupby(["date", "floor"], as_index=False)[["in_count", "out_count"]]
        .sum()
        .sort_values(["date", "floor"], kind="stable")
    )
    return _finalise(grouped, ["date", "floor"], ["Date", "Floor"])


def daily_totals(clean: pd.DataFrame) -> pd.DataFrame:
    """Daily IN/OUT totals across all floors."""
    if clean.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    grouped = (
        clean.groupby(["date"], as_index=False)[["in_count", "out_count"]]
        .sum()
        .sort_values("date", kind="stable")
    )
    return _finalise(grouped, ["date"], ["Date"])


def _finalise(grouped: pd.DataFrame, key_columns: list[str], key_labels: list[str]) -> pd.DataFrame:
    table = pd.DataFrame({label: grouped[key].astype(str) for key, label in zip(key_columns, key_labels)})
    total_in = grouped["in_count"].astype(int)
    total_out = grouped["out_count"].astype(int)
    table["Total IN"] = total_in.values
    table["Total OUT"] = total_out.values
    table["Total events"] = (total_in + total_out).values
    table["Difference"] = (total_in - total_out).values
    table["Error rate %"] = error_rate(total_in, total_out).values
    return table.reset_index(drop=True)


def kpis(clean: pd.DataFrame) -> dict:
    """Headline numbers for the top of the report."""
    if clean.empty:
        return {
            "total_in": 0,
            "total_out": 0,
            "difference": 0,
            "error_rate": None,
            "floors": 0,
            "days": 0,
            "hours_covered": 0,
            "busiest_day": None,
            "busiest_day_events": 0,
            "busiest_hour": None,
            "busiest_hour_events": 0,
            "busiest_slot": None,
            "busiest_floor": None,
        }

    total_in = int(clean["in_count"].sum())
    total_out = int(clean["out_count"].sum())
    events = clean["in_count"] + clean["out_count"]

    by_day = clean.assign(events=events).groupby("date")["events"].sum()
    by_hour = clean.assign(events=events).groupby("hour")["events"].sum()
    by_floor = clean.assign(events=events).groupby("floor")["events"].sum()
    by_slot = clean.assign(events=events).groupby(["date", "hour"])["events"].sum()

    busiest_day = by_day.idxmax()
    busiest_hour = int(by_hour.idxmax())
    busiest_slot_key = by_slot.idxmax()

    denominator = total_in + total_out
    return {
        "total_in": total_in,
        "total_out": total_out,
        "difference": total_in - total_out,
        "error_rate": (abs(total_in - total_out) / denominator * 100.0) if denominator else None,
        "floors": int(clean["floor"].nunique()),
        "days": int(clean["date"].nunique()),
        "hours_covered": int(clean["hour"].nunique()),
        "busiest_day": str(busiest_day),
        "busiest_day_events": int(by_day.max()),
        "busiest_hour": hour_label(busiest_hour),
        "busiest_hour_events": int(by_hour.max()),
        "busiest_slot": f"{busiest_slot_key[0]} {hour_label(busiest_slot_key[1])}",
        "busiest_floor": str(by_floor.idxmax()),
    }


def coverage_issues(clean: pd.DataFrame) -> list[Issue]:
    """Dataset-level checks that only make sense once all valid rows are known."""
    issues: list[Issue] = []
    if clean.empty:
        return issues

    floors = sorted(clean["floor"].unique())
    dates = sorted(clean["date"].unique())

    # A floor that is entirely absent on a day the file otherwise covers.
    absent: list[str] = []
    for day in dates:
        present = set(clean.loc[clean["date"] == day, "floor"])
        for floor in floors:
            if floor not in present:
                absent.append(f"{day}: {floor}")
    if absent:
        issues.append(
            Issue(
                "warning",
                "coverage",
                f"{len(absent)} floor/day combination(s) have no rows at all: "
                + "; ".join(absent[:10])
                + (" ..." if len(absent) > 10 else ""),
            )
        )

    # Hours missing for one floor while other floors report them on the same day.
    gaps: list[str] = []
    for day in dates:
        day_rows = clean[clean["date"] == day]
        expected_hours = sorted(day_rows["hour"].unique())
        for floor in sorted(day_rows["floor"].unique()):
            have = set(day_rows.loc[day_rows["floor"] == floor, "hour"])
            missing = [h for h in expected_hours if h not in have]
            if missing:
                gaps.append(f"{day} / {floor}: {', '.join(hour_label(h) for h in missing)}")
    if gaps:
        issues.append(
            Issue(
                "warning",
                "missing hours",
                f"{len(gaps)} floor/day combination(s) are missing hours that other floors report "
                "on the same day: " + "; ".join(gaps[:10]) + (" ..." if len(gaps) > 10 else ""),
            )
        )

    # Calendar gaps between the first and last date in the file.
    if len(dates) > 1:
        full_range = pd.date_range(min(dates), max(dates), freq="D").date
        missing_days = [str(d) for d in full_range if d not in set(dates)]
        if missing_days:
            issues.append(
                Issue(
                    "info",
                    "date gaps",
                    f"{len(missing_days)} calendar day(s) between {min(dates)} and {max(dates)} have "
                    "no data: " + ", ".join(missing_days[:10]) + (" ..." if len(missing_days) > 10 else ""),
                )
            )

    # Sensors that recorded nothing all day.
    silent = clean.groupby(["date", "floor"])[["in_count", "out_count"]].sum()
    silent = silent[(silent["in_count"] == 0) & (silent["out_count"] == 0)]
    if not silent.empty:
        labels = [f"{d} / {f}" for d, f in silent.index]
        issues.append(
            Issue(
                "warning",
                "no activity",
                f"{len(labels)} floor/day combination(s) recorded zero IN and zero OUT for the whole "
                "day (possible sensor outage): " + "; ".join(labels[:10]) + (" ..." if len(labels) > 10 else ""),
            )
        )

    # Very large imbalances usually mean a miscounting sensor.
    per_floor_day = clean.groupby(["date", "floor"], as_index=False)[["in_count", "out_count"]].sum()
    rates = error_rate(per_floor_day["in_count"], per_floor_day["out_count"])
    flagged = per_floor_day[(rates > 25).fillna(False).values]
    if not flagged.empty:
        labels = [
            f"{row.date} / {row.floor} ({rate:.0f}%)"
            for row, rate in zip(flagged.itertuples(), rates[(rates > 25).fillna(False)])
        ]
        issues.append(
            Issue(
                "info",
                "imbalance",
                f"{len(labels)} floor/day combination(s) have an IN/OUT error rate above 25%: "
                + "; ".join(labels[:10])
                + (" ..." if len(labels) > 10 else ""),
            )
        )

    return issues
