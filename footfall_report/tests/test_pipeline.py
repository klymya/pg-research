"""Tests for the parsing/analysis pipeline.  Run with:  pytest -q  (from the project root)"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footfall import analysis  # noqa: E402
from footfall.config import Config, replace  # noqa: E402
from footfall.parsing import parse  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "sample_data"
CFG = Config.load()

HEADER = b"Date,Hour,Floor,IN,OUT\n"


def rows_of(result):
    return result.clean.to_dict("records")


def reasons_of(result):
    return " | ".join(result.rejected["Reason"])


def test_clean_sample_parses_completely():
    result = parse((SAMPLES / "counts_clean.csv").read_bytes(), CFG)
    assert result.ok
    assert len(result.rejected) == 0
    assert len(result.clean) == result.raw_row_count


def test_totals_are_consistent_across_the_three_tables():
    result = parse((SAMPLES / "counts_clean.csv").read_bytes(), CFG)
    hourly = analysis.hourly_table(result.clean)
    by_floor = analysis.daily_by_floor(result.clean)
    totals = analysis.daily_totals(result.clean)
    kpi = analysis.kpis(result.clean)

    assert hourly["IN"].sum() == by_floor["Total IN"].sum() == totals["Total IN"].sum() == kpi["total_in"]
    assert hourly["OUT"].sum() == by_floor["Total OUT"].sum() == totals["Total OUT"].sum() == kpi["total_out"]
    assert kpi["days"] == len(totals)


def test_alternative_layout_gives_identical_totals():
    """Different headers, ';' delimiter and a single timestamp column -> same numbers."""
    a = parse((SAMPLES / "counts_clean.csv").read_bytes(), CFG)
    b = parse((SAMPLES / "counts_alternative_layout.csv").read_bytes(), CFG)
    assert b.mode == "combined" and b.delimiter == ";"
    assert analysis.kpis(a.clean)["total_in"] == analysis.kpis(b.clean)["total_in"]
    assert analysis.kpis(a.clean)["total_out"] == analysis.kpis(b.clean)["total_out"]


def test_error_rate_formula():
    rate = analysis.error_rate([100, 50, 0], [80, 50, 0])
    assert rate[0] == pytest.approx(abs(100 - 80) / 180 * 100)
    assert rate[1] == 0.0
    assert rate.isna()[2]  # no events at all -> undefined, never 0/0


@pytest.mark.parametrize(
    "row, expected_reason",
    [
        (b"2026-01-01,08:00,F1,,4\n", "missing IN count"),
        (b"2026-01-01,08:00,,5,4\n", "missing Floor"),
        (b"2026-02-31,08:00,F1,5,4\n", "invalid date"),
        (b"2026-01-01,25:00,F1,5,4\n", "hour out of range"),
        (b"2026-01-01,08:00,F1,five,4\n", "not a number"),
        (b"2026-01-01,08:00,F1,5,-4\n", "negative count"),
        (b"2026-01-01,08:00,F1,5.5,4\n", "not a whole number"),
        (b"2026-01-01,08:30,F1,5,4\n", "not on the hour"),
    ],
)
def test_invalid_rows_are_rejected_with_a_reason(row, expected_reason):
    result = parse(HEADER + row, CFG)
    assert len(result.clean) == 0
    assert len(result.rejected) == 1
    assert expected_reason in reasons_of(result)


def test_missing_required_column_is_an_error_and_stops_the_report():
    result = parse(b"Date,Hour,Floor,IN\n2026-01-01,08:00,F1,5\n", CFG)
    assert not result.ok
    assert any(i.severity == "error" and "Missing required column" in i.message for i in result.issues)


def test_exact_duplicates_are_counted_once_and_reported():
    data = HEADER + b"2026-01-01,08:00,F1,5,4\n" * 2
    result = parse(data, CFG)
    assert len(result.clean) == 1
    assert analysis.kpis(result.clean)["total_in"] == 5
    assert any(i.category == "duplicate rows" for i in result.issues)


def test_conflicting_duplicates_are_rejected_not_summed():
    data = HEADER + b"2026-01-01,08:00,F1,5,4\n2026-01-01,08:00,F1,9,4\n"
    result = parse(data, CFG)
    assert len(result.clean) == 0
    assert len(result.rejected) == 2
    assert "conflicting duplicate" in reasons_of(result)


def test_unexpected_floor_is_rejected_when_a_floor_list_is_configured():
    cfg = replace(CFG, allowed_floors=["Floor 1"])
    data = HEADER + b"2026-01-01,08:00,Floor 1,5,4\n2026-01-01,08:00,Basement,5,4\n"
    result = parse(data, cfg)
    assert len(result.clean) == 1
    assert "unexpected floor" in reasons_of(result)


def test_sub_hourly_rows_can_be_bucketed_when_enabled():
    cfg = replace(CFG, allow_sub_hourly=True)
    result = parse(HEADER + b"2026-01-01,08:30,F1,5,4\n", cfg)
    assert rows_of(result)[0]["hour"] == 8
    assert any(i.category == "sub-hourly data" for i in result.issues)


def test_time_and_number_formats():
    data = (
        b"date,time,floor,entries,exits\n"
        b"01/02/2026,8:00 AM,F1,\"1,200\",1150\n"
        b"2026-02-02,17,F1,10,9\n"
        b"2026-02-03,09:00-10:00,F1,10,9\n"
    )
    result = parse(data, CFG)
    assert len(result.clean) == 3
    records = rows_of(result)
    assert records[0]["date"] == datetime.date(2026, 1, 2) and records[0]["in_count"] == 1200
    assert records[1]["hour"] == 17
    assert records[2]["hour"] == 9


def test_dayfirst_switch_changes_interpretation():
    result = parse(HEADER + b"03/04/2026,08:00,F1,5,4\n", replace(CFG, dayfirst=True))
    assert rows_of(result)[0]["date"] == datetime.date(2026, 4, 3)


def test_coverage_checks_find_gaps_and_silent_sensors():
    data = (
        HEADER
        + b"2026-01-01,08:00,F1,5,4\n"
        + b"2026-01-01,09:00,F1,5,4\n"
        + b"2026-01-01,08:00,F2,0,0\n"  # F2 misses 09:00 and records nothing
        + b"2026-01-03,08:00,F1,5,4\n"  # 2026-01-02 missing entirely
        + b"2026-01-03,08:00,F2,5,4\n"
        + b"2026-01-03,09:00,F1,5,4\n"
        + b"2026-01-03,09:00,F2,5,4\n"
    )
    categories = {i.category for i in analysis.coverage_issues(parse(data, CFG).clean)}
    assert {"missing hours", "date gaps", "no activity"} <= categories


def test_empty_file_raises_a_clear_error():
    with pytest.raises(ValueError, match="empty"):
        parse(b"", CFG)
