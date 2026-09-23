"""Read a raw counts CSV and turn it into a validated, typed table.

Nothing is coerced away silently: every row that cannot be parsed is moved to
``ParseResult.rejected`` together with a human-readable reason, and every
assumption the parser had to make is reported as an ``Issue``.
"""

from __future__ import annotations

import csv as csv_module
import io
import math
import re
from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime

import pandas as pd

from .config import Config
from .issues import Issue

SOURCE_ROW = "_csv_row"

SPLIT_FIELDS = ("date", "hour", "floor", "in", "out")
COMBINED_FIELDS = ("datetime", "floor", "in", "out")

FIELD_LABELS = {
    "datetime": "Timestamp",
    "date": "Date",
    "hour": "Hour",
    "floor": "Floor",
    "in": "IN count",
    "out": "OUT count",
}

CLEAN_COLUMNS = ["date", "hour", "floor", "in_count", "out_count", "source_row"]

_HOUR_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?(am|pm)?$")


@dataclass
class ParseResult:
    ok: bool
    mode: str
    mapping: dict[str, str | None]
    headers: list[str]
    delimiter: str
    raw_row_count: int
    clean: pd.DataFrame
    rejected: pd.DataFrame
    issues: list[Issue] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)


def normalize_header(name: str) -> str:
    """'Floor Name ' -> 'floor_name' so header spelling/punctuation does not matter."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


# --------------------------------------------------------------------------- reading


def read_raw(data: bytes | str, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Read the file as *strings only* so we can validate before pandas coerces anything."""
    text = data.decode(cfg.encoding, errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
    if not text.strip():
        raise ValueError("The uploaded file is empty.")

    delimiter = cfg.delimiter
    if not delimiter:
        try:
            delimiter = csv_module.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv_module.Error:
            delimiter = ","

    try:
        df = pd.read_csv(
            io.StringIO(text),
            dtype=str,
            sep=delimiter,
            keep_default_na=False,
            na_filter=False,
            skipinitialspace=True,
            engine="python",
        )
    except Exception as exc:  # malformed CSV, ragged rows, ...
        raise ValueError(f"The file could not be read as CSV (delimiter {delimiter!r}): {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]

    warnings: list[Issue] = []
    header_cells = next(csv_module.reader([text.splitlines()[0]], delimiter=delimiter), [])
    normalized = [normalize_header(c) for c in header_cells]
    duplicates = sorted({n for n in normalized if n and normalized.count(n) > 1})
    if duplicates:
        warnings.append(
            Issue(
                "warning",
                "columns",
                f"The header contains duplicate column name(s): {', '.join(duplicates)}. "
                "They were renamed on read (e.g. 'in.1') - check the column mapping.",
            )
        )
    blank_headers = sum(1 for c in header_cells if not str(c).strip())
    if blank_headers:
        warnings.append(
            Issue("warning", "columns", f"{blank_headers} column(s) in the header have no name.")
        )

    df.insert(0, SOURCE_ROW, range(2, len(df) + 2))  # 1 = header row
    return df, {"delimiter": delimiter, "warnings": warnings}


# --------------------------------------------------------------------------- mapping


def detect_mode(headers: list[str], cfg: Config) -> str:
    """Decide whether time is stored as date+hour or as a single timestamp."""
    if cfg.time_mode in ("split", "combined"):
        return cfg.time_mode
    present = {normalize_header(h) for h in headers}

    def has(field_name: str) -> bool:
        return bool(present & {normalize_header(a) for a in cfg.aliases.get(field_name, [])})

    if has("date") and has("hour"):
        return "split"
    if has("datetime"):
        return "combined"
    return "split"


def resolve_mapping(headers: list[str], cfg: Config, mode: str) -> dict[str, str | None]:
    """Map logical fields -> actual column names. Exact matches win over partial ones."""
    fields = COMBINED_FIELDS if mode == "combined" else SPLIT_FIELDS
    candidates = {h: normalize_header(h) for h in headers if h != SOURCE_ROW}
    mapping: dict[str, str | None] = {f: None for f in fields}
    claimed: set[str] = set()

    for field_name in fields:
        aliases = {normalize_header(a) for a in cfg.aliases.get(field_name, [])}
        for header, norm in candidates.items():
            if header not in claimed and norm in aliases:
                mapping[field_name] = header
                claimed.add(header)
                break

    # Second pass: token-level containment ("count_in" matches alias "in",
    # but "training" does not, because whole tokens must line up).
    for field_name in fields:
        if mapping[field_name]:
            continue
        aliases = sorted(
            {normalize_header(a) for a in cfg.aliases.get(field_name, [])}, key=len, reverse=True
        )
        for header, norm in candidates.items():
            if header in claimed:
                continue
            if any(a and f"_{a}_" in f"_{norm}_" for a in aliases):
                mapping[field_name] = header
                claimed.add(header)
                break
    return mapping


# --------------------------------------------------------------------------- value parsing


def is_missing(value, cfg: Config) -> bool:
    return value is None or str(value).strip().lower() in cfg.missing_token_set


def parse_date_value(value, cfg: Config) -> date_type:
    text = str(value).strip()
    for fmt in cfg.date_formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    timestamp = pd.to_datetime(text, dayfirst=cfg.dayfirst, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"invalid date {text!r}")
    return timestamp.date()


def parse_hour_value(value, cfg: Config) -> tuple[int, bool]:
    """Return (hour, is_sub_hourly). Accepts 8, '08', '08:00', '08:00:00', '8am', '08:00-09:00'."""
    text = re.sub(r"\s+", "", str(value).strip().lower())
    if text.endswith("h") and text[:-1].isdigit():
        text = text[:-1]
    if re.search(r"[-–—]", text):  # hour ranges: keep the start of the interval
        head = re.split(r"[-–—]", text)[0]
        if _HOUR_RE.match(head):
            text = head

    match = _HOUR_RE.match(text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        second = int(match.group(3) or 0)
        meridiem = match.group(4)
        if meridiem:
            if not 1 <= hour <= 12:
                raise ValueError(f"invalid 12-hour time {value!r}")
            hour = hour % 12 + (12 if meridiem == "pm" else 0)
        if not 0 <= hour <= 23:
            raise ValueError(f"hour out of range (0-23): {value!r}")
        if not 0 <= minute <= 59 or not 0 <= second <= 59:
            raise ValueError(f"invalid time {value!r}")
        return hour, bool(minute or second)

    timestamp = pd.to_datetime(str(value).strip(), errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"invalid hour {value!r}")
    return int(timestamp.hour), bool(timestamp.minute or timestamp.second)


def parse_datetime_value(value, cfg: Config) -> tuple[date_type, int, bool]:
    """Return (date, hour, is_sub_hourly) for a single timestamp column."""
    text = str(value).strip()
    for fmt in cfg.date_formats:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.date(), parsed.hour, bool(parsed.minute or parsed.second)
        except ValueError:
            continue
    timestamp = pd.to_datetime(text, dayfirst=cfg.dayfirst, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"invalid timestamp {text!r}")
    return timestamp.date(), int(timestamp.hour), bool(timestamp.minute or timestamp.second)


def parse_count_value(value, cfg: Config) -> int:
    text = str(value).strip().replace(" ", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", text):  # 1,234
        text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"not a number ({value!r})") from None
    if not math.isfinite(number):
        raise ValueError(f"not a finite number ({value!r})")
    if cfg.require_integer_counts and abs(number - round(number)) > 1e-9:
        raise ValueError(f"not a whole number ({value!r})")
    count = int(round(number))
    if count < 0 and not cfg.allow_negative_counts:
        raise ValueError(f"negative count ({value!r})")
    if cfg.max_plausible_hourly_count is not None and count > cfg.max_plausible_hourly_count:
        raise ValueError(f"above the configured maximum of {cfg.max_plausible_hourly_count} ({value!r})")
    return count


def floor_allowed(floor: str, cfg: Config) -> bool:
    if not cfg.allowed_floors:
        return True
    if cfg.floor_case_sensitive:
        return floor in cfg.allowed_floors
    return floor.casefold() in {f.casefold() for f in cfg.allowed_floors}


# --------------------------------------------------------------------------- pipeline


def _empty_result(ok, mode, mapping, headers, delimiter, raw_rows, rejected, issues, extras) -> ParseResult:
    return ParseResult(
        ok=ok,
        mode=mode,
        mapping=mapping,
        headers=headers,
        delimiter=delimiter,
        raw_row_count=raw_rows,
        clean=pd.DataFrame(columns=CLEAN_COLUMNS),
        rejected=rejected,
        issues=issues,
        extra_columns=extras,
    )


def parse(
    data: bytes | str,
    cfg: Config,
    mapping: dict[str, str | None] | None = None,
    mode: str | None = None,
) -> ParseResult:
    """Full pipeline: read -> map columns -> validate every row -> de-duplicate."""
    raw, meta = read_raw(data, cfg)
    headers = [c for c in raw.columns if c != SOURCE_ROW]
    delimiter = meta["delimiter"]
    issues: list[Issue] = list(meta["warnings"])

    mode = mode or detect_mode(headers, cfg)
    fields = COMBINED_FIELDS if mode == "combined" else SPLIT_FIELDS
    mapping = mapping or resolve_mapping(headers, cfg, mode)
    mapping = {f: mapping.get(f) for f in fields}
    extras = [h for h in headers if h not in set(filter(None, mapping.values()))]

    missing_columns = [f for f in fields if not mapping.get(f)]
    if missing_columns:
        issues.append(
            Issue(
                "error",
                "missing columns",
                "Missing required column(s): "
                + ", ".join(FIELD_LABELS[f] for f in missing_columns)
                + ". Columns found in the file: "
                + (", ".join(headers) or "(none)")
                + ". Map them in the sidebar or add the header name to config.yaml.",
            )
        )
        return _empty_result(
            False, mode, mapping, headers, delimiter, len(raw), pd.DataFrame(), issues, extras
        )

    if raw.empty:
        issues.append(Issue("error", "empty file", "The file has a header but no data rows."))
        return _empty_result(
            False, mode, mapping, headers, delimiter, 0, pd.DataFrame(), issues, extras
        )

    records: list[dict] = []
    reject_reasons: dict[int, str] = {}
    bucketed_rows: list[int] = []

    for record in raw.to_dict("records"):
        source_row = int(record[SOURCE_ROW])
        values = {f: record.get(mapping[f], "") for f in fields}
        reasons: list[str] = []

        for f in fields:
            if is_missing(values[f], cfg):
                reasons.append(f"missing {FIELD_LABELS[f]} value (column '{mapping[f]}')")

        parsed_date: date_type | None = None
        parsed_hour: int | None = None
        sub_hourly = False

        if mode == "combined" and not is_missing(values["datetime"], cfg):
            try:
                parsed_date, parsed_hour, sub_hourly = parse_datetime_value(values["datetime"], cfg)
            except ValueError as exc:
                reasons.append(f"{FIELD_LABELS['datetime']}: {exc}")
        elif mode == "split":
            if not is_missing(values["date"], cfg):
                try:
                    parsed_date = parse_date_value(values["date"], cfg)
                except ValueError as exc:
                    reasons.append(f"{FIELD_LABELS['date']}: {exc}")
            if not is_missing(values["hour"], cfg):
                try:
                    parsed_hour, sub_hourly = parse_hour_value(values["hour"], cfg)
                except ValueError as exc:
                    reasons.append(f"{FIELD_LABELS['hour']}: {exc}")

        if sub_hourly:
            if cfg.allow_sub_hourly:
                bucketed_rows.append(source_row)
            else:
                reasons.append(
                    "timestamp is not on the hour - enable 'Accept sub-hourly timestamps' to bucket it"
                )

        floor = "" if is_missing(values["floor"], cfg) else str(values["floor"]).strip()
        if floor and not floor_allowed(floor, cfg):
            reasons.append(
                f"unexpected floor {floor!r} (expected one of: {', '.join(cfg.allowed_floors)})"
            )

        counts: dict[str, int] = {}
        for f in ("in", "out"):
            if is_missing(values[f], cfg):
                continue
            try:
                counts[f] = parse_count_value(values[f], cfg)
            except ValueError as exc:
                reasons.append(f"{FIELD_LABELS[f]}: {exc}")

        if reasons or parsed_date is None or parsed_hour is None or len(counts) < 2:
            reject_reasons[source_row] = "; ".join(reasons) or "row could not be parsed"
            continue

        records.append(
            {
                "date": parsed_date,
                "hour": parsed_hour,
                "floor": floor,
                "in_count": counts["in"],
                "out_count": counts["out"],
                "source_row": source_row,
            }
        )

    clean = pd.DataFrame(records, columns=CLEAN_COLUMNS)

    if bucketed_rows:
        issues.append(
            Issue(
                "info",
                "sub-hourly data",
                f"{len(bucketed_rows)} row(s) had a timestamp that was not on the hour and were "
                "bucketed into their hour (per configuration).",
                rows=bucketed_rows,
            )
        )

    # Exact duplicates: keep one copy, report the rest.
    value_columns = ["date", "hour", "floor", "in_count", "out_count"]
    if not clean.empty:
        duplicate_mask = clean.duplicated(subset=value_columns, keep="first")
        if duplicate_mask.any():
            dropped = clean[duplicate_mask]
            issues.append(
                Issue(
                    "warning",
                    "duplicate rows",
                    f"{len(dropped)} exact duplicate row(s) (same date, hour, floor, IN and OUT). "
                    "One copy of each was kept for the totals; the listed rows were discarded.",
                    rows=[int(r) for r in dropped["source_row"]],
                )
            )
            clean = clean[~duplicate_mask]

        # Same date/hour/floor but different counts: unresolvable, so reject all of them.
        conflict_mask = clean.duplicated(subset=["date", "hour", "floor"], keep=False)
        if conflict_mask.any():
            conflicting = clean[conflict_mask]
            for source_row in conflicting["source_row"]:
                reject_reasons[int(source_row)] = (
                    "conflicting duplicate: another row has the same date/hour/floor but "
                    "different IN/OUT values"
                )
            issues.append(
                Issue(
                    "warning",
                    "conflicting duplicates",
                    f"{len(conflicting)} row(s) share a date/hour/floor with a different row but "
                    "report different counts. They were excluded from all totals - the correct "
                    "value cannot be guessed.",
                    rows=[int(r) for r in conflicting["source_row"]],
                )
            )
            clean = clean[~conflict_mask]

    clean = clean.sort_values(["date", "hour", "floor"], kind="stable").reset_index(drop=True)

    rejected = raw[raw[SOURCE_ROW].isin(reject_reasons.keys())].copy()
    if not rejected.empty:
        rejected.insert(1, "Reason", rejected[SOURCE_ROW].map(reject_reasons))
        rejected = rejected.rename(columns={SOURCE_ROW: "CSV row"}).sort_values("CSV row")
        issues.append(
            Issue(
                "warning",
                "rejected rows",
                f"{len(rejected)} row(s) failed validation and are excluded from every table and "
                "total below. See the rejected-rows table for the exact reason per row.",
                rows=[int(r) for r in rejected["CSV row"]],
            )
        )

    if extras:
        issues.append(
            Issue("info", "unused columns", f"Columns not used by the report: {', '.join(extras)}.")
        )

    # Floor labels that differ only by case/whitespace are almost always the same floor.
    if not clean.empty:
        by_normalized: dict[str, set[str]] = {}
        for label in clean["floor"].unique():
            by_normalized.setdefault(normalize_header(label), set()).add(label)
        variants = {k: v for k, v in by_normalized.items() if len(v) > 1}
        if variants:
            examples = "; ".join(" / ".join(sorted(v)) for v in variants.values())
            issues.append(
                Issue(
                    "warning",
                    "floor labels",
                    "Floor labels that differ only in case or punctuation are reported separately: "
                    f"{examples}. Fix them in the source file if they are the same floor.",
                )
            )

    if cfg.allowed_floors and not clean.empty:
        seen = {f.casefold() for f in clean["floor"].unique()}
        never_seen = [f for f in cfg.allowed_floors if f.casefold() not in seen]
        if never_seen:
            issues.append(
                Issue(
                    "warning",
                    "missing floors",
                    f"Expected floor(s) with no valid data in this file: {', '.join(never_seen)}.",
                )
            )
    elif not cfg.allowed_floors:
        issues.append(
            Issue(
                "info",
                "floor check",
                "No expected floor list configured, so unknown floor labels cannot be detected. "
                "Set 'allowed_floors' in config.yaml (or in the sidebar) to enforce one.",
            )
        )

    if clean.empty:
        issues.append(
            Issue("error", "no data", "No rows passed validation - no report could be produced.")
        )

    return ParseResult(
        ok=not clean.empty,
        mode=mode,
        mapping=mapping,
        headers=headers,
        delimiter=delimiter,
        raw_row_count=len(raw),
        clean=clean,
        rejected=rejected,
        issues=issues,
        extra_columns=extras,
    )
