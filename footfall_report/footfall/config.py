"""Configuration: everything about the CSV layout lives here, nothing is hard-coded."""

from __future__ import annotations

from dataclasses import dataclass, field, replace  # noqa: F401  (replace re-exported for the UI)
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Fallbacks used when config.yaml is absent or a key is omitted.
DEFAULT_ALIASES: dict[str, list[str]] = {
    "datetime": ["timestamp", "datetime", "date_time", "date and time", "interval_start"],
    "date": ["date", "day", "report_date", "business_date"],
    "hour": ["hour", "time", "hour_of_day", "hour_start", "interval"],
    "floor": ["floor", "level", "zone", "area", "floor_name", "location"],
    "in": ["in", "ins", "in_count", "count_in", "entries", "entering", "enters", "footfall_in"],
    "out": ["out", "outs", "out_count", "count_out", "exits", "exiting", "leaving", "footfall_out"],
}

DEFAULT_MISSING_TOKENS = ["", "na", "n/a", "null", "none", "nan", "-", "--", "?"]


@dataclass
class Config:
    """CSV structure + validation policy. Loaded from config.yaml, overridable in the UI."""

    # How time is stored: "split" (separate date and hour columns),
    # "combined" (one timestamp column) or "auto" (detect from the header).
    time_mode: str = "auto"

    # field name -> accepted column headers (case/punctuation insensitive)
    aliases: dict[str, list[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_ALIASES.items()})

    # Explicit strptime formats tried before pandas' own parser, e.g. ["%d/%m/%Y"].
    date_formats: list[str] = field(default_factory=list)
    dayfirst: bool = False

    # Empty list = accept any floor label. Non-empty = anything else is rejected.
    allowed_floors: list[str] = field(default_factory=list)
    floor_case_sensitive: bool = False

    # Count validation
    allow_negative_counts: bool = False
    require_integer_counts: bool = True
    max_plausible_hourly_count: int | None = None  # None = no upper bound

    # Timestamps that are not on the hour (10:30): False = reject, True = bucket into the hour
    allow_sub_hourly: bool = False

    # Values treated as "missing" (compared lower-cased and stripped)
    missing_tokens: list[str] = field(default_factory=lambda: list(DEFAULT_MISSING_TOKENS))

    # File reading
    delimiter: str | None = None  # None = sniff
    encoding: str = "utf-8-sig"

    @property
    def missing_token_set(self) -> set[str]:
        return {str(t).strip().lower() for t in self.missing_tokens}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load config.yaml if present; missing keys fall back to the defaults above."""
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}

        cfg = cls()
        columns = data.get("columns") or {}
        if columns:
            aliases = {k: list(v) for k, v in cfg.aliases.items()}
            for field_name, values in columns.items():
                if values:
                    aliases[field_name] = [str(v) for v in values]
            cfg.aliases = aliases

        counts = data.get("counts") or {}
        csv_opts = data.get("csv") or {}

        cfg.time_mode = str(data.get("time_mode", cfg.time_mode)).lower()
        cfg.date_formats = [str(f) for f in (data.get("date_formats") or [])]
        cfg.dayfirst = bool(data.get("dayfirst", cfg.dayfirst))
        cfg.allowed_floors = [str(f) for f in (data.get("allowed_floors") or [])]
        cfg.floor_case_sensitive = bool(data.get("floor_case_sensitive", cfg.floor_case_sensitive))
        cfg.allow_sub_hourly = bool(data.get("allow_sub_hourly", cfg.allow_sub_hourly))
        if data.get("missing_tokens") is not None:
            cfg.missing_tokens = [str(t) for t in data["missing_tokens"]]

        cfg.allow_negative_counts = bool(counts.get("allow_negative", cfg.allow_negative_counts))
        cfg.require_integer_counts = bool(counts.get("require_integer", cfg.require_integer_counts))
        max_count = counts.get("max_plausible_hourly_count")
        cfg.max_plausible_hourly_count = int(max_count) if max_count not in (None, "") else None

        delimiter = csv_opts.get("delimiter")
        cfg.delimiter = str(delimiter) if delimiter not in (None, "", "auto") else None
        cfg.encoding = str(csv_opts.get("encoding") or cfg.encoding)

        if cfg.time_mode not in ("auto", "split", "combined"):
            raise ValueError(f"config.yaml: time_mode must be auto|split|combined, got {cfg.time_mode!r}")
        return cfg
