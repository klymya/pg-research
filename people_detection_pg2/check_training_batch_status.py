"""Inspect training batch logs and classify experiment status.

This script scans a log directory and prints one status per ``.log`` file using
the markers written by ``run_training_experiments_batch.py`` and
``run_training_pipeline.py``.

Statuses:
    - finished
    - in_process
    - failed
    - not_started

Example:
    python people_detection_pg2/check_training_batch_status.py
    python people_detection_pg2/check_training_batch_status.py --log-dir /path/to/logs
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LOG_DIR = Path("/home/klymy/projects/researches/people_detection_pg2/logs/training_batch_runs")
STATUS_FINISHED = "finished"
STATUS_IN_PROCESS = "in_process"
STATUS_FAILED = "failed"
STATUS_NOT_STARTED = "not_started"
STATUS_ORDER = (
    STATUS_FAILED,
    STATUS_IN_PROCESS,
    STATUS_NOT_STARTED,
    STATUS_FINISHED,
)


class TrainingStatusError(Exception):
    """Raised when the requested log directory cannot be analyzed."""


@dataclass(frozen=True, slots=True)
class LogAnalysis:
    """Parsed status markers for one log file."""

    path: Path
    status: str
    reason: str
    gpu_id: str | None
    started_at: str | None
    finished_at: str | None
    returncode: int | None
    dry_run: bool
    command_error_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze training batch logs and report experiment status."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory containing per-experiment .log files.",
    )
    return parser.parse_args()


def list_log_files(log_dir: Path) -> list[Path]:
    if not log_dir.is_dir():
        raise TrainingStatusError(f"Log directory not found: {log_dir}")
    log_files = sorted(path for path in log_dir.iterdir() if path.is_file() and path.suffix == ".log")
    if not log_files:
        raise TrainingStatusError(f"No .log files found under {log_dir}")
    return log_files


def parse_bool_marker(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def parse_int_marker(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def analyze_log_file(path: Path) -> LogAnalysis:
    gpu_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    returncode: int | None = None
    dry_run = False
    saw_dry_run_execution = False
    command_error_count = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("gpu_id="):
                gpu_id = line.partition("=")[2].strip() or None
            elif line.startswith("dry_run="):
                marker = parse_bool_marker(line.partition("=")[2])
                if marker is not None:
                    dry_run = marker
            elif line.startswith("started_at="):
                started_at = line.partition("=")[2].strip() or None
            elif line.startswith("finished_at="):
                finished_at = line.partition("=")[2].strip() or None
            elif line.startswith("returncode="):
                returncode = parse_int_marker(line.partition("=")[2])

            if "Dry run: command not executed" in line:
                saw_dry_run_execution = True
            if "Error running command:" in line:
                command_error_count += 1

    if dry_run or saw_dry_run_execution:
        return LogAnalysis(
            path=path,
            status=STATUS_NOT_STARTED,
            reason="dry-run log; command was not executed",
            gpu_id=gpu_id,
            started_at=started_at,
            finished_at=finished_at,
            returncode=returncode,
            dry_run=dry_run,
            command_error_count=command_error_count,
        )

    if started_at is None:
        return LogAnalysis(
            path=path,
            status=STATUS_NOT_STARTED,
            reason="no started_at marker found",
            gpu_id=gpu_id,
            started_at=started_at,
            finished_at=finished_at,
            returncode=returncode,
            dry_run=dry_run,
            command_error_count=command_error_count,
        )

    if command_error_count > 0 or (returncode is not None and returncode != 0):
        if command_error_count > 0:
            reason = f"found {command_error_count} 'Error running command' marker(s)"
        else:
            reason = f"finished with non-zero returncode={returncode}"
        return LogAnalysis(
            path=path,
            status=STATUS_FAILED,
            reason=reason,
            gpu_id=gpu_id,
            started_at=started_at,
            finished_at=finished_at,
            returncode=returncode,
            dry_run=dry_run,
            command_error_count=command_error_count,
        )

    if finished_at is None:
        return LogAnalysis(
            path=path,
            status=STATUS_IN_PROCESS,
            reason="started_at found but finished_at marker is missing",
            gpu_id=gpu_id,
            started_at=started_at,
            finished_at=finished_at,
            returncode=returncode,
            dry_run=dry_run,
            command_error_count=command_error_count,
        )

    return LogAnalysis(
        path=path,
        status=STATUS_FINISHED,
        reason="finished_at found and no command error markers detected",
        gpu_id=gpu_id,
        started_at=started_at,
        finished_at=finished_at,
        returncode=returncode,
        dry_run=dry_run,
        command_error_count=command_error_count,
    )


def analyze_log_files(log_files: Iterable[Path]) -> list[LogAnalysis]:
    return [analyze_log_file(path) for path in log_files]


def print_summary(analyses: list[LogAnalysis]) -> None:
    counts = Counter(analysis.status for analysis in analyses)
    print("Summary")
    for status in STATUS_ORDER:
        print(f"  {status}: {counts.get(status, 0)}")
    print(f"  total: {len(analyses)}")


def print_details(analyses: list[LogAnalysis]) -> None:
    print("\nDetails")
    for analysis in sorted(analyses, key=lambda item: (STATUS_ORDER.index(item.status), item.path.name)):
        gpu_display = analysis.gpu_id if analysis.gpu_id is not None else "-"
        started_display = analysis.started_at if analysis.started_at is not None else "-"
        finished_display = analysis.finished_at if analysis.finished_at is not None else "-"
        returncode_display = str(analysis.returncode) if analysis.returncode is not None else "-"
        print(
            f"{analysis.path.name}: {analysis.status} | gpu={gpu_display} | "
            f"started={started_display} | finished={finished_display} | "
            f"returncode={returncode_display} | reason={analysis.reason}"
        )


def main() -> int:
    args = parse_args()
    log_files = list_log_files(args.log_dir.resolve())
    analyses = analyze_log_files(log_files)
    print_summary(analyses)
    print_details(analyses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
