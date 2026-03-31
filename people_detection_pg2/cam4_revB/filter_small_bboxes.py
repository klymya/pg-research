#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_BASE_PTH = Path("/home/klymy/data/cam4_revB/yolo_dataset")
DEFAULT_MARGIN = 0.10
DEFAULT_MIN_INSIDE_RATIO = 0.01
DEFAULT_MAX_WORKERS = 8
RATIO_COMPARISON_EPS = 1e-12
REMOVAL_REASON_LOW_INSIDE_RATIO = "removed_low_inside_ratio"
REMOVAL_REASON_INVALID_BBOX = "removed_invalid_bbox"


class LabelFormatError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedLabel:
    class_id: float
    x_center: float
    y_center: float
    width: float
    height: float
    keypoint_x: float
    keypoint_y: float
    original_line: str


@dataclass(frozen=True)
class RemovalRecord:
    split: str
    label_path: str
    line_index: int
    class_id: float
    x_center: float
    y_center: float
    width: float
    height: float
    keypoint_x: float
    keypoint_y: float
    inside_ratio: float
    reason: str


@dataclass(frozen=True)
class FileReport:
    split: str
    label_path: str
    total_labels: int
    kept_labels: int
    removed_labels: int
    removed_low_inside_ratio: int
    removed_invalid_bbox: int
    removed_records: list[RemovalRecord]


def build_output_dataset_name(base_dataset: str, margin: float) -> str:
    return f"{base_dataset}_margin_{int(margin * 100)}"


def create_project_file(
    file_path: Path,
    dataset_path: Path,
    label_mapping: dict[int, str],
    val_folder: str = "images/val",
) -> None:
    label_lines = "\n".join(f"  {class_id}: {label_name}" for class_id, label_name in label_mapping.items())
    contents = (
        f"path: {dataset_path}\n"
        "train: images/train\n"
        f"val: {val_folder}\n"
        "kpt_shape:\n"
        "- 1\n"
        "- 2\n"
        "names:\n"
        f"{label_lines}\n"
    )
    file_path.write_text(contents)


def create_project_files(dataset_path: Path) -> None:
    create_project_file(dataset_path / "project.yaml", dataset_path, {0: "person"})
    create_project_file(
        dataset_path / "project_holdout.yaml",
        dataset_path,
        {0: "person"},
        val_folder="images/holdout",
    )


def validate_parameters(margin: float, min_inside_ratio: float, max_workers: int) -> None:
    if not (0.0 <= margin < 0.5):
        raise ValueError(f"margin must satisfy 0 <= margin < 0.5, got {margin}")
    if not (0.0 <= min_inside_ratio <= 1.0):
        raise ValueError(
            f"min_inside_ratio must satisfy 0 <= min_inside_ratio <= 1, got {min_inside_ratio}"
        )
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")


def copy_dataset(src_dataset_path: Path, dst_dataset_path: Path, overwrite: bool) -> None:
    if not src_dataset_path.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src_dataset_path}")
    if dst_dataset_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination dataset already exists: {dst_dataset_path}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(dst_dataset_path)

    shutil.copytree(src_dataset_path, dst_dataset_path)

    for cache_path in sorted(dst_dataset_path.rglob("*.cache")):
        cache_path.unlink()


def find_label_files(dataset_path: Path) -> list[Path]:
    labels_root = dataset_path / "labels"
    if not labels_root.exists():
        raise FileNotFoundError(f"Dataset labels directory does not exist: {labels_root}")
    return sorted(path for path in labels_root.rglob("*.txt") if path.is_file())


def infer_split(label_path: Path, labels_root: Path) -> str:
    relative_path = label_path.relative_to(labels_root)
    return relative_path.parts[0] if relative_path.parts else "unknown"


def parse_label_line(line: str, label_path: Path, line_index: int) -> ParsedLabel:
    stripped = line.strip()
    parts = stripped.split()
    if len(parts) != 7:
        raise LabelFormatError(
            f"Expected exactly 7 numeric fields in {label_path}:{line_index}, got {len(parts)}"
        )

    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise LabelFormatError(
            f"Failed to parse numeric label fields in {label_path}:{line_index}"
        ) from exc

    class_id, x_center, y_center, width, height, keypoint_x, keypoint_y = values
    normalized_line = line if line.endswith("\n") else f"{line}\n"
    return ParsedLabel(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        keypoint_x=keypoint_x,
        keypoint_y=keypoint_y,
        original_line=normalized_line,
    )


def get_intersection_area_with_inner_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    margin: float,
) -> float:
    box_left = x_center - width / 2
    box_right = x_center + width / 2
    box_top = y_center - height / 2
    box_bottom = y_center + height / 2

    inner_left = margin
    inner_right = 1.0 - margin
    inner_top = margin
    inner_bottom = 1.0 - margin

    inter_left = max(box_left, inner_left)
    inter_right = min(box_right, inner_right)
    inter_top = max(box_top, inner_top)
    inter_bottom = min(box_bottom, inner_bottom)

    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    return inter_width * inter_height


def get_inside_ratio(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    margin: float,
) -> float:
    box_area = width * height
    if box_area <= 0.0:
        return 0.0

    intersection_area = get_intersection_area_with_inner_box(
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        margin=margin,
    )
    return intersection_area / box_area


def is_invalid_bbox(parsed_label: ParsedLabel) -> bool:
    return parsed_label.width <= 0.0 or parsed_label.height <= 0.0


def evaluate_label(
    parsed_label: ParsedLabel,
    margin: float,
    min_inside_ratio: float,
) -> tuple[bool, float, str | None]:
    inside_ratio = get_inside_ratio(
        x_center=parsed_label.x_center,
        y_center=parsed_label.y_center,
        width=parsed_label.width,
        height=parsed_label.height,
        margin=margin,
    )

    if is_invalid_bbox(parsed_label):
        return False, inside_ratio, REMOVAL_REASON_INVALID_BBOX
    if inside_ratio + RATIO_COMPARISON_EPS < min_inside_ratio:
        return False, inside_ratio, REMOVAL_REASON_LOW_INSIDE_RATIO
    return True, inside_ratio, None


def process_label_file(
    label_path: Path,
    labels_root: Path,
    margin: float,
    min_inside_ratio: float,
) -> FileReport:
    split = infer_split(label_path, labels_root)
    kept_lines: list[str] = []
    removed_records: list[RemovalRecord] = []

    lines = label_path.read_text().splitlines(keepends=True)
    for line_index, line in enumerate(lines, start=1):
        parsed = parse_label_line(line, label_path, line_index)
        keep_label, inside_ratio, reason = evaluate_label(parsed, margin, min_inside_ratio)
        if keep_label:
            kept_lines.append(parsed.original_line)
            continue

        assert reason is not None
        removed_records.append(
            RemovalRecord(
                split=split,
                label_path=str(label_path),
                line_index=line_index,
                class_id=parsed.class_id,
                x_center=parsed.x_center,
                y_center=parsed.y_center,
                width=parsed.width,
                height=parsed.height,
                keypoint_x=parsed.keypoint_x,
                keypoint_y=parsed.keypoint_y,
                inside_ratio=inside_ratio,
                reason=reason,
            )
        )

    label_path.write_text("".join(kept_lines))

    removed_low_inside_ratio = sum(
        1 for record in removed_records if record.reason == REMOVAL_REASON_LOW_INSIDE_RATIO
    )
    removed_invalid_bbox = sum(
        1 for record in removed_records if record.reason == REMOVAL_REASON_INVALID_BBOX
    )
    return FileReport(
        split=split,
        label_path=str(label_path),
        total_labels=len(lines),
        kept_labels=len(kept_lines),
        removed_labels=len(removed_records),
        removed_low_inside_ratio=removed_low_inside_ratio,
        removed_invalid_bbox=removed_invalid_bbox,
        removed_records=removed_records,
    )


def _empty_summary() -> dict[str, Any]:
    return {
        "files_processed": 0,
        "labels_total": 0,
        "labels_kept": 0,
        "labels_removed": 0,
        "removed_low_inside_ratio": 0,
        "removed_invalid_bbox": 0,
        "removal_rate": 0.0,
    }


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    labels_total = int(summary["labels_total"])
    labels_removed = int(summary["labels_removed"])
    summary["removal_rate"] = labels_removed / labels_total if labels_total else 0.0
    return summary


def aggregate_reports(file_reports: list[FileReport]) -> tuple[dict[str, Any], list[RemovalRecord]]:
    totals = _empty_summary()
    per_split: dict[str, dict[str, Any]] = {}
    removed_records: list[RemovalRecord] = []

    for file_report in file_reports:
        split_summary = per_split.setdefault(file_report.split, _empty_summary())

        for summary in (totals, split_summary):
            summary["files_processed"] += 1
            summary["labels_total"] += file_report.total_labels
            summary["labels_kept"] += file_report.kept_labels
            summary["labels_removed"] += file_report.removed_labels
            summary["removed_low_inside_ratio"] += file_report.removed_low_inside_ratio
            summary["removed_invalid_bbox"] += file_report.removed_invalid_bbox

        removed_records.extend(file_report.removed_records)

    per_split = {
        split: _finalize_summary(summary)
        for split, summary in sorted(per_split.items())
    }
    totals = _finalize_summary(totals)
    removed_records.sort(key=lambda record: (record.label_path, record.line_index))
    return {"total": totals, "per_split": per_split}, removed_records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def write_removed_labels_csv(path: Path, removed_records: list[RemovalRecord]) -> None:
    fieldnames = [
        "split",
        "label_path",
        "line_index",
        "class_id",
        "x_center",
        "y_center",
        "width",
        "height",
        "keypoint_x",
        "keypoint_y",
        "inside_ratio",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in removed_records:
            writer.writerow(asdict(record))


def write_split_summary_csv(path: Path, per_split_summary: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "files_processed",
        "labels_total",
        "labels_kept",
        "labels_removed",
        "removed_low_inside_ratio",
        "removed_invalid_bbox",
        "removal_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, summary in per_split_summary.items():
            writer.writerow({"split": split, **summary})


def filter_dataset(
    src_dataset_path: Path,
    dst_dataset_path: Path,
    margin: float,
    min_inside_ratio: float,
    max_workers: int,
    overwrite: bool,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    validate_parameters(margin=margin, min_inside_ratio=min_inside_ratio, max_workers=max_workers)
    copy_dataset(src_dataset_path=src_dataset_path, dst_dataset_path=dst_dataset_path, overwrite=overwrite)
    create_project_files(dst_dataset_path)

    labels_root = dst_dataset_path / "labels"
    label_files = find_label_files(dst_dataset_path)
    if max_workers == 1:
        file_reports = [
            process_label_file(
                label_path=label_path,
                labels_root=labels_root,
                margin=margin,
                min_inside_ratio=min_inside_ratio,
            )
            for label_path in label_files
        ]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            file_reports = list(
                executor.map(
                    lambda label_path: process_label_file(
                        label_path=label_path,
                        labels_root=labels_root,
                        margin=margin,
                        min_inside_ratio=min_inside_ratio,
                    ),
                    label_files,
                )
            )

    summary, removed_records = aggregate_reports(file_reports)

    report_root = report_dir if report_dir is not None else dst_dataset_path
    report_root.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "config": {
            "src_dataset_path": str(src_dataset_path),
            "dst_dataset_path": str(dst_dataset_path),
            "margin": margin,
            "min_inside_ratio": min_inside_ratio,
            "max_workers": max_workers,
        },
        "summary": summary,
        "removed_label_count": len(removed_records),
        "label_file_count": len(label_files),
    }
    write_json(report_root / "filter_report.json", report_payload)
    write_removed_labels_csv(report_root / "removed_labels.csv", removed_records)
    write_split_summary_csv(report_root / "split_summary.csv", summary["per_split"])
    return report_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a YOLO dataset, filter labels using the inner-box intersection rule, "
            "and write additive audit reports."
        )
    )
    parser.add_argument(
        "--base-pth",
        default=str(DEFAULT_BASE_PTH),
        help=f"Base path containing YOLO datasets. Default: {DEFAULT_BASE_PTH}",
    )
    parser.add_argument(
        "--base-dataset",
        required=True,
        help="Dataset folder name under --base-pth, for example batch_H270_16-44-09.",
    )
    parser.add_argument(
        "--output-dataset",
        help="Optional output dataset folder name. Default: <base_dataset>_margin_<int(margin * 100)>",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=f"Inner-box margin in normalized coordinates. Default: {DEFAULT_MARGIN}",
    )
    parser.add_argument(
        "--min-inside-ratio",
        type=float,
        default=DEFAULT_MIN_INSIDE_RATIO,
        help=(
            "Minimum allowed inside_ratio. Labels are removed when inside_ratio < threshold. "
            f"Default: {DEFAULT_MIN_INSIDE_RATIO}"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of worker threads for label filtering. Default: {DEFAULT_MAX_WORKERS}",
    )
    parser.add_argument(
        "--report-dir",
        help=(
            "Optional directory for filter_report.json, removed_labels.csv, and split_summary.csv. "
            "Default: output dataset root."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination dataset if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_pth)
    output_dataset = args.output_dataset or build_output_dataset_name(args.base_dataset, args.margin)
    src_dataset_path = base_path / args.base_dataset
    dst_dataset_path = base_path / output_dataset
    report_dir = Path(args.report_dir) if args.report_dir else None

    report = filter_dataset(
        src_dataset_path=src_dataset_path,
        dst_dataset_path=dst_dataset_path,
        margin=args.margin,
        min_inside_ratio=args.min_inside_ratio,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
        report_dir=report_dir,
    )

    total = report["summary"]["total"]
    removed_pct = total["removal_rate"] * 100.0
    print(
        "Inner-box filtering with "
        f"margin={args.margin} and min_inside_ratio={args.min_inside_ratio} removed "
        f"{total['labels_removed']} / {total['labels_total']} bboxes ({removed_pct:.2f}%)."
    )
    print(f"Filtered dataset: {dst_dataset_path}")
    report_root = report_dir if report_dir is not None else dst_dataset_path
    print(f"Report files: {report_root}")


if __name__ == "__main__":
    main()
