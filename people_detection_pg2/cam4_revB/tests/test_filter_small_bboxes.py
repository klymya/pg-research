from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = "/home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes.py"
SPEC = importlib.util.spec_from_file_location("filter_small_bboxes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _make_dataset(tmp_path: Path, dataset_name: str = "batch_fixture") -> Path:
    dataset_root = tmp_path / "yolo_dataset" / dataset_name
    for split in ("train", "val", "holdout"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "images" / split / f"{split}_img.jpg").write_bytes(b"jpg")
    (dataset_root / "stale.cache").write_text("old")
    return dataset_root


def test_get_inside_ratio_for_box_fully_inside_inner_region() -> None:
    inside_ratio = MODULE.get_inside_ratio(
        x_center=0.5,
        y_center=0.5,
        width=0.2,
        height=0.2,
        margin=0.1,
    )

    assert inside_ratio == pytest.approx(1.0)


def test_evaluate_label_keeps_threshold_boundary() -> None:
    parsed = MODULE.ParsedLabel(
        class_id=0.0,
        x_center=0.1,
        y_center=0.5,
        width=0.2,
        height=0.2,
        keypoint_x=0.1,
        keypoint_y=0.5,
        original_line="0 0.1 0.5 0.2 0.2 0.1 0.5\n",
    )

    keep_label, inside_ratio, reason = MODULE.evaluate_label(
        parsed_label=parsed,
        margin=0.1,
        min_inside_ratio=0.5,
    )

    assert inside_ratio == pytest.approx(0.5)
    assert keep_label is True
    assert reason is None


def test_evaluate_label_rejects_invalid_bbox() -> None:
    parsed = MODULE.ParsedLabel(
        class_id=0.0,
        x_center=0.5,
        y_center=0.5,
        width=0.0,
        height=0.2,
        keypoint_x=0.5,
        keypoint_y=0.5,
        original_line="0 0.5 0.5 0.0 0.2 0.5 0.5\n",
    )

    keep_label, inside_ratio, reason = MODULE.evaluate_label(
        parsed_label=parsed,
        margin=0.1,
        min_inside_ratio=0.01,
    )

    assert inside_ratio == 0.0
    assert keep_label is False
    assert reason == MODULE.REMOVAL_REASON_INVALID_BBOX


def test_parse_label_line_rejects_malformed_field_count(tmp_path: Path) -> None:
    label_path = tmp_path / "bad.txt"
    try:
        MODULE.parse_label_line("0 0.1 0.2 0.3 0.4 0.5", label_path, 1)
    except MODULE.LabelFormatError as exc:
        assert "Expected exactly 7 numeric fields" in str(exc)
    else:
        raise AssertionError("Expected LabelFormatError for malformed field count")


def test_filter_dataset_creates_reports_and_preserves_flow(tmp_path: Path) -> None:
    src_dataset = _make_dataset(tmp_path, dataset_name="batch_fixture")
    (src_dataset / "labels" / "train" / "train_img.txt").write_text(
        "\n".join(
            [
                "0 0.5 0.5 0.2 0.2 0.5 0.5",
                "0 0.01 0.5 0.2 0.2 0.01 0.5",
            ]
        )
        + "\n"
    )
    (src_dataset / "labels" / "val" / "val_img.txt").write_text(
        "0 0.1 0.5 0.2 0.2 0.1 0.5\n"
    )
    (src_dataset / "labels" / "holdout" / "holdout_img.txt").write_text(
        "0 0.5 0.5 0.0 0.2 0.5 0.5\n"
    )

    dst_dataset = src_dataset.parent / MODULE.build_output_dataset_name("batch_fixture", 0.1)
    report = MODULE.filter_dataset(
        src_dataset_path=src_dataset,
        dst_dataset_path=dst_dataset,
        margin=0.1,
        min_inside_ratio=0.5,
        max_workers=1,
        overwrite=False,
    )

    assert dst_dataset.exists()
    assert not (dst_dataset / "stale.cache").exists()
    assert (dst_dataset / "project.yaml").exists()
    assert (dst_dataset / "project_holdout.yaml").exists()

    train_lines = (dst_dataset / "labels" / "train" / "train_img.txt").read_text().splitlines()
    val_lines = (dst_dataset / "labels" / "val" / "val_img.txt").read_text().splitlines()
    holdout_lines = (dst_dataset / "labels" / "holdout" / "holdout_img.txt").read_text().splitlines()

    assert len(train_lines) == 1
    assert len(val_lines) == 1
    assert holdout_lines == []

    report_json = json.loads((dst_dataset / "filter_report.json").read_text())
    assert report_json["config"]["min_inside_ratio"] == 0.5
    assert report_json["summary"]["total"]["labels_total"] == 4
    assert report_json["summary"]["total"]["labels_removed"] == 2
    assert report_json["summary"]["per_split"]["train"]["labels_removed"] == 1
    assert report_json["summary"]["per_split"]["holdout"]["removed_invalid_bbox"] == 1

    with (dst_dataset / "removed_labels.csv").open("rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["reason"] for row in rows} == {
        MODULE.REMOVAL_REASON_LOW_INSIDE_RATIO,
        MODULE.REMOVAL_REASON_INVALID_BBOX,
    }

    with (dst_dataset / "split_summary.csv").open("rt", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    assert {row["split"] for row in split_rows} == {"train", "val", "holdout"}
    assert report["summary"]["total"]["removal_rate"] == 0.5


def test_build_output_dataset_name_matches_existing_pattern() -> None:
    assert MODULE.build_output_dataset_name("batch_H270_16-44-09", 0.1) == "batch_H270_16-44-09_margin_10"
