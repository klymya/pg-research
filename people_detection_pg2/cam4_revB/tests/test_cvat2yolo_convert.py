from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image
import pytest


MODULE_PATH = "/home/klymy/projects/researches/people_detection_pg2/cam4_revB/cvat2yolo_convert.py"
SPEC = importlib.util.spec_from_file_location("cvat2yolo_convert", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(64, 128, 192)).save(path)


def _write_xml(path: Path, image_entries: list[str]) -> None:
    xml = "<annotations>\n" + "\n".join(image_entries) + "\n</annotations>\n"
    path.write_text(xml)


def _image_entry(
    image_name: str,
    width: int,
    height: int,
    object_id: str = "1",
    bbox: tuple[float, float, float, float] = (10.0, 10.0, 30.0, 30.0),
    point: tuple[float, float] = (20.0, 20.0),
) -> str:
    xtl, ytl, xbr, ybr = bbox
    x, y = point
    return f"""
  <image id="0" name="{image_name}" width="{width}" height="{height}">
    <box label="Person" xtl="{xtl}" ytl="{ytl}" xbr="{xbr}" ybr="{ybr}">
      <attribute name="ID">{object_id}</attribute>
    </box>
    <points label="Dot" points="{x},{y}">
      <attribute name="ID">{object_id}</attribute>
    </points>
  </image>""".strip()


def test_build_sample_id_preserves_current_naming_convention() -> None:
    relative_key = "4.2_batch12/4.2_H_cam_248_viasCEO_5.tif"
    assert MODULE.build_sample_id(relative_key) == "4.2_batch12____4.2_H_cam_248_viasCEO_5"


def test_canonicalize_path_resolves_relative_paths(tmp_path: Path) -> None:
    root_path = tmp_path / "cam4_revB"
    relative = "4.2_batch12/4.2_H_cam_248_viasCEO_5.tif"
    expected = str((root_path / relative).resolve(strict=False))
    assert MODULE.canonicalize_path(relative, base_path=root_path) == expected


def test_parse_annotation_xml_reports_ambiguous_object_id(tmp_path: Path) -> None:
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text(
        """
<annotations>
  <image id="0" name="4.2_batch12/4.2_H_cam_248_viasCEO_5.tif" width="64" height="48">
    <box label="Person" xtl="1" ytl="2" xbr="10" ybr="20"><attribute name="ID">1</attribute></box>
    <points label="Dot" points="5,6"><attribute name="ID">1</attribute></points>
    <points label="Dot" points="7,8"><attribute name="ID">1</attribute></points>
  </image>
</annotations>
""".strip()
    )

    objects_by_image, invalid_records = MODULE.parse_annotation_xml(xml_path)

    assert objects_by_image == {"4.2_batch12/4.2_H_cam_248_viasCEO_5.tif": []}
    assert len(invalid_records) == 1
    assert invalid_records[0].reason == "ambiguous_object_id"


def test_run_conversion_end_to_end(tmp_path: Path) -> None:
    root_path = tmp_path / "cam4_revB"
    batch12 = root_path / "4.2_batch12"
    batch13 = root_path / "4.2_batch13"

    train_path = batch12 / "4.2_H_cam_248_viasCEO_5.tif"
    val_path = batch12 / "4.2_H_cam_249_ViaSroom_100.tif"
    holdout_path = batch13 / "4.2_H_cam_300_menivimlobby_22.tif"
    for path in (train_path, val_path, holdout_path):
        _write_image(path)

    subsets_path = root_path / "train_val_holdout_subsets_batches_12.json"
    subsets_path.write_text(
        json.dumps(
            {
                "train": [str(train_path)],
                "val": [str(val_path)],
                "holdout": [str(holdout_path)],
            }
        )
    )

    xml_path = root_path / "batch12_corrected.xml"
    _write_xml(
        xml_path,
        [
            _image_entry("4.2_batch12/4.2_H_cam_248_viasCEO_5.tif", 64, 48),
            _image_entry("4.2_batch12/4.2_H_cam_249_ViaSroom_100.tif", 64, 48),
            _image_entry("4.2_batch13/4.2_H_cam_300_menivimlobby_22.tif", 64, 48),
        ],
    )

    output_dataset_path = root_path / "yolo_dataset" / "batch_12_fixture"
    summary = MODULE.run_conversion(
        root_path=root_path,
        annotation_xmls=[xml_path],
        image_roots=[batch12, batch13],
        subsets_json_path=subsets_path,
        output_dataset_path=output_dataset_path,
        overwrite=False,
        output_image_ext="jpg",
        max_workers=1,
    )

    assert (output_dataset_path / "images" / "train").exists()
    assert (output_dataset_path / "images" / "val").exists()
    assert (output_dataset_path / "images" / "holdout").exists()
    assert (output_dataset_path / "labels" / "train" / "4.2_batch12____4.2_H_cam_248_viasCEO_5.txt").exists()
    assert (output_dataset_path / "images" / "holdout" / "4.2_batch13____4.2_H_cam_300_menivimlobby_22.jpg").exists()
    assert (output_dataset_path / "project.yaml").exists()
    assert (output_dataset_path / "project_holdout.yaml").exists()

    label_line = (
        output_dataset_path / "labels" / "train" / "4.2_batch12____4.2_H_cam_248_viasCEO_5.txt"
    ).read_text().strip()
    parts = [float(item) for item in label_line.split()]
    assert parts[0] == 0.0
    assert parts[1:] == pytest.approx(
        [0.3125, 0.4166666666666667, 0.3125, 0.4166666666666667, 0.3125, 0.4166666666666667]
    )

    summary_json = json.loads((output_dataset_path / "conversion_summary.json").read_text())
    assert summary_json["counts"]["converted_images_total"] == 3
    assert summary_json["per_split"]["train"]["converted_images"] == 1
    assert summary_json["per_split"]["train"]["samples_with_dropped_objects"] == 0
    assert summary["counts"]["objects_written_total"] == 3


def test_run_conversion_reports_missing_source_image(tmp_path: Path) -> None:
    root_path = tmp_path / "cam4_revB"
    batch12 = root_path / "4.2_batch12"
    existing_path = batch12 / "4.2_H_cam_248_viasCEO_5.tif"
    _write_image(existing_path)

    subsets_path = root_path / "subsets.json"
    missing_path = batch12 / "4.2_H_cam_249_ViaSroom_100.tif"
    subsets_path.write_text(
        json.dumps({"train": [str(existing_path), str(missing_path)], "val": [], "holdout": []})
    )

    xml_path = root_path / "batch12_corrected.xml"
    _write_xml(xml_path, [_image_entry("4.2_batch12/4.2_H_cam_248_viasCEO_5.tif", 64, 48)])

    output_dataset_path = root_path / "yolo_dataset" / "fixture"
    summary = MODULE.run_conversion(
        root_path=root_path,
        annotation_xmls=[xml_path],
        image_roots=[batch12],
        subsets_json_path=subsets_path,
        output_dataset_path=output_dataset_path,
        overwrite=False,
        output_image_ext="jpg",
        max_workers=1,
    )

    assert summary["image_issue_reason_counts"]["missing_source_image"] == 1
    with (output_dataset_path / "image_issues.csv").open("rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["reason"] == "missing_source_image" for row in rows)


def test_run_conversion_reports_missing_annotation(tmp_path: Path) -> None:
    root_path = tmp_path / "cam4_revB"
    batch12 = root_path / "4.2_batch12"
    image_path = batch12 / "4.2_H_cam_248_viasCEO_5.tif"
    _write_image(image_path)

    subsets_path = root_path / "subsets.json"
    subsets_path.write_text(json.dumps({"train": [str(image_path)], "val": [], "holdout": []}))

    xml_path = root_path / "batch12_corrected.xml"
    _write_xml(xml_path, [])

    output_dataset_path = root_path / "yolo_dataset" / "fixture"
    summary = MODULE.run_conversion(
        root_path=root_path,
        annotation_xmls=[xml_path],
        image_roots=[batch12],
        subsets_json_path=subsets_path,
        output_dataset_path=output_dataset_path,
        overwrite=False,
        output_image_ext="jpg",
        max_workers=1,
    )

    assert summary["image_issue_reason_counts"]["missing_annotation"] == 1
    assert summary["counts"]["converted_images_total"] == 0


def test_run_conversion_reports_source_image_not_in_split(tmp_path: Path) -> None:
    root_path = tmp_path / "cam4_revB"
    batch12 = root_path / "4.2_batch12"
    used_image = batch12 / "4.2_H_cam_248_viasCEO_5.tif"
    extra_image = batch12 / "4.2_H_cam_249_ViaSroom_100.tif"
    _write_image(used_image)
    _write_image(extra_image)

    subsets_path = root_path / "subsets.json"
    subsets_path.write_text(json.dumps({"train": [str(used_image)], "val": [], "holdout": []}))

    xml_path = root_path / "batch12_corrected.xml"
    _write_xml(xml_path, [_image_entry("4.2_batch12/4.2_H_cam_248_viasCEO_5.tif", 64, 48)])

    output_dataset_path = root_path / "yolo_dataset" / "fixture"
    summary = MODULE.run_conversion(
        root_path=root_path,
        annotation_xmls=[xml_path],
        image_roots=[batch12],
        subsets_json_path=subsets_path,
        output_dataset_path=output_dataset_path,
        overwrite=False,
        output_image_ext="jpg",
        max_workers=1,
    )

    assert summary["image_issue_reason_counts"]["source_image_not_in_split"] == 1


def test_run_conversion_skips_invalid_bbox_and_writes_empty_label(tmp_path: Path) -> None:
    root_path = tmp_path / "cam4_revB"
    batch12 = root_path / "4.2_batch12"
    image_path = batch12 / "4.2_H_cam_248_viasCEO_5.tif"
    _write_image(image_path)

    subsets_path = root_path / "subsets.json"
    subsets_path.write_text(json.dumps({"train": [str(image_path)], "val": [], "holdout": []}))

    xml_path = root_path / "batch12_corrected.xml"
    _write_xml(
        xml_path,
        [
            _image_entry(
                "4.2_batch12/4.2_H_cam_248_viasCEO_5.tif",
                64,
                48,
                bbox=(30.0, 10.0, 10.0, 30.0),
            )
        ],
    )

    output_dataset_path = root_path / "yolo_dataset" / "fixture"
    summary = MODULE.run_conversion(
        root_path=root_path,
        annotation_xmls=[xml_path],
        image_roots=[batch12],
        subsets_json_path=subsets_path,
        output_dataset_path=output_dataset_path,
        overwrite=False,
        output_image_ext="jpg",
        max_workers=1,
    )

    label_path = output_dataset_path / "labels" / "train" / "4.2_batch12____4.2_H_cam_248_viasCEO_5.txt"
    assert label_path.exists()
    assert label_path.read_text() == ""
    assert summary["invalid_reason_counts"]["invalid_bbox_geometry"] == 1
    assert summary["per_split"]["train"]["empty_label_files"] == 1
    assert summary["per_split"]["train"]["samples_with_dropped_objects"] == 1


def test_run_conversion_fail_on_issues_raises(tmp_path: Path) -> None:
    root_path = tmp_path / "cam4_revB"
    batch12 = root_path / "4.2_batch12"
    image_path = batch12 / "4.2_H_cam_248_viasCEO_5.tif"
    _write_image(image_path)

    subsets_path = root_path / "subsets.json"
    subsets_path.write_text(json.dumps({"train": [str(image_path)], "val": [], "holdout": []}))

    xml_path = root_path / "batch12_corrected.xml"
    _write_xml(xml_path, [])

    output_dataset_path = root_path / "yolo_dataset" / "fixture"
    try:
        MODULE.run_conversion(
            root_path=root_path,
            annotation_xmls=[xml_path],
            image_roots=[batch12],
            subsets_json_path=subsets_path,
            output_dataset_path=output_dataset_path,
            overwrite=False,
            output_image_ext="jpg",
            max_workers=1,
            fail_on_issues=True,
        )
    except MODULE.ConversionValidationError as exc:
        assert "missing_annotation=1" in str(exc)
    else:
        raise AssertionError("Expected ConversionValidationError with --fail-on-issues behavior")


def test_run_conversion_requires_overwrite_flag(tmp_path: Path) -> None:
    root_path = tmp_path / "cam4_revB"
    batch12 = root_path / "4.2_batch12"
    image_path = batch12 / "4.2_H_cam_248_viasCEO_5.tif"
    _write_image(image_path)

    subsets_path = root_path / "subsets.json"
    subsets_path.write_text(json.dumps({"train": [str(image_path)], "val": [], "holdout": []}))

    xml_path = root_path / "batch12_corrected.xml"
    _write_xml(xml_path, [_image_entry("4.2_batch12/4.2_H_cam_248_viasCEO_5.tif", 64, 48)])

    output_dataset_path = root_path / "yolo_dataset" / "fixture"
    MODULE.run_conversion(
        root_path=root_path,
        annotation_xmls=[xml_path],
        image_roots=[batch12],
        subsets_json_path=subsets_path,
        output_dataset_path=output_dataset_path,
        overwrite=False,
        output_image_ext="jpg",
        max_workers=1,
    )

    try:
        MODULE.run_conversion(
            root_path=root_path,
            annotation_xmls=[xml_path],
            image_roots=[batch12],
            subsets_json_path=subsets_path,
            output_dataset_path=output_dataset_path,
            overwrite=False,
            output_image_ext="jpg",
            max_workers=1,
        )
    except FileExistsError as exc:
        assert "Pass --overwrite" in str(exc)
    else:
        raise AssertionError("Expected FileExistsError when rerunning without overwrite")
