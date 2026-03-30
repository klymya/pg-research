from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = "/home/klymy/projects/researches/people_detection_pg2/cam4_revB/antileakage_split.py"
SPEC = importlib.util.spec_from_file_location("antileakage_split", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extract_raw_scene_group_id_supports_historical_variants() -> None:
    cases = {
        "/home/klymy/data/cam4_revB/4.2_batch2/4.2_H260_me_2.tif": "4.2_h260_me",
        "/home/klymy/data/cam4_revB/4.2_batch2/4.2_H_cam_300_menivimlobby_22.tif": "4.2_h_cam_300_menivimlobby",
        "/home/klymy/data/cam4_revB/yolo_dataset/batch12/images/train/4.2_batch12____4.2_H_cam_248_viasCEO_5.jpg": "4.2_h_cam_248_viasceo",
        "/home/klymy/data/cam4_revB/yolo_dataset/batch8_10_margin_15_wb_alif/images/train/4.2_H_cam_180__5.jpg": "4.2_h_cam_180_",
    }

    for path, expected in cases.items():
        assert MODULE.extract_raw_scene_group_id(path) == expected


def test_extract_raw_scene_group_id_rejects_malformed_name() -> None:
    bad_path = "/home/klymy/data/cam4_revB/4.2_batch3/4.2_H_cam___12.tif"
    try:
        MODULE.extract_raw_scene_group_id(bad_path)
    except MODULE.GroupIdParseError as exc:
        assert "Cannot extract raw_scene_group_id" in str(exc)
    else:
        raise AssertionError("Expected GroupIdParseError for malformed name")


def test_collect_protected_holdout_groups_reports_malformed(tmp_path: Path) -> None:
    subset_path = tmp_path / "subsets.json"
    subset_data = {
        "train": [],
        "val": [],
        "holdout": [
            "/x/4.2_H_cam_300_menivimlobby_22.tif",
            "/x/4.2_H_cam___12.tif",
        ],
    }
    subset_path.write_text(json.dumps(subset_data))

    protected_groups, source_to_groups, malformed = MODULE.collect_protected_holdout_groups([str(subset_path)])

    assert protected_groups == {"4.2_h_cam_300_menivimlobby"}
    assert source_to_groups[str(subset_path)] == ["4.2_h_cam_300_menivimlobby"]
    assert len(malformed) == 1
    assert malformed[0]["path"].endswith("4.2_H_cam___12.tif")


def test_split_new_batch_keeps_groups_isolated_between_train_and_val() -> None:
    new_paths = [
        "/x/4.2_batch12/4.2_H_cam_238_menivimCS_1.tif",
        "/x/4.2_batch12/4.2_H_cam_238_menivimCS_2.tif",
        "/x/4.2_batch12/4.2_H_cam_239_elevator_1.tif",
        "/x/4.2_batch12/4.2_H_cam_239_elevator_2.tif",
        "/x/4.2_batch12/4.2_H_cam_300_menivimlobby_2.tif",
    ]

    split_data, malformed = MODULE.split_new_batch(
        new_paths=new_paths,
        protected_holdout_groups={"4.2_h_cam_300_menivimlobby"},
        val_ratio=0.4,
        seed=42,
    )

    assert malformed == []
    train_groups = {MODULE.extract_raw_scene_group_id(path) for path in split_data["train"]}
    val_groups = {MODULE.extract_raw_scene_group_id(path) for path in split_data["val"]}
    holdout_groups = {MODULE.extract_raw_scene_group_id(path) for path in split_data["holdout"]}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(holdout_groups)
    assert val_groups.isdisjoint(holdout_groups)
    assert holdout_groups == {"4.2_h_cam_300_menivimlobby"}


def test_validate_split_flags_malformed_paths() -> None:
    split_data = {
        "train": ["/x/4.2_H_cam_238_menivimCS_1.tif"],
        "val": ["/x/4.2_H_cam_239_elevator_1.tif"],
        "holdout": [],
    }
    validation = MODULE.validate_split(
        split_data,
        protected_holdout_groups=set(),
        malformed_paths=[{"path": "/x/4.2_H_cam___12.tif", "error": "bad"}],
        target_val_ratio=0.1,
    )

    assert validation["is_valid"] is False
    assert validation["malformed_path_count"] == 1
