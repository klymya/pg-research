from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = "/home/klymy/projects/researches/people_detection_pg2/cam4_revB/regenerate_yolo_datasets_v2.py"
MODULE_DIR = str(Path(MODULE_PATH).parent)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
SPEC = importlib.util.spec_from_file_location("regenerate_yolo_datasets_v2", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_split_with_fixed_holdout_preserves_holdout_and_keeps_train_val_groups_disjoint() -> None:
    all_paths = [
        "/x/4.2_batch1/4.2_H_cam_238_scenea_1.tif",
        "/x/4.2_batch1/4.2_H_cam_238_scenea_2.tif",
        "/x/4.2_batch1/4.2_H_cam_239_sceneb_1.tif",
        "/x/4.2_batch1/4.2_H_cam_239_sceneb_2.tif",
        "/x/4.2_batch1/4.2_H_cam_240_scenec_1.tif",
        "/x/4.2_batch1/4.2_H_cam___11.tif",
    ]
    split_data, malformed_holdout = MODULE.split_with_fixed_holdout(
        all_paths=all_paths,
        fixed_holdout_paths=["/x/4.2_batch1/4.2_H_cam___11.tif"],
        val_ratio=0.4,
        seed=42,
    )

    assert split_data["holdout"] == ["/x/4.2_batch1/4.2_H_cam___11.tif"]
    assert malformed_holdout and malformed_holdout[0]["path"].endswith("4.2_H_cam___11.tif")

    train_groups = {MODULE.antileakage_split.extract_raw_scene_group_id(path) for path in split_data["train"]}
    val_groups = {MODULE.antileakage_split.extract_raw_scene_group_id(path) for path in split_data["val"]}
    assert train_groups.isdisjoint(val_groups)


def test_compose_dataset_plan_preserves_base_holdout_and_background_rules() -> None:
    batch1_3_spec = MODULE.StageSpec(
        stage_name="batch1_3",
        batch_ids=(1, 2, 3),
        dataset_name="batch1_3",
        subset_json_name="train_val_holdout_subsets_batches_1_3.json",
        raw_roots=(),
        annotation_xmls=(),
        historical_subset_json=None,
        use_fixed_base_holdout=True,
    )
    batch4_5_spec = MODULE.StageSpec(
        stage_name="batch4_5",
        batch_ids=(4, 5),
        dataset_name="batch4_5",
        subset_json_name="train_val_holdout_subsets_batches_4_5.json",
        raw_roots=(),
        annotation_xmls=(),
        historical_subset_json=None,
    )
    stage_plans = {
        "batch1_3": MODULE.StagePlan(
            spec=batch1_3_spec,
            subsets={
                "train": ["/x/4.2_batch1/4.2_H_cam_238_scenea_1.tif"],
                "val": ["/x/4.2_batch2/4.2_H_cam_239_sceneb_1.tif"],
                "holdout": ["/x/4.2_batch3/4.2_H_cam___11.tif"],
            },
            validation={},
            historical_subset_match=False,
        ),
        "batch4_5": MODULE.StagePlan(
            spec=batch4_5_spec,
            subsets={
                "train": ["/x/4.2_batch4/4.2_H_cam_250_scened_1.tif"],
                "val": ["/x/4.2_batch5/4.2_H_cam_251_scenee_1.tif"],
                "holdout": ["/x/4.2_batch5/4.2_H_cam_300_overlap_1.tif"],
            },
            validation={},
            historical_subset_match=False,
        ),
    }

    dataset_plan = MODULE.compose_dataset_plan(
        MODULE.DatasetBuildSpec(
            dataset_name="batch1_5_bg",
            stage_names=("batch1_3", "batch4_5"),
            include_background=True,
        ),
        stage_plans=stage_plans,
        background_paths=["/x/4.2_background/4.2_H_cam_180__5.tif"],
    )

    assert dataset_plan.holdout_sources == ("/x/4.2_batch3/4.2_H_cam___11.tif",)
    assert dataset_plan.holdout_v2_sources == (
        "/x/4.2_batch3/4.2_H_cam___11.tif",
        "/x/4.2_batch5/4.2_H_cam_300_overlap_1.tif",
    )
    assert dataset_plan.background_sources == ("/x/4.2_background/4.2_H_cam_180__5.tif",)
    assert dataset_plan.included_batches == (1, 2, 3, 4, 5)


def test_validate_stage_subsets_can_allow_fixed_holdout_malformed_paths() -> None:
    subsets = {
        "train": ["/x/4.2_H_cam_238_scenea_1.tif", "/x/4.2_H_cam_238_scenea_2.tif"],
        "val": ["/x/4.2_H_cam_239_sceneb_1.tif"],
        "holdout": ["/x/4.2_H_cam___11.tif"],
    }

    validation = MODULE.validate_stage_subsets(
        subsets=subsets,
        protected_holdout_groups=set(),
        target_val_ratio=0.1,
        allow_malformed_paths=True,
        enforce_holdout_group_isolation=False,
        enforce_protected_group_exclusion=False,
    )

    assert validation["is_valid"] is True
    assert validation["malformed_path_count"] == 1


def test_build_artifact_names_uses_margin_10_and_preprocess_suffixes() -> None:
    artifacts = MODULE.build_artifact_names("batch1_7_bg", 0.10)

    assert artifacts.raw == "batch1_7_bg"
    assert artifacts.margin == "batch1_7_bg_margin_10"
    assert artifacts.wb == "batch1_7_bg_margin_10_wb_alif"
    assert artifacts.clahe == "batch1_7_bg_margin_10_wb_alif_clahe"


def test_merge_dataset_without_holdout_v2_keeps_holdout_layout_unchanged(tmp_path: Path) -> None:
    base = tmp_path / "base"
    update = tmp_path / "update"
    result = tmp_path / "result"

    for dataset in (base, update):
        for subfolder in (
            "images/train",
            "images/val",
            "images/holdout",
            "labels/train",
            "labels/val",
            "labels/holdout",
        ):
            (dataset / subfolder).mkdir(parents=True, exist_ok=True)

    (base / "images" / "holdout" / "holdout_a.jpg").write_text("img")
    (base / "labels" / "holdout" / "holdout_a.txt").write_text("")
    (update / "images" / "train" / "bg.jpg").write_text("img")
    (update / "labels" / "train" / "bg.txt").write_text("")

    summary = MODULE.merge_dataset_without_holdout_v2(
        base_dataset_path=base,
        update_dataset_path=update,
        result_dataset_path=result,
        overwrite=False,
    )

    assert (result / "images" / "train" / "bg.jpg").exists()
    assert (result / "images" / "holdout" / "holdout_a.jpg").exists()
    assert not (result / "images" / "holdout_v2").exists()
    assert summary["counts"]["removed_cache_files"] == 0
