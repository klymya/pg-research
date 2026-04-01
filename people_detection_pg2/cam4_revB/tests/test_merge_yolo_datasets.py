from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = "/home/klymy/projects/researches/people_detection_pg2/cam4_revB/merge_yolo_datasets.py"
MODULE_DIR = str(Path(MODULE_PATH).parent)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
SPEC = importlib.util.spec_from_file_location("merge_yolo_datasets", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _make_dataset(root: Path, name: str) -> Path:
    dataset = root / name
    for subfolder in (
        "images/train",
        "images/val",
        "images/holdout",
        "labels/train",
        "labels/val",
        "labels/holdout",
    ):
        (dataset / subfolder).mkdir(parents=True, exist_ok=True)
    return dataset


def test_merge_yolo_datasets_preserves_holdout_v2_flow(tmp_path: Path) -> None:
    datasets_root = tmp_path / "yolo_dataset"
    base = _make_dataset(datasets_root, "base")
    update = _make_dataset(datasets_root, "update")
    result = datasets_root / "result"

    _write_text(base / "images" / "train" / "base_train.jpg", "img")
    _write_text(base / "labels" / "train" / "base_train.txt", "label")
    _write_text(base / "images" / "holdout" / "holdout_a.jpg", "img")
    _write_text(base / "labels" / "holdout" / "holdout_a.txt", "label")
    (base / "images" / "holdout_v2").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "holdout_v2").mkdir(parents=True, exist_ok=True)
    _write_text(base / "images" / "holdout_v2" / "holdout_a.jpg", "img")
    _write_text(base / "labels" / "holdout_v2" / "holdout_a.txt", "label")

    _write_text(update / "images" / "train" / "upd_train.jpg", "img")
    _write_text(update / "labels" / "train" / "upd_train.txt", "label")
    _write_text(update / "images" / "holdout" / "holdout_b.jpg", "img")
    _write_text(update / "labels" / "holdout" / "holdout_b.txt", "label")
    (base / "stale.cache").write_text("x")

    summary = MODULE.merge_yolo_datasets(
        base_dataset_path=base,
        update_dataset_path=update,
        result_dataset_path=result,
        overwrite=False,
    )

    assert (result / "images" / "train" / "upd_train.jpg").exists()
    assert (result / "labels" / "train" / "upd_train.txt").exists()
    assert (result / "images" / "holdout_v2" / "holdout_a.jpg").exists()
    assert (result / "images" / "holdout_v2" / "holdout_b.jpg").exists()
    assert (result / "project.yaml").exists()
    assert (result / "project_holdout.yaml").exists()
    assert (result / "project_holdout_v2.yaml").exists()
    assert summary["holdout_v2_merge"]["images/holdout_to_holdout_v2"]["already_present"] == 1

    summary_json = json.loads((result / "merge_summary.json").read_text())
    assert summary_json["counts"]["removed_cache_files"] == 1


def test_merge_yolo_datasets_rejects_conflicting_duplicates(tmp_path: Path) -> None:
    datasets_root = tmp_path / "yolo_dataset"
    base = _make_dataset(datasets_root, "base")
    update = _make_dataset(datasets_root, "update")
    result = datasets_root / "result"

    _write_text(base / "images" / "train" / "same.jpg", "base_img")
    _write_text(base / "labels" / "train" / "same.txt", "base_label")
    _write_text(update / "images" / "train" / "same.jpg", "new_img")
    _write_text(update / "labels" / "train" / "same.txt", "new_label")

    try:
        MODULE.merge_yolo_datasets(
            base_dataset_path=base,
            update_dataset_path=update,
            result_dataset_path=result,
            overwrite=False,
        )
    except ValueError as exc:
        assert "Duplicate destination with different content" in str(exc)
    else:
        raise AssertionError("Expected ValueError for conflicting duplicates")
