from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image


def _build_fake_cv2_module():
    module = types.ModuleType("cv2")
    module.COLOR_BGR2RGB = 1
    module.COLOR_RGB2BGR = 2
    module.COLOR_RGB2LAB = 3
    module.COLOR_LAB2RGB = 4

    def imread(path: str):
        try:
            return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
        except Exception:
            return None

    def imwrite(path: str, img):
        Image.fromarray(img[:, :, ::-1]).save(path)
        return True

    def cvtColor(img, code):
        if code == module.COLOR_BGR2RGB or code == module.COLOR_RGB2BGR:
            return img[:, :, ::-1].copy()
        if code == module.COLOR_RGB2LAB or code == module.COLOR_LAB2RGB:
            return img.copy()
        raise ValueError(code)

    def split(img):
        return [img[:, :, idx].copy() for idx in range(img.shape[2])]

    def merge(channels):
        return np.stack(channels, axis=2)

    class _Clahe:
        def apply(self, channel):
            return channel

    def createCLAHE(*args, **kwargs):
        return _Clahe()

    module.imread = imread
    module.imwrite = imwrite
    module.cvtColor = cvtColor
    module.split = split
    module.merge = merge
    module.createCLAHE = createCLAHE
    return module


if "cv2" not in sys.modules:
    sys.modules["cv2"] = _build_fake_cv2_module()


MODULE_PATH = "/home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py"
MODULE_DIR = str(Path(MODULE_PATH).parent)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
SPEC = importlib.util.spec_from_file_location("preprocess_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _make_rgb_image(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((8, 8, 3), value, dtype=np.uint8)
    Image.fromarray(img).save(path)


def _make_dataset(tmp_path: Path, name: str) -> Path:
    dataset = tmp_path / "yolo_dataset" / name
    for split in ("train", "val", "holdout"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    _make_rgb_image(dataset / "images" / "train" / "img1.jpg", value=40)
    _make_rgb_image(dataset / "images" / "val" / "img2.jpg", value=70)
    _make_rgb_image(dataset / "images" / "holdout" / "img3.jpg", value=100)
    (dataset / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2 0.5 0.5\n")
    (dataset / "labels" / "val" / "img2.txt").write_text("0 0.5 0.5 0.2 0.2 0.5 0.5\n")
    (dataset / "labels" / "holdout" / "img3.txt").write_text("0 0.5 0.5 0.2 0.2 0.5 0.5\n")
    (dataset / "stale.cache").write_text("x")
    return dataset


def test_build_output_dataset_name() -> None:
    assert MODULE.build_output_dataset_name("batch12_margin_10", "wb_alif") == "batch12_margin_10_wb_alif"


def test_preprocess_dataset_writes_summary_and_projects(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path, "batch12_margin_10")
    dst = src.parent / "batch12_margin_10_wb_alif"

    summary = MODULE.preprocess_dataset(
        base_dataset_path=src,
        output_dataset_path=dst,
        mode="wb_alif",
        overwrite=False,
        max_workers=1,
    )

    assert dst.exists()
    assert not (dst / "stale.cache").exists()
    assert (dst / "project.yaml").exists()
    assert (dst / "project_holdout.yaml").exists()
    assert summary["counts"]["images_total"] == 3
    assert summary["counts"]["images_processed"] == 3
    assert summary["counts"]["issue_count"] == 0

    summary_json = json.loads((dst / "preprocess_summary.json").read_text())
    assert summary_json["config"]["mode"] == "wb_alif"
    assert summary_json["counts"]["removed_cache_files"] == 1


def test_preprocess_dataset_fail_on_issues(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path, "batch12_margin_10")
    (src / "images" / "train" / "img1.jpg").write_text("not an image")
    dst = src.parent / "batch12_margin_10_wb_alif_clahe"

    try:
        MODULE.preprocess_dataset(
            base_dataset_path=src,
            output_dataset_path=dst,
            mode="wb_alif_clahe",
            overwrite=False,
            max_workers=1,
            fail_on_issues=True,
        )
    except ValueError as exc:
        assert "Preprocessing issues detected" in str(exc)
    else:
        raise AssertionError("Expected ValueError when preprocessing with fail_on_issues")
