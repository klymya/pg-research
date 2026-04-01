from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

from PIL import Image
import numpy as np


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


MODULE_PATH = "/home/klymy/projects/researches/people_detection_pg2/cam4_revB/run_batch_injection.py"
MODULE_DIR = str(Path(MODULE_PATH).parent)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
SPEC = importlib.util.spec_from_file_location("run_batch_injection", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 140, 180)).save(path)


def _write_xml(path: Path, image_name: str) -> None:
    path.write_text(
        f"""
<annotations>
  <image id="0" name="{image_name}" width="64" height="48">
    <box label="Person" xtl="10" ytl="10" xbr="30" ybr="30">
      <attribute name="ID">1</attribute>
    </box>
    <points label="Dot" points="20,20">
      <attribute name="ID">1</attribute>
    </points>
  </image>
</annotations>
""".strip()
    )


def _make_merge_base_dataset(root: Path, name: str) -> Path:
    dataset = root / "yolo_dataset" / name
    for subfolder in (
        "images/train",
        "images/val",
        "images/holdout",
        "labels/train",
        "labels/val",
        "labels/holdout",
    ):
        (dataset / subfolder).mkdir(parents=True, exist_ok=True)
    _write_image(dataset / "images" / "train" / "base.jpg")
    (dataset / "labels" / "train" / "base.txt").write_text("0 0.5 0.5 0.2 0.2 0.5 0.5\n")
    _write_image(dataset / "images" / "holdout" / "hold.jpg")
    (dataset / "labels" / "holdout" / "hold.txt").write_text("0 0.5 0.5 0.2 0.2 0.5 0.5\n")
    return dataset


def test_enforce_review_gate_requires_confirmation() -> None:
    class Args:
        review_confirmed = False
        review_note = "done"

    try:
        MODULE.enforce_review_gate(Args())
    except ValueError as exc:
        assert "--review-confirmed" in str(exc)
    else:
        raise AssertionError("Expected ValueError when review gate is not confirmed")


def test_run_batch_injection_end_to_end(tmp_path: Path) -> None:
    root_path = tmp_path / "cam4_revB"
    raw_batch = root_path / "4.2_batch12"
    image_path = raw_batch / "4.2_H_cam_248_viasCEO_5.tif"
    _write_image(image_path)

    annotation_xml = root_path / "batch12_corrected.xml"
    _write_xml(annotation_xml, "4.2_batch12/4.2_H_cam_248_viasCEO_5.tif")

    protected_subsets = root_path / "protected.json"
    protected_subsets.write_text(json.dumps({"train": [], "val": [], "holdout": []}))

    _make_merge_base_dataset(root_path, "base_dataset")

    sys.argv = [
        "run_batch_injection.py",
        "--root-pth",
        str(root_path),
        "--run-name",
        "test_run",
        "--new-batch-root",
        str(raw_batch),
        "--protected-subsets-json",
        str(protected_subsets),
        "--annotation-xml",
        str(annotation_xml),
        "--review-confirmed",
        "--review-note",
        "manual review done",
        "--converted-dataset-name",
        "batch12",
        "--preprocess-mode",
        "wb_alif",
        "--merge-base-dataset",
        "base_dataset",
        "--merge-result-dataset",
        "merged_dataset",
        "--val-ratio",
        "0.0",
        "--overwrite",
    ]

    MODULE.main()

    run_dir = root_path / "batch_injection_runs" / "test_run"
    manifest = json.loads((run_dir / "batch_injection_run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["steps"]["anti_leakage_split"]["status"] == "completed"
    assert manifest["steps"]["cvat_to_yolo_conversion"]["status"] == "completed"
    assert manifest["steps"]["border_filtering"]["status"] == "completed"
    assert manifest["steps"]["preprocessing"]["status"] == "completed"
    assert manifest["steps"]["dataset_merge"]["status"] == "completed"

    final_dataset = root_path / "yolo_dataset" / "merged_dataset"
    assert final_dataset.exists()
    assert (final_dataset / "merge_summary.json").exists()
    assert (root_path / "yolo_dataset" / "batch12" / "conversion_summary.json").exists()
    assert (root_path / "yolo_dataset" / "batch12_margin_10" / "filter_report.json").exists()
    assert (root_path / "yolo_dataset" / "batch12_margin_10_wb_alif" / "preprocess_summary.json").exists()
