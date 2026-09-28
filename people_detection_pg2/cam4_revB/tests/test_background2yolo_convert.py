from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import background2yolo_convert as b2y  # noqa: E402


def _make_image(path: Path, size=(64, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)


def test_converts_backgrounds_with_empty_labels_and_exclusion(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _make_image(root / "bg_v2" / "a.tif")
    _make_image(root / "bg_v2" / "b.tif", size=(32, 24))
    _make_image(root / "bg_v2" / "dup.tif")
    (root / "bg_v2" / ".DS_Store").write_text("junk")
    _make_image(root / "bg_old" / "dup.tif")

    out = root / "yolo_dataset_v2" / "bg"
    summary = b2y.run_conversion(
        root_path=root,
        image_roots=[root / "bg_v2"],
        output_dataset_path=out,
        exclude_roots=[root / "bg_old"],
        resize=(64, 48),
        max_workers=1,
    )

    assert summary["counts"] == {"background_images_written": 2, "train": 2, "val": 0, "skipped_excluded": 1}
    images = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert images == ["bg_v2____a.jpg", "bg_v2____b.jpg"]
    for name in images:
        label = out / "labels" / "train" / name.replace(".jpg", ".txt")
        assert label.read_text() == ""
        with Image.open(out / "images" / "train" / name) as image:
            assert image.size == (64, 48)
    assert (out / "images" / "val").is_dir()
    assert (out / "project.yaml").exists()
    assert (out / "background_conversion_summary.json").exists()


def test_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _make_image(root / "bg" / "a.tif")
    out = root / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        b2y.run_conversion(root_path=root, image_roots=[root / "bg"], output_dataset_path=out, max_workers=1)


def test_image_root_outside_root_is_rejected(tmp_path: Path) -> None:
    _make_image(tmp_path / "elsewhere" / "a.tif")
    with pytest.raises(ValueError):
        b2y.run_conversion(
            root_path=tmp_path / "root",
            image_roots=[tmp_path / "elsewhere"],
            output_dataset_path=tmp_path / "out",
            max_workers=1,
        )


def test_val_split_is_seeded_and_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "root"
    for i in range(20):
        _make_image(root / "bg" / f"img_{i:02d}.tif", size=(8, 6))

    def val_names(out_name: str, seed: int) -> list[str]:
        out = root / out_name
        summary = b2y.run_conversion(
            root_path=root, image_roots=[root / "bg"], output_dataset_path=out,
            val_count=5, seed=seed, max_workers=1,
        )
        assert summary["counts"]["train"] == 15 and summary["counts"]["val"] == 5
        val = sorted(p.stem for p in (out / "images" / "val").iterdir())
        train = {p.stem for p in (out / "images" / "train").iterdir()}
        assert not train & set(val)
        assert val == sorted(p.stem for p in (out / "labels" / "val").iterdir())
        return val

    first = val_names("a", seed=42)
    assert first == val_names("b", seed=42)
    assert first != val_names("c", seed=7)


def test_val_count_must_leave_train_images(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _make_image(root / "bg" / "a.tif")
    with pytest.raises(ValueError):
        b2y.run_conversion(root_path=root, image_roots=[root / "bg"], output_dataset_path=root / "o",
                           val_count=1, max_workers=1)
