from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_LABEL_MAPPING = {0: "person"}


def write_project_file(
    file_path: Path,
    dataset_path: Path,
    label_mapping: dict[int, str] | None = None,
    val_folder: str = "images/val",
) -> None:
    mapping = label_mapping or DEFAULT_LABEL_MAPPING
    label_lines = "\n".join(f"  {class_id}: {label_name}" for class_id, label_name in mapping.items())
    file_path.write_text(
        f"path: {dataset_path}\n"
        "train: images/train\n"
        f"val: {val_folder}\n"
        "kpt_shape:\n"
        "- 1\n"
        "- 2\n"
        "names:\n"
        f"{label_lines}\n"
    )


def refresh_project_files(
    dataset_path: Path,
    label_mapping: dict[int, str] | None = None,
) -> list[str]:
    written: list[str] = []
    write_project_file(dataset_path / "project.yaml", dataset_path, label_mapping, val_folder="images/val")
    written.append("project.yaml")

    images_root = dataset_path / "images"
    if (images_root / "holdout").exists():
        write_project_file(
            dataset_path / "project_holdout.yaml",
            dataset_path,
            label_mapping,
            val_folder="images/holdout",
        )
        written.append("project_holdout.yaml")

    if (images_root / "holdout_v2").exists():
        write_project_file(
            dataset_path / "project_holdout_v2.yaml",
            dataset_path,
            label_mapping,
            val_folder="images/holdout_v2",
        )
        written.append("project_holdout_v2.yaml")

    return written


def remove_cache_files(dataset_path: Path) -> int:
    count = 0
    for cache_path in sorted(dataset_path.rglob("*.cache")):
        cache_path.unlink()
        count += 1
    return count


def copy_dataset_tree(src_path: Path, dst_path: Path, overwrite: bool) -> None:
    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src_path}")
    if dst_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst_path}. Pass overwrite=True to replace it.")
        shutil.rmtree(dst_path)
    shutil.copytree(src_path, dst_path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def canonicalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def count_files(directory: Path, pattern: str = "*") -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))
