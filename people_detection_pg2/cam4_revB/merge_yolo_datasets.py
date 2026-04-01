#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path
from typing import Any

from pipeline_dataset_utils import (
    copy_dataset_tree,
    refresh_project_files,
    remove_cache_files,
    write_json,
)


DEFAULT_BASE_PTH = Path("/home/klymy/data/cam4_revB/yolo_dataset")
TRAIN_VAL_SUBFOLDERS = [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val",
]


def copy_file_with_duplicate_policy(src_path: Path, dst_path: Path) -> str:
    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return "copied"

    if filecmp.cmp(src_path, dst_path, shallow=False):
        return "already_present"

    raise ValueError(f"Duplicate destination with different content: {dst_path}")


def copy_subfolder_contents(src_dir: Path, dst_dir: Path) -> dict[str, int]:
    counts = {"copied": 0, "already_present": 0}
    if not src_dir.exists():
        return counts

    for src_path in sorted(path for path in src_dir.iterdir() if path.is_file()):
        result = copy_file_with_duplicate_policy(src_path, dst_dir / src_path.name)
        counts[result] += 1
    return counts


def merge_yolo_datasets(
    base_dataset_path: Path,
    update_dataset_path: Path,
    result_dataset_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    copy_dataset_tree(base_dataset_path, result_dataset_path, overwrite=overwrite)
    removed_cache_files = remove_cache_files(result_dataset_path)

    train_val_results: dict[str, dict[str, int]] = {}
    for subfolder in TRAIN_VAL_SUBFOLDERS:
        src_dir = update_dataset_path / subfolder
        dst_dir = result_dataset_path / subfolder
        train_val_results[subfolder] = copy_subfolder_contents(src_dir, dst_dir)

    holdout_v2_results: dict[str, dict[str, int]] = {}
    for top_level in ("images", "labels"):
        src_holdout_dir = result_dataset_path / top_level / "holdout"
        dst_holdout_v2_dir = result_dataset_path / top_level / "holdout_v2"
        dst_holdout_v2_dir.mkdir(parents=True, exist_ok=True)

        existing_counts = copy_subfolder_contents(src_holdout_dir, dst_holdout_v2_dir)
        update_counts = copy_subfolder_contents(
            update_dataset_path / top_level / "holdout",
            dst_holdout_v2_dir,
        )
        holdout_v2_results[f"{top_level}/holdout_to_holdout_v2"] = existing_counts
        holdout_v2_results[f"{top_level}/update_holdout_to_holdout_v2"] = update_counts

    written_project_files = refresh_project_files(result_dataset_path)
    summary = {
        "config": {
            "base_dataset_path": str(base_dataset_path),
            "update_dataset_path": str(update_dataset_path),
            "result_dataset_path": str(result_dataset_path),
        },
        "counts": {
            "removed_cache_files": removed_cache_files,
        },
        "train_val_merge": train_val_results,
        "holdout_v2_merge": holdout_v2_results,
        "project_files_written": written_project_files,
    }
    write_json(result_dataset_path / "merge_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a new YOLO dataset into a base dataset while preserving current holdout_v2 behavior."
    )
    parser.add_argument(
        "--base-pth",
        default=str(DEFAULT_BASE_PTH),
        help=f"Base path containing YOLO datasets. Default: {DEFAULT_BASE_PTH}",
    )
    parser.add_argument("--base-dataset", required=True, help="Base dataset name under --base-pth.")
    parser.add_argument("--update-dataset", required=True, help="Update dataset name under --base-pth.")
    parser.add_argument("--result-dataset", required=True, help="Result dataset name under --base-pth.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the result dataset if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_pth)
    summary = merge_yolo_datasets(
        base_dataset_path=base_path / args.base_dataset,
        update_dataset_path=base_path / args.update_dataset,
        result_dataset_path=base_path / args.result_dataset,
        overwrite=args.overwrite,
    )
    print(f"Merged dataset written to {summary['config']['result_dataset_path']}")


if __name__ == "__main__":
    main()
