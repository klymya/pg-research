#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

from alif_awb import white_balance_vectorized
from pipeline_dataset_utils import (
    copy_dataset_tree,
    refresh_project_files,
    remove_cache_files,
    write_json,
)


DEFAULT_BASE_PTH = Path("/home/klymy/data/cam4_revB/yolo_dataset")
DEFAULT_MAX_WORKERS = 8
VALID_MODES = {"wb_alif", "wb_alif_clahe"}


def build_output_dataset_name(base_dataset: str, mode: str) -> str:
    suffix = {
        "wb_alif": "_wb_alif",
        "wb_alif_clahe": "_wb_alif_clahe",
    }[mode]
    return f"{base_dataset}{suffix}"


def apply_clahe(image_rgb):
    lab = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def transform_image(image_bgr, mode: str):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = white_balance_vectorized(image_rgb)
    if mode == "wb_alif_clahe":
        result = apply_clahe(result)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def _process_one_image(args: tuple[Path, str]) -> dict[str, Any]:
    image_path, mode = args
    image = cv2.imread(str(image_path))
    if image is None:
        return {"image_path": str(image_path), "status": "read_error"}

    result = transform_image(image, mode)
    success = cv2.imwrite(str(image_path), result)
    return {"image_path": str(image_path), "status": "ok" if success else "write_error"}


def preprocess_dataset(
    base_dataset_path: Path,
    output_dataset_path: Path,
    mode: str,
    overwrite: bool,
    max_workers: int,
    fail_on_issues: bool = False,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {sorted(VALID_MODES)}")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    copy_dataset_tree(base_dataset_path, output_dataset_path, overwrite=overwrite)
    removed_cache_files = remove_cache_files(output_dataset_path)
    written_project_files = refresh_project_files(output_dataset_path)

    image_paths = sorted((output_dataset_path / "images").rglob("*.jpg"))
    tasks = [(image_path, mode) for image_path in image_paths]
    if max_workers == 1:
        results = [_process_one_image(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_process_one_image, tasks))

    status_counts: dict[str, int] = {}
    for item in results:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    issues = [item for item in results if item["status"] != "ok"]

    if fail_on_issues and issues:
        raise ValueError(f"Preprocessing issues detected: {status_counts}")

    summary = {
        "config": {
            "base_dataset_path": str(base_dataset_path),
            "output_dataset_path": str(output_dataset_path),
            "mode": mode,
            "max_workers": max_workers,
            "fail_on_issues": fail_on_issues,
        },
        "counts": {
            "images_total": len(image_paths),
            "images_processed": status_counts.get("ok", 0),
            "issue_count": len(issues),
            "removed_cache_files": removed_cache_files,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "project_files_written": written_project_files,
        "issues_sample": issues[:100],
    }
    write_json(output_dataset_path / "preprocess_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a YOLO dataset and apply ALIF white balance or ALIF white balance + CLAHE."
    )
    parser.add_argument(
        "--base-pth",
        default=str(DEFAULT_BASE_PTH),
        help=f"Base path containing YOLO datasets. Default: {DEFAULT_BASE_PTH}",
    )
    parser.add_argument("--base-dataset", required=True, help="Source dataset name under --base-pth.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(VALID_MODES),
        help="Preprocessing mode to apply to copied dataset images.",
    )
    parser.add_argument(
        "--output-dataset",
        help="Optional output dataset name. Defaults to <base-dataset>_wb_alif or <base-dataset>_wb_alif_clahe.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output dataset if it exists.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum worker threads. Default: {DEFAULT_MAX_WORKERS}",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Fail if any image cannot be read or written during preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_pth)
    output_dataset = args.output_dataset or build_output_dataset_name(args.base_dataset, args.mode)
    summary = preprocess_dataset(
        base_dataset_path=base_path / args.base_dataset,
        output_dataset_path=base_path / output_dataset,
        mode=args.mode,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
        fail_on_issues=args.fail_on_issues,
    )
    print(
        f"Preprocessed {summary['counts']['images_processed']} / {summary['counts']['images_total']} images "
        f"into {output_dataset}"
    )


if __name__ == "__main__":
    main()
