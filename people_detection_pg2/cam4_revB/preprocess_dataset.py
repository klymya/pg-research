#!/usr/bin/env python3
"""Copy a YOLO dataset and apply the cam4_revB image preprocessing pipeline.

Pipeline (per image):
    raw -> ALIF white balance -> CLAHE -> optional sharpness augmentation

White balance and CLAHE are enabled by the ``--mode`` selection. Sharpness
augmentation samples a per-image delta from a uniform distribution and
applies the calibrated soften/sharpen operator; it is enabled by default.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import zlib
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

# The sharpness operators + calibration table live with the research that
# derived them, in the release_june_26 investigation. Make that folder
# importable so `sharpness_tools` resolves to the canonical source module.
_SHARPNESS_TOOLS_DIR = (
    Path(__file__).resolve().parent
    / ".." / "investigations" / "release_june_26" / "sharpness_research"
).resolve()
if str(_SHARPNESS_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARPNESS_TOOLS_DIR))
from sharpness_tools import adjust_sharpness  # noqa: E402


DEFAULT_BASE_PTH = Path("/home/klymy/data/cam4_revB/yolo_dataset")
DEFAULT_MAX_WORKERS = 8
VALID_MODES = {"wb_alif", "wb_alif_clahe", "clahe_only"}

DEFAULT_SHARPNESS_RANGE_PCT = 10.0
DEFAULT_AUGMENTATION_SEED = 42
MAX_SHARPNESS_RANGE_PCT = 15.0  # calibration table only covers +/- 15 %


def build_output_dataset_name(base_dataset: str, mode: str) -> str:
    suffix = {
        "wb_alif": "_wb_alif",
        "wb_alif_clahe": "_wb_alif_clahe",
        "clahe_only": "_clahe",
    }[mode]
    return f"{base_dataset}{suffix}"


def apply_clahe(image_rgb):
    lab = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def transform_image(image_bgr, mode: str, sharpness_delta_pct: float | None = None):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # White balance for wb_alif and wb_alif_clahe modes (clahe_only assumes
    # the input already went through white balance).
    if mode in {"wb_alif", "wb_alif_clahe"}:
        result = white_balance_vectorized(image_rgb)
    else:
        result = image_rgb

    # CLAHE for wb_alif_clahe and clahe_only modes.
    if mode in {"wb_alif_clahe", "clahe_only"}:
        result = apply_clahe(result)

    # Optional sharpness augmentation. `adjust_sharpness` selects the
    # sharpen or soften operator by sign and interpolates the calibrated
    # parameter for the requested delta.
    if sharpness_delta_pct is not None:
        result = adjust_sharpness(result, float(sharpness_delta_pct))

    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def _per_image_rng(seed_key: str, base_seed: int) -> random.Random:
    """Deterministic per-image RNG. CRC32 is stable across Python runs,
    unlike the built-in ``hash()`` under a randomized PYTHONHASHSEED.
    """
    path_seed = zlib.crc32(seed_key.encode("utf-8"))
    return random.Random((path_seed ^ int(base_seed)) & 0xFFFFFFFF)


def _process_one_image(args) -> dict[str, Any]:
    (
        image_path,
        mode,
        sharpness_augmentation,
        sharpness_range_pct,
        augmentation_seed,
        seed_key,
    ) = args
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "image_path": str(image_path),
            "status": "read_error",
            "sharpness_delta_pct": None,
        }

    if sharpness_augmentation:
        rng = _per_image_rng(seed_key, augmentation_seed)
        delta = rng.uniform(-sharpness_range_pct, sharpness_range_pct)
    else:
        delta = None

    result = transform_image(image, mode, sharpness_delta_pct=delta)
    success = cv2.imwrite(str(image_path), result)
    return {
        "image_path": str(image_path),
        "status": "ok" if success else "write_error",
        "sharpness_delta_pct": delta,
    }


def preprocess_dataset(
    base_dataset_path: Path,
    output_dataset_path: Path,
    mode: str,
    overwrite: bool,
    max_workers: int,
    fail_on_issues: bool = False,
    sharpness_augmentation: bool = True,
    sharpness_range_pct: float = DEFAULT_SHARPNESS_RANGE_PCT,
    augmentation_seed: int = DEFAULT_AUGMENTATION_SEED,
) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode {mode!r}. Expected one of {sorted(VALID_MODES)}")
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if sharpness_augmentation and not (0.0 <= sharpness_range_pct <= MAX_SHARPNESS_RANGE_PCT):
        raise ValueError(
            f"sharpness_range_pct must be in [0, {MAX_SHARPNESS_RANGE_PCT}]; "
            f"got {sharpness_range_pct}"
        )

    copy_dataset_tree(base_dataset_path, output_dataset_path, overwrite=overwrite)
    removed_cache_files = remove_cache_files(output_dataset_path)
    written_project_files = refresh_project_files(output_dataset_path)

    image_paths = sorted((output_dataset_path / "images").rglob("*.jpg"))
    # Seed keys are derived from the path *inside* the dataset so that the
    # augmentation for a given source image is stable across dst renames.
    tasks = [
        (
            image_path,
            mode,
            sharpness_augmentation,
            float(sharpness_range_pct),
            int(augmentation_seed),
            str(image_path.relative_to(output_dataset_path)),
        )
        for image_path in image_paths
    ]
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

    augmentation_stats: dict[str, Any] | None = None
    if sharpness_augmentation:
        deltas = [
            item["sharpness_delta_pct"]
            for item in results
            if item["status"] == "ok" and item.get("sharpness_delta_pct") is not None
        ]
        if deltas:
            augmentation_stats = {
                "count": len(deltas),
                "min_delta_pct": min(deltas),
                "max_delta_pct": max(deltas),
                "mean_delta_pct": statistics.fmean(deltas),
                "stdev_delta_pct": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
            }

    summary = {
        "config": {
            "base_dataset_path": str(base_dataset_path),
            "output_dataset_path": str(output_dataset_path),
            "mode": mode,
            "max_workers": max_workers,
            "fail_on_issues": fail_on_issues,
            "sharpness_augmentation": sharpness_augmentation,
            "sharpness_range_pct": (
                float(sharpness_range_pct) if sharpness_augmentation else None
            ),
            "augmentation_seed": (
                int(augmentation_seed) if sharpness_augmentation else None
            ),
        },
        "counts": {
            "images_total": len(image_paths),
            "images_processed": status_counts.get("ok", 0),
            "issue_count": len(issues),
            "removed_cache_files": removed_cache_files,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "augmentation_stats": augmentation_stats,
        "project_files_written": written_project_files,
        "issues_sample": issues[:100],
    }
    write_json(output_dataset_path / "preprocess_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a YOLO dataset and run cam4_revB preprocessing: ALIF white "
            "balance, optional CLAHE, and optional sharpness augmentation."
        )
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
    parser.add_argument(
        "--sharpness-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample a per-image uniform delta in [-range, +range] percent and "
            "apply the calibrated soften/sharpen operator after CLAHE. "
            "Use --no-sharpness-augmentation to disable. Default: enabled."
        ),
    )
    parser.add_argument(
        "--sharpness-range-pct",
        type=float,
        default=DEFAULT_SHARPNESS_RANGE_PCT,
        help=(
            f"Half-width of the uniform augmentation range in percent "
            f"(clamped to [0, {MAX_SHARPNESS_RANGE_PCT}]). "
            f"Default: {DEFAULT_SHARPNESS_RANGE_PCT}."
        ),
    )
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=DEFAULT_AUGMENTATION_SEED,
        help=(
            "Base seed mixed with each image's CRC32 path hash to derive a "
            f"per-image RNG (reproducible across runs). Default: {DEFAULT_AUGMENTATION_SEED}."
        ),
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
        sharpness_augmentation=args.sharpness_augmentation,
        sharpness_range_pct=args.sharpness_range_pct,
        augmentation_seed=args.augmentation_seed,
    )
    print(
        f"Preprocessed {summary['counts']['images_processed']} / {summary['counts']['images_total']} images "
        f"into {output_dataset}"
    )


if __name__ == "__main__":
    main()
