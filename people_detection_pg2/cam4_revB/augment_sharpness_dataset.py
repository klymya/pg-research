#!/usr/bin/env python3
"""Copy a YOLO dataset and apply the cam4_revB sharpness augmentation.

This is a standalone counterpart to ``preprocess_dataset.py``. It shares the
same random-augmentation policy (per-image uniform delta, calibrated
soften/sharpen operator, deterministic RNG) but skips the ALIF white
balance and CLAHE stages.

When to use it
--------------
Use this script when you already have a preprocessed YOLO dataset
(e.g. ``batchX_..._wb_alif_clahe``) and want an augmented variant of
just that dataset without re-running the pixel-level WB / CLAHE
preprocessing. Common cases:

- generate an augmented replica of an existing WB+CLAHE dataset for a
  training-robustness ablation,
- produce an augmented copy at a different range or seed for an
  augmentation-strength sweep,
- refresh a dataset's augmentation without touching the source dataset.

Pipeline (per image)
--------------------

    raw copy -> per-image delta ~ Uniform(-range, +range) percent
             -> adjust_sharpness(delta) using CALIBRATION_LAPLACIAN

The operator (sharpen or soften) and its parameter come from the
calibrated lookup table
``sharpness_tools.CALIBRATION_LAPLACIAN`` in
[sharpness_tools.py](/home/klymy/projects/researches/people_detection_pg2/investigations/release_june_26/sharpness_research/sharpness_tools.py),
which covers signed Laplacian-variance deltas in the range ``[-15 %, +15 %]``.

Determinism
-----------
The per-image RNG is seeded from a CRC32 of the image's path *inside*
the output dataset (e.g. ``images/train/foo.jpg``) XOR'd with
``--augmentation-seed`` (default 42). This makes the augmentation:

- deterministic across runs (same source + same seed -> same deltas),
- independent per image (no thread-pool ordering effects),
- stable across output-dataset renames (the seed key does not include
  the ``--output-dataset`` name).

Inputs
------
- ``--base-pth`` (default ``/home/klymy/data/cam4_revB/yolo_dataset``)
- ``--base-dataset``  name of the source dataset under ``--base-pth``
- ``--output-dataset`` name of the destination dataset under ``--base-pth``

Optional:
- ``--overwrite`` replace an existing output dataset
- ``--max-workers`` thread-pool size (default 8)
- ``--fail-on-issues`` raise if any image read/write fails
- ``--sharpness-range-pct`` half-width of the uniform range (default 10, clamped to [0, 15])
- ``--augmentation-seed`` base seed mixed into the per-image RNG (default 42)

Outputs
-------
The output dataset receives:

- a copy of every file from the source dataset,
- refreshed ``project.yaml`` / ``project_holdout.yaml`` / ``project_holdout_v2.yaml``,
- pixel-updated ``.jpg`` files under ``images/**``,
- ``sharpness_augmentation_summary.json`` describing:
    * the augmentation config used,
    * image counts and per-status counts,
    * ``augmentation_stats``: ``count``, ``min/max/mean/stdev`` of the sampled deltas,
    * which project files were refreshed,
    * a truncated ``issues_sample`` when any image failed.

Example
-------

Standard use — clone a preprocessed dataset and add the default ±10 %
augmentation:

.. code-block:: bash

   python /home/klymy/projects/researches/people_detection_pg2/cam4_revB/augment_sharpness_dataset.py \\
     --base-pth /home/klymy/data/cam4_revB/yolo_dataset \\
     --base-dataset batch1_10_bg_margin_10_wb_alif_clahe \\
     --output-dataset batch1_10_bg_margin_10_wb_alif_clahe_aug10 \\
     --overwrite

Narrower range for an ablation:

.. code-block:: bash

   python .../augment_sharpness_dataset.py \\
     --base-dataset batch1_10_bg_margin_10_wb_alif_clahe \\
     --output-dataset batch1_10_bg_margin_10_wb_alif_clahe_aug5 \\
     --sharpness-range-pct 5

Different seed for a repeat run:

.. code-block:: bash

   python .../augment_sharpness_dataset.py \\
     --base-dataset batch1_10_bg_margin_10_wb_alif_clahe \\
     --output-dataset batch1_10_bg_margin_10_wb_alif_clahe_aug10_seed2026 \\
     --augmentation-seed 2026
"""
from __future__ import annotations

import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2

from pipeline_dataset_utils import (
    copy_dataset_tree,
    refresh_project_files,
    remove_cache_files,
    write_json,
)

# Reuse the shared RNG helper and defaults from preprocess_dataset. Importing
# from that module also picks up its sys.path hook so `sharpness_tools`
# resolves to the release_june_26 investigation module.
from preprocess_dataset import (
    DEFAULT_AUGMENTATION_SEED,
    DEFAULT_SHARPNESS_RANGE_PCT,
    MAX_SHARPNESS_RANGE_PCT,
    _per_image_rng,
)
from sharpness_tools import adjust_sharpness  # noqa: E402  (sys.path set up above)


DEFAULT_BASE_PTH = Path("/home/klymy/data/cam4_revB/yolo_dataset")
DEFAULT_MAX_WORKERS = 8


def _augment_one_image(args) -> dict[str, Any]:
    image_path, sharpness_range_pct, augmentation_seed, seed_key = args
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return {
            "image_path": str(image_path),
            "status": "read_error",
            "sharpness_delta_pct": None,
        }

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rng = _per_image_rng(seed_key, augmentation_seed)
    delta = rng.uniform(-sharpness_range_pct, sharpness_range_pct)
    result_rgb = adjust_sharpness(image_rgb, float(delta))
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(image_path), result_bgr)

    return {
        "image_path": str(image_path),
        "status": "ok" if success else "write_error",
        "sharpness_delta_pct": delta,
    }


def augment_sharpness_dataset(
    base_dataset_path: Path,
    output_dataset_path: Path,
    overwrite: bool,
    max_workers: int,
    fail_on_issues: bool = False,
    sharpness_range_pct: float = DEFAULT_SHARPNESS_RANGE_PCT,
    augmentation_seed: int = DEFAULT_AUGMENTATION_SEED,
) -> dict[str, Any]:
    """Copy ``base_dataset_path`` to ``output_dataset_path`` and augment.

    The augmentation is identical in policy to the sharpness step in
    ``preprocess_dataset.preprocess_dataset``: per image, sample a
    ``Uniform(-sharpness_range_pct, +sharpness_range_pct)`` delta from a
    CRC32-seeded RNG and apply ``sharpness_tools.adjust_sharpness``.

    Returns a summary dict; also writes it to
    ``<output_dataset_path>/sharpness_augmentation_summary.json``.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if not (0.0 <= sharpness_range_pct <= MAX_SHARPNESS_RANGE_PCT):
        raise ValueError(
            f"sharpness_range_pct must be in [0, {MAX_SHARPNESS_RANGE_PCT}]; "
            f"got {sharpness_range_pct}"
        )

    copy_dataset_tree(base_dataset_path, output_dataset_path, overwrite=overwrite)
    removed_cache_files = remove_cache_files(output_dataset_path)
    written_project_files = refresh_project_files(output_dataset_path)

    image_paths = sorted((output_dataset_path / "images").rglob("*.jpg"))
    tasks = [
        (
            image_path,
            float(sharpness_range_pct),
            int(augmentation_seed),
            str(image_path.relative_to(output_dataset_path)),
        )
        for image_path in image_paths
    ]
    if max_workers == 1:
        results = [_augment_one_image(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_augment_one_image, tasks))

    status_counts: dict[str, int] = {}
    for item in results:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    issues = [item for item in results if item["status"] != "ok"]

    if fail_on_issues and issues:
        raise ValueError(f"Augmentation issues detected: {status_counts}")

    deltas = [
        item["sharpness_delta_pct"]
        for item in results
        if item["status"] == "ok" and item.get("sharpness_delta_pct") is not None
    ]
    augmentation_stats: dict[str, Any] | None = None
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
            "max_workers": max_workers,
            "fail_on_issues": fail_on_issues,
            "sharpness_range_pct": float(sharpness_range_pct),
            "augmentation_seed": int(augmentation_seed),
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
    write_json(output_dataset_path / "sharpness_augmentation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing YOLO dataset and apply the cam4_revB sharpness "
            "augmentation (uniform delta soften/sharpen) to the copy."
        )
    )
    parser.add_argument(
        "--base-pth",
        default=str(DEFAULT_BASE_PTH),
        help=f"Base path containing YOLO datasets. Default: {DEFAULT_BASE_PTH}",
    )
    parser.add_argument(
        "--base-dataset",
        required=True,
        help="Source dataset name under --base-pth (must exist).",
    )
    parser.add_argument(
        "--output-dataset",
        required=True,
        help="Destination dataset name under --base-pth.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination dataset if it already exists.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum worker threads. Default: {DEFAULT_MAX_WORKERS}",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Fail if any image cannot be read or written during augmentation.",
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
            f"per-image RNG (reproducible across runs). "
            f"Default: {DEFAULT_AUGMENTATION_SEED}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_pth)
    summary = augment_sharpness_dataset(
        base_dataset_path=base_path / args.base_dataset,
        output_dataset_path=base_path / args.output_dataset,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
        fail_on_issues=args.fail_on_issues,
        sharpness_range_pct=args.sharpness_range_pct,
        augmentation_seed=args.augmentation_seed,
    )
    print(
        f"Augmented {summary['counts']['images_processed']} / {summary['counts']['images_total']} images "
        f"into {args.output_dataset}"
    )


if __name__ == "__main__":
    main()
