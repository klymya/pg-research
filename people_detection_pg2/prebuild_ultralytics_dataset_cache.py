"""Prebuild Ultralytics dataset caches and lock them read-only.

This script walks YOLO dataset directories under a root folder, materializes the
Ultralytics ``labels/<split>.cache`` files once, and then marks those cache
files read-only so later training runs cannot race while rewriting them.

Example:
    python people_detection_pg2/prebuild_ultralytics_dataset_cache.py
    python people_detection_pg2/prebuild_ultralytics_dataset_cache.py --overwrite
"""

from __future__ import annotations

import argparse
import logging
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATASETS_ROOT = Path("/home/klymy/data/cam4_revB/yolo_dataset_v2")
PREFERRED_SPLIT_ORDER = ("train", "val", "holdout", "holdout_v2")
OWNER_READ_WRITE_MODE = 0o644
READ_ONLY_MODE = 0o444


class CachePrebuildError(Exception):
    """Raised when cache prebuild configuration or execution fails."""


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """One dataset split that should have a Ultralytics cache file."""

    dataset_name: str
    dataset_path: Path
    project_yaml_path: Path
    split_name: str
    images_dir: Path
    labels_dir: Path
    cache_path: Path


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Outcome for a single split cache build."""

    dataset_name: str
    split_name: str
    cache_path: Path
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prebuild Ultralytics labels/*.cache files for every dataset under "
            "a datasets root and mark them read-only."
        )
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help="Root folder that contains YOLO dataset directories with project.yaml files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild caches even if labels/<split>.cache already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned cache operations without executing them.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def ensure_ultralytics_imports() -> tuple[object, object]:
    project_root = Path(__file__).resolve().parent
    vendored_ultralytics_root = project_root / "ultralytics_yolov8"
    if not vendored_ultralytics_root.is_dir():
        raise CachePrebuildError(f"Vendored Ultralytics directory not found: {vendored_ultralytics_root}")

    sys.path.insert(0, str(vendored_ultralytics_root))
    try:
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.utils import yaml_load
    except ImportError as exc:
        raise CachePrebuildError(
            "Failed to import vendored Ultralytics components from "
            f"{vendored_ultralytics_root}. Make sure the runtime environment has "
            "OpenCV and the training dependencies installed."
        ) from exc
    return YOLODataset, yaml_load


def is_read_only(path: Path) -> bool:
    return (path.stat().st_mode & stat.S_IWUSR) == 0


def set_owner_writable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod((mode | stat.S_IWUSR) & ~stat.S_IWGRP & ~stat.S_IWOTH)


def set_read_only(path: Path) -> None:
    path.chmod(READ_ONLY_MODE)


def discover_dataset_paths(datasets_root: Path) -> list[Path]:
    if not datasets_root.is_dir():
        raise CachePrebuildError(f"Datasets root does not exist or is not a directory: {datasets_root}")

    if (datasets_root / "project.yaml").is_file():
        return [datasets_root]

    dataset_paths = sorted(
        path for path in datasets_root.iterdir() if path.is_dir() and (path / "project.yaml").is_file()
    )
    if not dataset_paths:
        raise CachePrebuildError(f"No dataset directories with project.yaml found under {datasets_root}")
    return dataset_paths


def ordered_split_names(split_names: Iterable[str]) -> list[str]:
    split_name_set = set(split_names)
    ordered = [name for name in PREFERRED_SPLIT_ORDER if name in split_name_set]
    ordered.extend(sorted(split_name_set - set(PREFERRED_SPLIT_ORDER)))
    return ordered


def discover_split_plans(dataset_path: Path) -> list[SplitPlan]:
    project_yaml_path = dataset_path / "project.yaml"
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"

    if not images_root.is_dir():
        raise CachePrebuildError(f"Dataset images directory not found: {images_root}")
    if not labels_root.is_dir():
        raise CachePrebuildError(f"Dataset labels directory not found: {labels_root}")

    available_split_names = []
    for images_dir in sorted(path for path in images_root.iterdir() if path.is_dir()):
        labels_dir = labels_root / images_dir.name
        if not labels_dir.is_dir():
            logging.warning(
                "Skipping dataset=%s split=%s because labels directory is missing: %s",
                dataset_path.name,
                images_dir.name,
                labels_dir,
            )
            continue
        available_split_names.append(images_dir.name)

    split_plans = [
        SplitPlan(
            dataset_name=dataset_path.name,
            dataset_path=dataset_path,
            project_yaml_path=project_yaml_path,
            split_name=split_name,
            images_dir=images_root / split_name,
            labels_dir=labels_root / split_name,
            cache_path=labels_root / f"{split_name}.cache",
        )
        for split_name in ordered_split_names(available_split_names)
    ]
    if not split_plans:
        raise CachePrebuildError(f"Dataset {dataset_path} does not contain any valid images/<split> + labels/<split> pairs")
    return split_plans


def remove_existing_cache_files(cache_path: Path) -> None:
    npy_path = cache_path.with_suffix(cache_path.suffix + ".npy")
    for path in (cache_path, npy_path):
        if not path.exists():
            continue
        if is_read_only(path):
            set_owner_writable(path)
        path.unlink()


def build_cache_for_split(
    plan: SplitPlan,
    *,
    overwrite: bool,
    dry_run: bool,
    yolo_dataset_cls: object,
    yaml_load: object,
) -> SplitResult:
    if dry_run:
        message = (
            f"would build {plan.cache_path} from images={plan.images_dir} "
            f"project={plan.project_yaml_path}"
        )
        return SplitResult(plan.dataset_name, plan.split_name, plan.cache_path, "dry_run", message)
    if yolo_dataset_cls is None or yaml_load is None:
        raise CachePrebuildError("Ultralytics cache builder dependencies are not initialized.")

    if overwrite:
        remove_existing_cache_files(plan.cache_path)

    project_data = yaml_load(plan.project_yaml_path)
    project_data["path"] = str(plan.dataset_path)

    yolo_dataset_cls(
        img_path=str(plan.images_dir),
        data=project_data,
        task="pose",
        augment=False,
        cache=False,
        rect=False,
        batch_size=16,
        prefix=f"{plan.dataset_name}:{plan.split_name}: ",
    )

    if not plan.cache_path.is_file():
        raise CachePrebuildError(
            f"Ultralytics did not create the expected cache file for "
            f"dataset={plan.dataset_name} split={plan.split_name}: {plan.cache_path}"
        )

    set_read_only(plan.cache_path)
    return SplitResult(plan.dataset_name, plan.split_name, plan.cache_path, "built", "cache built and locked read-only")


def main() -> int:
    configure_logging()
    args = parse_args()
    yolo_dataset_cls = None
    yaml_load = None
    if not args.dry_run:
        yolo_dataset_cls, yaml_load = ensure_ultralytics_imports()

    dataset_paths = discover_dataset_paths(args.datasets_root.resolve())
    logging.info("discovered %d datasets under %s", len(dataset_paths), args.datasets_root.resolve())
    logging.info(
        "cache files will be marked read-only after build; datasets must remain unchanged afterwards "
        "or Ultralytics may try to refresh caches on future runs"
    )

    results: list[SplitResult] = []
    failures: list[str] = []

    for dataset_path in dataset_paths:
        logging.info("processing dataset=%s", dataset_path.name)
        try:
            split_plans = discover_split_plans(dataset_path)
        except Exception as exc:
            failures.append(f"dataset={dataset_path.name}: {exc}")
            logging.error("failed to plan dataset=%s: %s", dataset_path.name, exc)
            continue

        for plan in split_plans:
            logging.info(
                "dataset=%s split=%s cache=%s",
                plan.dataset_name,
                plan.split_name,
                plan.cache_path,
            )
            try:
                result = build_cache_for_split(
                    plan,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    yolo_dataset_cls=yolo_dataset_cls,
                    yaml_load=yaml_load,
                )
                results.append(result)
                logging.info(
                    "dataset=%s split=%s status=%s message=%s",
                    result.dataset_name,
                    result.split_name,
                    result.status,
                    result.message,
                )
            except Exception as exc:
                failures.append(f"dataset={plan.dataset_name} split={plan.split_name}: {exc}")
                logging.error(
                    "failed dataset=%s split=%s: %s",
                    plan.dataset_name,
                    plan.split_name,
                    exc,
                )

    built_count = sum(result.status == "built" for result in results)
    dry_run_count = sum(result.status == "dry_run" for result in results)
    logging.info(
        "summary datasets=%d split_results=%d built=%d dry_run=%d failed=%d",
        len(dataset_paths),
        len(results),
        built_count,
        dry_run_count,
        len(failures),
    )

    if failures:
        for failure in failures:
            logging.error("failure %s", failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
