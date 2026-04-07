#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

import antileakage_split
import cvat2yolo_convert
import filter_small_bboxes
import merge_yolo_datasets
from pipeline_dataset_utils import copy_dataset_tree, refresh_project_files, remove_cache_files, write_json


LOGGER = logging.getLogger("regenerate_yolo_datasets_v2")

DEFAULT_ROOT_PTH = Path("/home/klymy/data/cam4_revB")
DEFAULT_YOLO_OUTPUT_DIRNAME = "yolo_dataset_v2"
DEFAULT_SUBSETS_OUTPUT_DIRNAME = "subsets_v2"
DEFAULT_HOLDOUT_JSON_NAME = "train_val_holdout_subsets_batches_1_3.json"
DEFAULT_BACKGROUND_DIRNAME = "4.2_background"
DEFAULT_OUTPUT_IMAGE_EXT = "jpg"
DEFAULT_MAX_WORKERS = 8
DEFAULT_VAL_RATIO = 0.10
DEFAULT_SEED = 42
DEFAULT_MARGIN = 0.10
DEFAULT_MIN_INSIDE_RATIO = 0.01

EXPECTED_SUBSET_KEYS = ("train", "val", "holdout")


@dataclass(frozen=True)
class StageSpec:
    stage_name: str
    batch_ids: tuple[int, ...]
    dataset_name: str
    subset_json_name: str
    raw_roots: tuple[Path, ...]
    annotation_xmls: tuple[Path, ...]
    historical_subset_json: Path | None
    use_fixed_base_holdout: bool = False
    reuse_historical_subsets: bool = False


@dataclass(frozen=True)
class DatasetBuildSpec:
    dataset_name: str
    stage_names: tuple[str, ...]
    include_background: bool


@dataclass(frozen=True)
class StagePlan:
    spec: StageSpec
    subsets: dict[str, list[str]]
    validation: dict[str, Any]
    historical_subset_match: bool | None


@dataclass(frozen=True)
class DatasetPlan:
    spec: DatasetBuildSpec
    included_batches: tuple[int, ...]
    train_sources: tuple[str, ...]
    val_sources: tuple[str, ...]
    holdout_sources: tuple[str, ...]
    holdout_v2_sources: tuple[str, ...]
    background_sources: tuple[str, ...]


@dataclass(frozen=True)
class DatasetArtifactNames:
    raw: str
    margin: str
    wb: str
    clahe: str


class RegenerationValidationError(ValueError):
    pass


def build_preprocess_output_name(base_dataset: str, mode: str) -> str:
    suffix = {
        "wb_alif": "_wb_alif",
        "wb_alif_clahe": "_wb_alif_clahe",
    }[mode]
    return f"{base_dataset}{suffix}"


def import_preprocess_dataset_module():
    try:
        import preprocess_dataset as module
    except ModuleNotFoundError as exc:
        if exc.name == "cv2":
            raise ModuleNotFoundError(
                "preprocess_dataset requires OpenCV (`cv2`). Dry-run mode works without it, "
                "but WB/CLAHE execution requires the dependency to be installed."
            ) from exc
        raise
    return module


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate cam4_revB cumulative YOLO datasets under yolo_dataset_v2 using the "
            "current anti-leakage split, CVAT-to-YOLO conversion, margin filtering, and "
            "image preprocessing utilities."
        )
    )
    parser.add_argument("--root-pth", default=str(DEFAULT_ROOT_PTH), help=f"Default: {DEFAULT_ROOT_PTH}")
    parser.add_argument(
        "--yolo-output-dirname",
        default=DEFAULT_YOLO_OUTPUT_DIRNAME,
        help=f"Output directory name under root. Default: {DEFAULT_YOLO_OUTPUT_DIRNAME}",
    )
    parser.add_argument(
        "--subsets-output-dirname",
        default=DEFAULT_SUBSETS_OUTPUT_DIRNAME,
        help=f"Subset output directory name under root. Default: {DEFAULT_SUBSETS_OUTPUT_DIRNAME}",
    )
    parser.add_argument(
        "--holdout-json-name",
        default=DEFAULT_HOLDOUT_JSON_NAME,
        help=f"Holdout definition JSON under root. Default: {DEFAULT_HOLDOUT_JSON_NAME}",
    )
    parser.add_argument(
        "--background-dirname",
        default=DEFAULT_BACKGROUND_DIRNAME,
        help=f"Background image directory under root. Default: {DEFAULT_BACKGROUND_DIRNAME}",
    )
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help=f"Default: {DEFAULT_VAL_RATIO}")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Default: {DEFAULT_SEED}")
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help=(
            "Margin filtering threshold in the current production implementation's normalized units. "
            f"Default: {DEFAULT_MARGIN}"
        ),
    )
    parser.add_argument(
        "--min-inside-ratio",
        type=float,
        default=DEFAULT_MIN_INSIDE_RATIO,
        help=f"Default: {DEFAULT_MIN_INSIDE_RATIO}",
    )
    parser.add_argument(
        "--output-image-ext",
        default=DEFAULT_OUTPUT_IMAGE_EXT,
        help=f"Converted image extension. Default: {DEFAULT_OUTPUT_IMAGE_EXT}",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum worker threads. Default: {DEFAULT_MAX_WORKERS}",
    )
    parser.add_argument(
        "--allow-annotation-issues",
        action="store_true",
        help="Allow converter-reported image/annotation issues instead of failing the run.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the validation and planning flow without writing outputs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def canonicalize_paths(paths: list[str] | tuple[str, ...]) -> list[str]:
    return sorted(str(Path(path).expanduser().resolve(strict=False)) for path in paths)


def normalize_subsets(subsets: dict[str, list[str]]) -> dict[str, list[str]]:
    if set(subsets) != set(EXPECTED_SUBSET_KEYS):
        raise RegenerationValidationError(
            f"Expected subset keys {EXPECTED_SUBSET_KEYS}, got {sorted(subsets)}"
        )

    normalized = {split: canonicalize_paths(list(subsets[split])) for split in EXPECTED_SUBSET_KEYS}
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for split, paths in normalized.items():
        for path in paths:
            previous_split = seen.get(path)
            if previous_split is not None:
                duplicates.append({"path": path, "first_split": previous_split, "second_split": split})
            else:
                seen[path] = split
    if duplicates:
        raise RegenerationValidationError(
            f"Subset JSON contains duplicate source paths across splits: {duplicates[:10]}"
        )
    return normalized


def load_subsets_json(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Subset JSON does not exist: {path}")
    return normalize_subsets(antileakage_split.load_subset_json(str(path)))


def stage_label(batch_ids: tuple[int, ...]) -> str:
    if len(batch_ids) == 1:
        return str(batch_ids[0])
    return f"{batch_ids[0]}_{batch_ids[-1]}"


def build_stage_specs(root_path: Path) -> list[StageSpec]:
    specs: list[StageSpec] = []
    definitions: tuple[tuple[str, tuple[int, ...], bool], ...] = (
        ("batch1_3", (1, 2, 3), True),
        ("batch4_5", (4, 5), False),
        ("batch6_7", (6, 7), False),
        ("batch8_10", (8, 9, 10), False),
        ("batch11", (11,), False),
        ("batch12", (12,), False),
    )
    for dataset_name, batch_ids, reuse_historical in definitions:
        label = stage_label(batch_ids)
        specs.append(
            StageSpec(
                stage_name=dataset_name,
                batch_ids=batch_ids,
                dataset_name=dataset_name,
                subset_json_name=f"train_val_holdout_subsets_batches_{label}.json",
                raw_roots=tuple(root_path / f"4.2_batch{batch_id}" for batch_id in batch_ids),
                annotation_xmls=tuple(
                    root_path / f"batch{batch_id}_corrected" / f"batch{batch_id}_corrected.xml"
                    for batch_id in batch_ids
                ),
                historical_subset_json=root_path / f"train_val_holdout_subsets_batches_{label}.json",
                use_fixed_base_holdout=dataset_name == "batch1_3",
                reuse_historical_subsets=reuse_historical and dataset_name != "batch1_3",
            )
        )
    return specs


def build_dataset_specs() -> list[DatasetBuildSpec]:
    return [
        DatasetBuildSpec(dataset_name="batch1_3", stage_names=("batch1_3",), include_background=False),
        DatasetBuildSpec(dataset_name="batch1_3_bg", stage_names=("batch1_3",), include_background=True),
        DatasetBuildSpec(dataset_name="batch1_5_bg", stage_names=("batch1_3", "batch4_5"), include_background=True),
        DatasetBuildSpec(dataset_name="batch1_7_bg", stage_names=("batch1_3", "batch4_5", "batch6_7"), include_background=True),
        DatasetBuildSpec(
            dataset_name="batch1_10_bg",
            stage_names=("batch1_3", "batch4_5", "batch6_7", "batch8_10"),
            include_background=True,
        ),
        DatasetBuildSpec(
            dataset_name="batch1_11_bg",
            stage_names=("batch1_3", "batch4_5", "batch6_7", "batch8_10", "batch11"),
            include_background=True,
        ),
        DatasetBuildSpec(
            dataset_name="batch1_12_bg",
            stage_names=("batch1_3", "batch4_5", "batch6_7", "batch8_10", "batch11", "batch12"),
            include_background=True,
        ),
    ]


def build_artifact_names(dataset_name: str, margin: float) -> DatasetArtifactNames:
    margin_name = filter_small_bboxes.build_output_dataset_name(dataset_name, margin)
    return DatasetArtifactNames(
        raw=dataset_name,
        margin=margin_name,
        wb=build_preprocess_output_name(margin_name, "wb_alif"),
        clahe=build_preprocess_output_name(margin_name, "wb_alif_clahe"),
    )


def ensure_inputs_exist(stage_specs: list[StageSpec], holdout_json_path: Path, background_dir: Path) -> None:
    if not holdout_json_path.exists():
        raise FileNotFoundError(f"Original holdout JSON does not exist: {holdout_json_path}")
    if not background_dir.exists():
        raise FileNotFoundError(f"Background directory does not exist: {background_dir}")

    for spec in stage_specs:
        for raw_root in spec.raw_roots:
            if not raw_root.exists():
                raise FileNotFoundError(f"Missing raw batch directory for {spec.stage_name}: {raw_root}")
        for annotation_xml in spec.annotation_xmls:
            if not annotation_xml.exists():
                raise FileNotFoundError(f"Missing corrected XML for {spec.stage_name}: {annotation_xml}")
        if spec.historical_subset_json is not None and spec.reuse_historical_subsets and not spec.historical_subset_json.exists():
            raise FileNotFoundError(
                f"Historical subsets JSON is required for {spec.stage_name}: {spec.historical_subset_json}"
            )


def discover_raw_batch_paths(raw_roots: tuple[Path, ...]) -> list[str]:
    paths: list[str] = []
    for raw_root in raw_roots:
        tif_paths = sorted(raw_root.glob("*.tif"))
        if not tif_paths:
            raise RegenerationValidationError(f"No .tif files found under batch root {raw_root}")
        paths.extend(str(path.resolve(strict=False)) for path in tif_paths)
    return sorted(paths)


def validate_stage_source_membership(spec: StageSpec, subsets: dict[str, list[str]]) -> None:
    expected_paths = set(discover_raw_batch_paths(spec.raw_roots))
    actual_paths = set().union(*(set(subsets[split]) for split in EXPECTED_SUBSET_KEYS))
    unexpected = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if unexpected or missing:
        raise RegenerationValidationError(
            f"Stage {spec.stage_name} subset membership mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def compare_to_historical_subsets(
    historical_subset_json: Path | None,
    regenerated_subsets: dict[str, list[str]],
) -> bool | None:
    if historical_subset_json is None or not historical_subset_json.exists():
        return None
    historical = load_subsets_json(historical_subset_json)
    return historical == regenerated_subsets


def validate_stage_subsets(
    subsets: dict[str, list[str]],
    protected_holdout_groups: set[str],
    target_val_ratio: float,
    allow_malformed_paths: bool,
    enforce_holdout_group_isolation: bool,
    enforce_protected_group_exclusion: bool,
) -> dict[str, Any]:
    split_to_paths = {split: sorted(subsets[split]) for split in EXPECTED_SUBSET_KEYS}
    split_to_groups: dict[str, set[str]] = {}
    malformed_paths: list[dict[str, str]] = []

    for split, paths in split_to_paths.items():
        grouped, malformed = antileakage_split.collect_group_ids(paths)
        split_to_groups[split] = set(grouped)
        malformed_paths.extend({"split": split, **item} for item in malformed)

    duplicate_paths = (
        (set(split_to_paths["train"]) & set(split_to_paths["val"]))
        | (set(split_to_paths["train"]) & set(split_to_paths["holdout"]))
        | (set(split_to_paths["val"]) & set(split_to_paths["holdout"]))
    )
    train_val_group_overlap = split_to_groups["train"] & split_to_groups["val"]
    train_holdout_group_overlap = split_to_groups["train"] & split_to_groups["holdout"]
    val_holdout_group_overlap = split_to_groups["val"] & split_to_groups["holdout"]
    leaked_protected_groups = (
        (split_to_groups["train"] & protected_holdout_groups)
        | (split_to_groups["val"] & protected_holdout_groups)
    )

    working_frame_count = len(split_to_paths["train"]) + len(split_to_paths["val"])
    actual_val_ratio = len(split_to_paths["val"]) / working_frame_count if working_frame_count else 0.0
    is_valid = not any(
        [
            duplicate_paths,
            train_val_group_overlap,
            train_holdout_group_overlap and enforce_holdout_group_isolation,
            val_holdout_group_overlap and enforce_holdout_group_isolation,
            leaked_protected_groups and enforce_protected_group_exclusion,
            malformed_paths and not allow_malformed_paths,
        ]
    )

    return {
        "is_valid": is_valid,
        "counts": {split: len(paths) for split, paths in split_to_paths.items()},
        "group_counts": {split: len(groups) for split, groups in split_to_groups.items()},
        "duplicate_path_count": len(duplicate_paths),
        "train_val_group_overlap_count": len(train_val_group_overlap),
        "train_holdout_group_overlap_count": len(train_holdout_group_overlap),
        "val_holdout_group_overlap_count": len(val_holdout_group_overlap),
        "leaked_protected_group_count": len(leaked_protected_groups),
        "train_val_group_overlap_sample": sorted(train_val_group_overlap)[:20],
        "leaked_protected_group_sample": sorted(leaked_protected_groups)[:20],
        "malformed_path_count": len(malformed_paths),
        "malformed_path_sample": malformed_paths[:20],
        "target_val_ratio": target_val_ratio,
        "actual_val_ratio": actual_val_ratio,
        "allow_malformed_paths": allow_malformed_paths,
        "enforce_holdout_group_isolation": enforce_holdout_group_isolation,
        "enforce_protected_group_exclusion": enforce_protected_group_exclusion,
    }


def split_with_fixed_holdout(
    all_paths: list[str],
    fixed_holdout_paths: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    all_path_set = set(all_paths)
    fixed_holdout_set = set(fixed_holdout_paths)
    missing_fixed_holdout = sorted(fixed_holdout_set - all_path_set)
    if missing_fixed_holdout:
        raise RegenerationValidationError(
            f"Fixed holdout paths are not present in the raw batch set: {missing_fixed_holdout[:10]}"
        )

    working_paths = sorted(all_path_set - fixed_holdout_set)
    working_groups, malformed_working = antileakage_split.collect_group_ids(working_paths)
    if malformed_working:
        raise RegenerationValidationError(
            f"Malformed working paths encountered while building fixed-holdout split: {malformed_working[:10]}"
        )

    group_ids = sorted(working_groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    target_val_frames = int(len(working_paths) * val_ratio)
    val_group_ids: list[str] = []
    val_frame_count = 0
    for group_id in group_ids:
        if val_frame_count >= target_val_frames:
            break
        val_group_ids.append(group_id)
        val_frame_count += len(working_groups[group_id])

    val_group_set = set(val_group_ids)
    train_paths: list[str] = []
    val_paths: list[str] = []
    for group_id, group_items in working_groups.items():
        if group_id in val_group_set:
            val_paths.extend(group_items)
        else:
            train_paths.extend(group_items)

    malformed_holdout: list[dict[str, str]] = []
    for holdout_path in sorted(fixed_holdout_paths):
        try:
            antileakage_split.extract_raw_scene_group_id(holdout_path)
        except antileakage_split.GroupIdParseError as exc:
            malformed_holdout.append({"path": holdout_path, "error": str(exc)})

    return (
        {
            "train": sorted(train_paths),
            "val": sorted(val_paths),
            "holdout": sorted(fixed_holdout_paths),
        },
        malformed_holdout,
    )


def build_stage_plan(
    spec: StageSpec,
    original_holdout_json: Path,
    protected_holdout_groups: set[str],
    val_ratio: float,
    seed: int,
) -> StagePlan:
    if spec.use_fixed_base_holdout:
        original_subsets = load_subsets_json(original_holdout_json)
        raw_paths = discover_raw_batch_paths(spec.raw_roots)
        split_data, malformed_paths = split_with_fixed_holdout(
            all_paths=raw_paths,
            fixed_holdout_paths=original_subsets["holdout"],
            val_ratio=val_ratio,
            seed=seed,
        )
        subsets = normalize_subsets(split_data)
    elif spec.reuse_historical_subsets:
        subsets = load_subsets_json(spec.historical_subset_json)
        malformed_paths = []
    else:
        raw_paths = discover_raw_batch_paths(spec.raw_roots)
        split_data, malformed_paths = antileakage_split.split_new_batch(
            new_paths=raw_paths,
            protected_holdout_groups=protected_holdout_groups,
            val_ratio=val_ratio,
            seed=seed,
        )
        subsets = normalize_subsets(split_data)

    validate_stage_source_membership(spec, subsets)
    validation = validate_stage_subsets(
        subsets=subsets,
        protected_holdout_groups=protected_holdout_groups,
        target_val_ratio=val_ratio,
        allow_malformed_paths=spec.use_fixed_base_holdout or spec.reuse_historical_subsets,
        enforce_holdout_group_isolation=not spec.use_fixed_base_holdout,
        enforce_protected_group_exclusion=not spec.use_fixed_base_holdout,
    )
    if not validation["is_valid"]:
        raise RegenerationValidationError(
            f"Stage {spec.stage_name} failed anti-leakage validation: {validation}"
        )

    historical_subset_match = compare_to_historical_subsets(spec.historical_subset_json, subsets)
    if spec.stage_name == "batch1_3":
        original_subsets = load_subsets_json(original_holdout_json)
        if subsets["holdout"] != original_subsets["holdout"]:
            raise RegenerationValidationError(
                "The base batch1_3 holdout does not exactly match the original holdout JSON."
            )

    return StagePlan(
        spec=spec,
        subsets=subsets,
        validation=validation,
        historical_subset_match=historical_subset_match,
    )


def discover_background_paths(background_dir: Path) -> list[str]:
    background_paths = sorted(str(path.resolve(strict=False)) for path in background_dir.glob("*.tif"))
    if not background_paths:
        raise RegenerationValidationError(f"No background .tif files found under {background_dir}")
    return background_paths


def flatten_stage_batches(stage_names: tuple[str, ...], stage_plans: dict[str, StagePlan]) -> tuple[int, ...]:
    batch_ids: list[int] = []
    for stage_name in stage_names:
        batch_ids.extend(stage_plans[stage_name].spec.batch_ids)
    return tuple(batch_ids)


def compose_dataset_plan(
    spec: DatasetBuildSpec,
    stage_plans: dict[str, StagePlan],
    background_paths: list[str],
) -> DatasetPlan:
    if "batch1_3" not in stage_plans:
        raise RegenerationValidationError("Stage plans must include batch1_3")

    base_holdout_sources = tuple(stage_plans["batch1_3"].subsets["holdout"])
    train_sources: set[str] = set()
    val_sources: set[str] = set()
    all_holdout_sources: set[str] = set()
    later_holdout_sources: set[str] = set()

    for stage_name in spec.stage_names:
        stage_plan = stage_plans[stage_name]
        train_sources.update(stage_plan.subsets["train"])
        val_sources.update(stage_plan.subsets["val"])
        all_holdout_sources.update(stage_plan.subsets["holdout"])
        if stage_name != "batch1_3":
            later_holdout_sources.update(stage_plan.subsets["holdout"])

    background_source_set = set(background_paths) if spec.include_background else set()
    if background_source_set & train_sources:
        raise RegenerationValidationError(
            f"Background samples overlap train sources for {spec.dataset_name}"
        )

    if train_sources & val_sources:
        raise RegenerationValidationError(f"Train/val overlap detected for {spec.dataset_name}")
    if train_sources & set(base_holdout_sources):
        raise RegenerationValidationError(f"Train/base-holdout overlap detected for {spec.dataset_name}")
    if val_sources & set(base_holdout_sources):
        raise RegenerationValidationError(f"Val/base-holdout overlap detected for {spec.dataset_name}")

    expected_annotated_sources = set()
    for stage_name in spec.stage_names:
        stage_plan = stage_plans[stage_name]
        expected_annotated_sources.update(stage_plan.subsets["train"])
        expected_annotated_sources.update(stage_plan.subsets["val"])
        expected_annotated_sources.update(stage_plan.subsets["holdout"])
    if expected_annotated_sources != train_sources | val_sources | all_holdout_sources:
        raise RegenerationValidationError(
            f"Annotated source membership mismatch for {spec.dataset_name}"
        )

    holdout_sources = base_holdout_sources
    holdout_v2_sources = tuple(sorted(set(base_holdout_sources) | later_holdout_sources)) if len(spec.stage_names) > 1 else ()

    return DatasetPlan(
        spec=spec,
        included_batches=flatten_stage_batches(spec.stage_names, stage_plans),
        train_sources=tuple(sorted(train_sources)),
        val_sources=tuple(sorted(val_sources)),
        holdout_sources=holdout_sources,
        holdout_v2_sources=holdout_v2_sources,
        background_sources=tuple(sorted(background_source_set)),
    )


def path_to_sample_id(root_path: Path, source_path: str) -> str:
    source = Path(source_path).resolve(strict=False)
    relative_key = cvat2yolo_convert.relative_key_from_path(root_path.resolve(strict=False), source)
    return cvat2yolo_convert.build_sample_id(relative_key)


def expected_sample_ids_for_sources(
    root_path: Path,
    *,
    train_sources: tuple[str, ...],
    val_sources: tuple[str, ...],
    holdout_sources: tuple[str, ...],
    holdout_v2_sources: tuple[str, ...] = (),
    background_sources: tuple[str, ...] = (),
) -> dict[str, set[str]]:
    return {
        **{
            "train": {path_to_sample_id(root_path, path) for path in (*train_sources, *background_sources)},
            "val": {path_to_sample_id(root_path, path) for path in val_sources},
            "holdout": {path_to_sample_id(root_path, path) for path in holdout_sources},
        },
        **(
            {"holdout_v2": {path_to_sample_id(root_path, path) for path in holdout_v2_sources}}
            if holdout_v2_sources
            else {}
        ),
    }


def expected_sample_ids_for_dataset_plan(root_path: Path, dataset_plan: DatasetPlan) -> dict[str, set[str]]:
    return expected_sample_ids_for_sources(
        root_path,
        train_sources=dataset_plan.train_sources,
        val_sources=dataset_plan.val_sources,
        holdout_sources=dataset_plan.holdout_sources,
        holdout_v2_sources=dataset_plan.holdout_v2_sources,
        background_sources=dataset_plan.background_sources,
    )


def expected_sample_ids_for_stage_plan(root_path: Path, stage_plan: StagePlan) -> dict[str, set[str]]:
    return expected_sample_ids_for_sources(
        root_path,
        train_sources=tuple(stage_plan.subsets["train"]),
        val_sources=tuple(stage_plan.subsets["val"]),
        holdout_sources=tuple(stage_plan.subsets["holdout"]),
    )


def current_dataset_counts(dataset_path: Path, image_ext: str = DEFAULT_OUTPUT_IMAGE_EXT) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "holdout", "holdout_v2"):
        image_dir = dataset_path / "images" / split
        if image_dir.exists():
            counts[split] = len(sorted(image_dir.glob(f"*.{image_ext}")))
    return counts


def validate_dataset(
    dataset_path: Path,
    expected_sample_ids: dict[str, set[str]],
    image_ext: str = DEFAULT_OUTPUT_IMAGE_EXT,
) -> dict[str, int]:
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"
    if not images_root.exists() or not labels_root.exists():
        raise RegenerationValidationError(f"Dataset is missing images/labels roots: {dataset_path}")

    for split, expected_ids in expected_sample_ids.items():
        image_dir = images_root / split
        label_dir = labels_root / split
        if not image_dir.exists():
            raise RegenerationValidationError(f"Missing image split directory {image_dir}")
        if not label_dir.exists():
            raise RegenerationValidationError(f"Missing label split directory {label_dir}")

        actual_image_ids = {path.stem for path in image_dir.glob(f"*.{image_ext}")}
        actual_label_ids = {path.stem for path in label_dir.glob("*.txt")}
        missing_images = sorted(expected_ids - actual_image_ids)
        unexpected_images = sorted(actual_image_ids - expected_ids)
        missing_labels = sorted(expected_ids - actual_label_ids)
        unexpected_labels = sorted(actual_label_ids - expected_ids)
        if missing_images or unexpected_images or missing_labels or unexpected_labels:
            raise RegenerationValidationError(
                f"Dataset mismatch for {dataset_path.name}/{split}: "
                f"missing_images={missing_images[:10]}, unexpected_images={unexpected_images[:10]}, "
                f"missing_labels={missing_labels[:10]}, unexpected_labels={unexpected_labels[:10]}"
            )

    unexpected_split_dirs = sorted(
        split_dir.name
        for split_dir in images_root.iterdir()
        if split_dir.is_dir() and split_dir.name not in expected_sample_ids
    )
    if unexpected_split_dirs:
        raise RegenerationValidationError(
            f"Unexpected image split directories in {dataset_path}: {unexpected_split_dirs}"
        )
    return current_dataset_counts(dataset_path, image_ext=image_ext)


def ensure_output_targets_available(
    yolo_output_dir: Path,
    subsets_output_dir: Path,
    stage_specs: list[StageSpec],
    dataset_specs: list[DatasetBuildSpec],
    margin: float,
    overwrite: bool,
) -> None:
    if overwrite:
        return

    targets: set[Path] = {subsets_output_dir / "regeneration_manifest.json"}
    for stage_spec in stage_specs:
        targets.add(subsets_output_dir / stage_spec.subset_json_name)
        targets.add(yolo_output_dir / stage_spec.dataset_name)
    for dataset_spec in dataset_specs:
        artifacts = build_artifact_names(dataset_spec.dataset_name, margin)
        targets.update(
            {
                yolo_output_dir / artifacts.raw,
                yolo_output_dir / artifacts.margin,
                yolo_output_dir / artifacts.wb,
                yolo_output_dir / artifacts.clahe,
            }
        )

    existing_targets = sorted(str(path) for path in targets if path.exists())
    if existing_targets:
        raise FileExistsError(
            "Refusing to overwrite existing v2 outputs without --overwrite: "
            f"{existing_targets[:20]}"
        )


def write_stage_subset_jsons(stage_plans: dict[str, StagePlan], subsets_output_dir: Path) -> dict[str, str]:
    written_paths: dict[str, str] = {}
    for stage_name, stage_plan in stage_plans.items():
        output_path = subsets_output_dir / stage_plan.spec.subset_json_name
        write_json(output_path, stage_plan.subsets)
        written_paths[stage_name] = str(output_path)
    return written_paths


def assert_converter_summary_is_clean(dataset_name: str, summary: dict[str, Any], allow_annotation_issues: bool) -> None:
    image_issue_total = int(summary["counts"]["image_issue_total"])
    invalid_object_total = int(summary["counts"]["invalid_object_total"])
    if allow_annotation_issues:
        return
    if image_issue_total or invalid_object_total:
        raise RegenerationValidationError(
            f"Conversion for {dataset_name} reported issues: "
            f"image_issue_total={image_issue_total}, invalid_object_total={invalid_object_total}"
        )


def build_dataset(
    *,
    build_kind: str,
    root_path: Path,
    output_dataset_path: Path,
    output_image_ext: str,
    stage_plan: StagePlan | None = None,
    subsets_output_dir: Path | None = None,
    base_dataset_path: Path | None = None,
    update_dataset_path: Path | None = None,
    background_paths: list[str] | None = None,
    max_workers: int,
    overwrite: bool,
    allow_annotation_issues: bool,
) -> dict[str, Any]:
    if build_kind == "stage_conversion":
        if stage_plan is None or subsets_output_dir is None:
            raise ValueError("stage_conversion requires stage_plan and subsets_output_dir")
        summary = cvat2yolo_convert.run_conversion(
            root_path=root_path,
            annotation_xmls=list(stage_plan.spec.annotation_xmls),
            image_roots=list(stage_plan.spec.raw_roots),
            subsets_json_path=subsets_output_dir / stage_plan.spec.subset_json_name,
            output_dataset_path=output_dataset_path,
            overwrite=overwrite,
            output_image_ext=output_image_ext,
            max_workers=max_workers,
            fail_on_issues=False,
        )
        assert_converter_summary_is_clean(stage_plan.spec.dataset_name, summary, allow_annotation_issues)
        return summary

    if build_kind == "background_append":
        if base_dataset_path is None or background_paths is None:
            raise ValueError("background_append requires base_dataset_path and background_paths")
        return append_background_dataset(
            root_path=root_path,
            src_dataset_path=base_dataset_path,
            dst_dataset_path=output_dataset_path,
            background_paths=background_paths,
            overwrite=overwrite,
            output_image_ext=output_image_ext,
        )

    if build_kind == "merge":
        if base_dataset_path is None or update_dataset_path is None:
            raise ValueError("merge requires base_dataset_path and update_dataset_path")
        return merge_yolo_datasets.merge_yolo_datasets(
            base_dataset_path=base_dataset_path,
            update_dataset_path=update_dataset_path,
            result_dataset_path=output_dataset_path,
            overwrite=overwrite,
        )

    raise ValueError(f"Unsupported build_kind {build_kind!r}")


def merge_dataset_without_holdout_v2(
    *,
    base_dataset_path: Path,
    update_dataset_path: Path,
    result_dataset_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    copy_dataset_tree(base_dataset_path, result_dataset_path, overwrite=overwrite)
    removed_cache_files = remove_cache_files(result_dataset_path)

    merge_results: dict[str, dict[str, int]] = {}
    for subfolder in (
        "images/train",
        "images/val",
        "images/holdout",
        "labels/train",
        "labels/val",
        "labels/holdout",
    ):
        merge_results[subfolder] = merge_yolo_datasets.copy_subfolder_contents(
            update_dataset_path / subfolder,
            result_dataset_path / subfolder,
        )

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
        "merge": merge_results,
        "project_files_written": written_project_files,
    }
    write_json(result_dataset_path / "merge_summary.json", summary)
    return summary


def write_background_sample(root_path: Path, background_path: str, dst_dataset_path: Path, output_image_ext: str) -> str:
    sample_id = path_to_sample_id(root_path, background_path)
    image_path = dst_dataset_path / "images" / "train" / f"{sample_id}.{output_image_ext}"
    label_path = dst_dataset_path / "labels" / "train" / f"{sample_id}.txt"
    if image_path.exists() or label_path.exists():
        raise RegenerationValidationError(f"Background sample already exists in destination dataset: {sample_id}")

    with Image.open(background_path) as image:
        save_kwargs: dict[str, Any] = {}
        if output_image_ext.lower() in {"jpg", "jpeg"}:
            save_kwargs["quality"] = 95
        image.convert("RGB").save(image_path, **save_kwargs)
    label_path.write_text("")
    return sample_id


def append_background_dataset(
    *,
    root_path: Path,
    src_dataset_path: Path,
    dst_dataset_path: Path,
    background_paths: list[str],
    overwrite: bool,
    output_image_ext: str,
) -> dict[str, Any]:
    if not src_dataset_path.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {src_dataset_path}")
    if dst_dataset_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst_dataset_path}")
        shutil.rmtree(dst_dataset_path)
    shutil.copytree(src_dataset_path, dst_dataset_path)

    for cache_path in sorted(dst_dataset_path.rglob("*.cache")):
        cache_path.unlink()

    sample_ids = [
        write_background_sample(root_path, background_path, dst_dataset_path, output_image_ext)
        for background_path in background_paths
    ]
    written_project_files = refresh_project_files(dst_dataset_path)
    summary = {
        "config": {
            "source_dataset_path": str(src_dataset_path),
            "output_dataset_path": str(dst_dataset_path),
            "background_source_count": len(background_paths),
            "output_image_ext": output_image_ext,
        },
        "counts": {
            "background_images_added": len(sample_ids),
        },
        "sample_ids": sample_ids,
        "project_files_written": written_project_files,
    }
    write_json(dst_dataset_path / "background_append_summary.json", summary)
    return summary


def build_background_chunk_dataset(
    *,
    root_path: Path,
    output_dataset_path: Path,
    background_paths: list[str],
    overwrite: bool,
    output_image_ext: str,
) -> dict[str, Any]:
    if output_dataset_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {output_dataset_path}")
        shutil.rmtree(output_dataset_path)

    for subfolder in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dataset_path / subfolder).mkdir(parents=True, exist_ok=True)

    sample_ids = [
        write_background_sample(root_path, background_path, output_dataset_path, output_image_ext)
        for background_path in background_paths
    ]
    written_project_files = refresh_project_files(output_dataset_path)
    summary = {
        "config": {
            "output_dataset_path": str(output_dataset_path),
            "background_source_count": len(background_paths),
            "output_image_ext": output_image_ext,
        },
        "counts": {
            "background_images_added": len(sample_ids),
        },
        "sample_ids": sample_ids,
        "project_files_written": written_project_files,
    }
    write_json(output_dataset_path / "background_chunk_summary.json", summary)
    return summary


def apply_margin_filter(
    *,
    src_dataset_path: Path,
    output_dataset_path: Path,
    margin: float,
    min_inside_ratio: float,
    max_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    return filter_small_bboxes.filter_dataset(
        src_dataset_path=src_dataset_path,
        dst_dataset_path=output_dataset_path,
        margin=margin,
        min_inside_ratio=min_inside_ratio,
        max_workers=max_workers,
        overwrite=overwrite,
    )


def apply_white_balance(
    *,
    src_dataset_path: Path,
    output_dataset_path: Path,
    max_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    preprocess_dataset_module = import_preprocess_dataset_module()
    return preprocess_dataset_module.preprocess_dataset(
        base_dataset_path=src_dataset_path,
        output_dataset_path=output_dataset_path,
        mode="wb_alif",
        overwrite=overwrite,
        max_workers=max_workers,
        fail_on_issues=True,
    )


def apply_clahe(
    *,
    src_dataset_path: Path,
    output_dataset_path: Path,
    max_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    preprocess_dataset_module = import_preprocess_dataset_module()
    return preprocess_dataset_module.preprocess_dataset(
        base_dataset_path=src_dataset_path,
        output_dataset_path=output_dataset_path,
        mode="wb_alif_clahe",
        overwrite=overwrite,
        max_workers=max_workers,
        fail_on_issues=True,
    )


def log_dataset_stage_summary(
    *,
    dataset_name: str,
    stage_name: str,
    included_batches: tuple[int, ...],
    dataset_path: Path,
    counts: dict[str, int],
) -> None:
    LOGGER.info(
        "dataset=%s stage=%s batches=%s train=%s val=%s holdout=%s holdout_v2=%s output=%s",
        dataset_name,
        stage_name,
        ",".join(str(batch_id) for batch_id in included_batches),
        counts.get("train", 0),
        counts.get("val", 0),
        counts.get("holdout", 0),
        counts.get("holdout_v2", 0),
        dataset_path,
    )


def write_dataset_regeneration_summary(
    *,
    dataset_path: Path,
    dataset_name: str,
    stage_name: str,
    dataset_plan: DatasetPlan,
    artifact_names: DatasetArtifactNames,
    counts: dict[str, int],
    source_summary: dict[str, Any] | None,
) -> None:
    payload = {
        "dataset_name": dataset_name,
        "stage_name": stage_name,
        "included_batches": list(dataset_plan.included_batches),
        "background_included": dataset_plan.spec.include_background,
        "artifact_names": {
            "raw": artifact_names.raw,
            "margin": artifact_names.margin,
            "wb": artifact_names.wb,
            "clahe": artifact_names.clahe,
        },
        "counts": counts,
        "expected_primary_sources": {
            "train": len(dataset_plan.train_sources) + len(dataset_plan.background_sources),
            "val": len(dataset_plan.val_sources),
            "holdout": len(dataset_plan.holdout_sources),
            "holdout_v2": len(dataset_plan.holdout_v2_sources),
        },
        "source_summary": source_summary,
    }
    write_json(dataset_path / "regeneration_summary.json", payload)


def build_manifest(
    root_path: Path,
    yolo_output_dir: Path,
    subsets_output_dir: Path,
    holdout_json_path: Path,
    background_dir: Path,
    val_ratio: float,
    seed: int,
    margin: float,
    min_inside_ratio: float,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "config": {
            "root_path": str(root_path),
            "yolo_output_dir": str(yolo_output_dir),
            "subsets_output_dir": str(subsets_output_dir),
            "holdout_json_path": str(holdout_json_path),
            "background_dir": str(background_dir),
            "val_ratio": val_ratio,
            "seed": seed,
            "margin": margin,
            "min_inside_ratio": min_inside_ratio,
            "dry_run": dry_run,
        },
        "status": "validating",
        "stage_plans": {},
        "dataset_plans": {},
        "written_subset_jsons": {},
        "materialized_datasets": {},
        "warnings": [],
    }


def cleanup_dataset_path(dataset_path: Path) -> None:
    if dataset_path.exists():
        shutil.rmtree(dataset_path)


def process_cumulative_dataset(
    *,
    root_path: Path,
    yolo_output_dir: Path,
    dataset_plan: DatasetPlan,
    artifact_names: DatasetArtifactNames,
    margin: float,
    min_inside_ratio: float,
    max_workers: int,
    overwrite: bool,
    output_image_ext: str,
    manifest: dict[str, Any],
    manifest_key: str,
) -> None:
    expected_ids = expected_sample_ids_for_dataset_plan(root_path, dataset_plan)

    raw_path = yolo_output_dir / artifact_names.raw
    raw_counts = validate_dataset(raw_path, expected_ids, image_ext=output_image_ext)
    log_dataset_stage_summary(
        dataset_name=dataset_plan.spec.dataset_name,
        stage_name="raw",
        included_batches=dataset_plan.included_batches,
        dataset_path=raw_path,
        counts=raw_counts,
    )
    manifest["materialized_datasets"][manifest_key]["raw"] = {
        "dataset_path": str(raw_path),
        "counts": raw_counts,
    }

    margin_path = yolo_output_dir / artifact_names.margin
    margin_summary = apply_margin_filter(
        src_dataset_path=raw_path,
        output_dataset_path=margin_path,
        margin=margin,
        min_inside_ratio=min_inside_ratio,
        max_workers=max_workers,
        overwrite=overwrite,
    )
    margin_counts = validate_dataset(margin_path, expected_ids, image_ext=output_image_ext)
    log_dataset_stage_summary(
        dataset_name=dataset_plan.spec.dataset_name,
        stage_name="margin",
        included_batches=dataset_plan.included_batches,
        dataset_path=margin_path,
        counts=margin_counts,
    )
    manifest["materialized_datasets"][manifest_key]["margin"] = {
        "dataset_path": str(margin_path),
        "counts": margin_counts,
        "summary": margin_summary["summary"]["total"],
    }

    wb_path = yolo_output_dir / artifact_names.wb
    wb_summary = apply_white_balance(
        src_dataset_path=margin_path,
        output_dataset_path=wb_path,
        max_workers=max_workers,
        overwrite=overwrite,
    )
    wb_counts = validate_dataset(wb_path, expected_ids, image_ext=output_image_ext)
    log_dataset_stage_summary(
        dataset_name=dataset_plan.spec.dataset_name,
        stage_name="wb",
        included_batches=dataset_plan.included_batches,
        dataset_path=wb_path,
        counts=wb_counts,
    )
    write_dataset_regeneration_summary(
        dataset_path=wb_path,
        dataset_name=dataset_plan.spec.dataset_name,
        stage_name="wb_alif",
        dataset_plan=dataset_plan,
        artifact_names=artifact_names,
        counts=wb_counts,
        source_summary=wb_summary,
    )
    manifest["materialized_datasets"][manifest_key]["wb"] = {
        "dataset_path": str(wb_path),
        "counts": wb_counts,
        "summary": wb_summary["counts"],
    }

    clahe_path = yolo_output_dir / artifact_names.clahe
    clahe_summary = apply_clahe(
        src_dataset_path=margin_path,
        output_dataset_path=clahe_path,
        max_workers=max_workers,
        overwrite=overwrite,
    )
    clahe_counts = validate_dataset(clahe_path, expected_ids, image_ext=output_image_ext)
    log_dataset_stage_summary(
        dataset_name=dataset_plan.spec.dataset_name,
        stage_name="clahe",
        included_batches=dataset_plan.included_batches,
        dataset_path=clahe_path,
        counts=clahe_counts,
    )
    write_dataset_regeneration_summary(
        dataset_path=clahe_path,
        dataset_name=dataset_plan.spec.dataset_name,
        stage_name="wb_alif_clahe",
        dataset_plan=dataset_plan,
        artifact_names=artifact_names,
        counts=clahe_counts,
        source_summary=clahe_summary,
    )
    manifest["materialized_datasets"][manifest_key]["clahe"] = {
        "dataset_path": str(clahe_path),
        "counts": clahe_counts,
        "summary": clahe_summary["counts"],
    }


def process_dataset_artifacts(
    *,
    root_path: Path,
    dataset_name: str,
    included_batches: tuple[int, ...],
    raw_path: Path,
    artifact_names: DatasetArtifactNames,
    expected_ids: dict[str, set[str]],
    margin: float,
    min_inside_ratio: float,
    max_workers: int,
    overwrite: bool,
    output_image_ext: str,
    summary_stage_prefix: str,
) -> dict[str, dict[str, Any]]:
    raw_counts = validate_dataset(raw_path, expected_ids, image_ext=output_image_ext)
    log_dataset_stage_summary(
        dataset_name=dataset_name,
        stage_name=f"{summary_stage_prefix}raw",
        included_batches=included_batches,
        dataset_path=raw_path,
        counts=raw_counts,
    )

    margin_path = raw_path.parent / artifact_names.margin
    margin_summary = apply_margin_filter(
        src_dataset_path=raw_path,
        output_dataset_path=margin_path,
        margin=margin,
        min_inside_ratio=min_inside_ratio,
        max_workers=max_workers,
        overwrite=overwrite,
    )
    margin_counts = validate_dataset(margin_path, expected_ids, image_ext=output_image_ext)
    log_dataset_stage_summary(
        dataset_name=dataset_name,
        stage_name=f"{summary_stage_prefix}margin",
        included_batches=included_batches,
        dataset_path=margin_path,
        counts=margin_counts,
    )

    wb_path = raw_path.parent / artifact_names.wb
    wb_summary = apply_white_balance(
        src_dataset_path=margin_path,
        output_dataset_path=wb_path,
        max_workers=max_workers,
        overwrite=overwrite,
    )
    wb_counts = validate_dataset(wb_path, expected_ids, image_ext=output_image_ext)
    log_dataset_stage_summary(
        dataset_name=dataset_name,
        stage_name=f"{summary_stage_prefix}wb",
        included_batches=included_batches,
        dataset_path=wb_path,
        counts=wb_counts,
    )

    clahe_path = raw_path.parent / artifact_names.clahe
    clahe_summary = apply_clahe(
        src_dataset_path=margin_path,
        output_dataset_path=clahe_path,
        max_workers=max_workers,
        overwrite=overwrite,
    )
    clahe_counts = validate_dataset(clahe_path, expected_ids, image_ext=output_image_ext)
    log_dataset_stage_summary(
        dataset_name=dataset_name,
        stage_name=f"{summary_stage_prefix}clahe",
        included_batches=included_batches,
        dataset_path=clahe_path,
        counts=clahe_counts,
    )

    return {
        "raw": {"dataset_path": str(raw_path), "counts": raw_counts},
        "margin": {
            "dataset_path": str(margin_path),
            "counts": margin_counts,
            "summary": margin_summary["summary"]["total"],
        },
        "wb": {
            "dataset_path": str(wb_path),
            "counts": wb_counts,
            "summary": wb_summary["counts"],
        },
        "clahe": {
            "dataset_path": str(clahe_path),
            "counts": clahe_counts,
            "summary": clahe_summary["counts"],
        },
    }


def materialize_pipeline(
    *,
    root_path: Path,
    yolo_output_dir: Path,
    subsets_output_dir: Path,
    stage_plans: dict[str, StagePlan],
    dataset_plans: dict[str, DatasetPlan],
    background_paths: list[str],
    margin: float,
    min_inside_ratio: float,
    output_image_ext: str,
    max_workers: int,
    overwrite: bool,
    allow_annotation_issues: bool,
    manifest: dict[str, Any],
) -> None:
    def artifact_paths(artifact_names: DatasetArtifactNames) -> dict[str, Path]:
        return {
            "raw": yolo_output_dir / artifact_names.raw,
            "margin": yolo_output_dir / artifact_names.margin,
            "wb": yolo_output_dir / artifact_names.wb,
            "clahe": yolo_output_dir / artifact_names.clahe,
        }

    def ensure_stage_chunk(stage_name: str) -> tuple[DatasetArtifactNames, dict[str, Path]]:
        stage_plan = stage_plans[stage_name]
        artifact_names = build_artifact_names(stage_name, margin)
        paths = artifact_paths(artifact_names)

        LOGGER.info("materializing stage dataset %s", stage_name)
        conversion_summary = build_dataset(
            build_kind="stage_conversion",
            root_path=root_path,
            output_dataset_path=paths["raw"],
            output_image_ext=output_image_ext,
            stage_plan=stage_plan,
            subsets_output_dir=subsets_output_dir,
            max_workers=max_workers,
            overwrite=overwrite,
            allow_annotation_issues=allow_annotation_issues,
        )
        stage_results = process_dataset_artifacts(
            root_path=root_path,
            dataset_name=stage_name,
            included_batches=stage_plan.spec.batch_ids,
            raw_path=paths["raw"],
            artifact_names=artifact_names,
            expected_ids=expected_sample_ids_for_stage_plan(root_path, stage_plan),
            margin=margin,
            min_inside_ratio=min_inside_ratio,
            max_workers=max_workers,
            overwrite=overwrite,
            output_image_ext=output_image_ext,
            summary_stage_prefix="chunk_",
        )
        manifest["materialized_datasets"][stage_name] = {
            "kind": "stage_chunk",
            "conversion_summary": conversion_summary["counts"],
            **stage_results,
        }
        write_json(subsets_output_dir / "regeneration_manifest.json", manifest)
        return artifact_names, paths

    def ensure_background_chunk() -> tuple[DatasetArtifactNames, dict[str, Path]]:
        artifact_names = build_artifact_names("background_chunk", margin)
        paths = artifact_paths(artifact_names)
        summary = build_background_chunk_dataset(
            root_path=root_path,
            output_dataset_path=paths["raw"],
            background_paths=background_paths,
            overwrite=overwrite,
            output_image_ext=output_image_ext,
        )
        expected_ids = expected_sample_ids_for_sources(
            root_path,
            train_sources=(),
            val_sources=(),
            holdout_sources=(),
            background_sources=tuple(background_paths),
        )
        chunk_results = process_dataset_artifacts(
            root_path=root_path,
            dataset_name="background_chunk",
            included_batches=(),
            raw_path=paths["raw"],
            artifact_names=artifact_names,
            expected_ids=expected_ids,
            margin=margin,
            min_inside_ratio=min_inside_ratio,
            max_workers=max_workers,
            overwrite=overwrite,
            output_image_ext=output_image_ext,
            summary_stage_prefix="chunk_",
        )
        manifest["materialized_datasets"]["background_chunk"] = {
            "kind": "background_chunk",
            "background_summary": summary["counts"],
            **chunk_results,
        }
        write_json(subsets_output_dir / "regeneration_manifest.json", manifest)
        return artifact_names, paths

    cumulative_steps: tuple[tuple[str, str | None], ...] = (
        ("batch1_3", None),
        ("batch1_3_bg", None),
        ("batch1_5_bg", "batch4_5"),
        ("batch1_7_bg", "batch6_7"),
        ("batch1_10_bg", "batch8_10"),
        ("batch1_11_bg", "batch11"),
        ("batch1_12_bg", "batch12"),
    )

    previous_cumulative_artifacts: DatasetArtifactNames | None = None

    for dataset_name, dependency_stage_name in cumulative_steps:
        dataset_plan = dataset_plans[dataset_name]
        cumulative_artifacts = build_artifact_names(dataset_name, margin)
        cumulative_paths = artifact_paths(cumulative_artifacts)
        expected_ids = expected_sample_ids_for_dataset_plan(root_path, dataset_plan)

        LOGGER.info("assembling cumulative dataset %s", dataset_name)
        if dataset_name == "batch1_3":
            stage_artifacts, stage_paths = ensure_stage_chunk("batch1_3")
            if stage_artifacts != cumulative_artifacts:
                raise RegenerationValidationError("Expected batch1_3 stage artifacts to equal cumulative artifacts")
            manifest["materialized_datasets"][dataset_name] = {
                "kind": "initial_cumulative",
                "included_batches": list(dataset_plan.included_batches),
                **{
                    split_name: manifest["materialized_datasets"]["batch1_3"][split_name]
                    for split_name in ("raw", "margin", "wb", "clahe")
                },
            }
            write_dataset_regeneration_summary(
                dataset_path=cumulative_paths["wb"],
                dataset_name=dataset_name,
                stage_name="wb_alif",
                dataset_plan=dataset_plan,
                artifact_names=cumulative_artifacts,
                counts=manifest["materialized_datasets"]["batch1_3"]["wb"]["counts"],
                source_summary=manifest["materialized_datasets"]["batch1_3"]["wb"]["summary"],
            )
            write_dataset_regeneration_summary(
                dataset_path=cumulative_paths["clahe"],
                dataset_name=dataset_name,
                stage_name="wb_alif_clahe",
                dataset_plan=dataset_plan,
                artifact_names=cumulative_artifacts,
                counts=manifest["materialized_datasets"]["batch1_3"]["clahe"]["counts"],
                source_summary=manifest["materialized_datasets"]["batch1_3"]["clahe"]["summary"],
            )
            previous_cumulative_artifacts = cumulative_artifacts
            write_json(subsets_output_dir / "regeneration_manifest.json", manifest)
            continue

        if previous_cumulative_artifacts is None:
            raise RegenerationValidationError("Missing previous cumulative artifacts")

        previous_paths = artifact_paths(previous_cumulative_artifacts)
        if dataset_name == "batch1_3_bg":
            chunk_artifacts, chunk_paths = ensure_background_chunk()
            merge_raw_summary = merge_dataset_without_holdout_v2(
                base_dataset_path=previous_paths["raw"],
                update_dataset_path=chunk_paths["raw"],
                result_dataset_path=cumulative_paths["raw"],
                overwrite=overwrite,
            )
            merge_margin_summary = merge_dataset_without_holdout_v2(
                base_dataset_path=previous_paths["margin"],
                update_dataset_path=chunk_paths["margin"],
                result_dataset_path=cumulative_paths["margin"],
                overwrite=overwrite,
            )
            merge_wb_summary = merge_dataset_without_holdout_v2(
                base_dataset_path=previous_paths["wb"],
                update_dataset_path=chunk_paths["wb"],
                result_dataset_path=cumulative_paths["wb"],
                overwrite=overwrite,
            )
            merge_clahe_summary = merge_dataset_without_holdout_v2(
                base_dataset_path=previous_paths["clahe"],
                update_dataset_path=chunk_paths["clahe"],
                result_dataset_path=cumulative_paths["clahe"],
                overwrite=overwrite,
            )
            for split_name in ("raw", "margin", "wb", "clahe"):
                validate_counts = validate_dataset(cumulative_paths[split_name], expected_ids, image_ext=output_image_ext)
                log_dataset_stage_summary(
                    dataset_name=dataset_name,
                    stage_name=split_name,
                    included_batches=dataset_plan.included_batches,
                    dataset_path=cumulative_paths[split_name],
                    counts=validate_counts,
                )
            manifest["materialized_datasets"][dataset_name] = {
                "kind": "background_incremental_merge",
                "included_batches": list(dataset_plan.included_batches),
                "raw": {"dataset_path": str(cumulative_paths["raw"]), "counts": current_dataset_counts(cumulative_paths["raw"], image_ext=output_image_ext), "summary": merge_raw_summary["counts"]},
                "margin": {"dataset_path": str(cumulative_paths["margin"]), "counts": current_dataset_counts(cumulative_paths["margin"], image_ext=output_image_ext), "summary": merge_margin_summary["counts"]},
                "wb": {"dataset_path": str(cumulative_paths["wb"]), "counts": current_dataset_counts(cumulative_paths["wb"], image_ext=output_image_ext), "summary": merge_wb_summary["counts"]},
                "clahe": {"dataset_path": str(cumulative_paths["clahe"]), "counts": current_dataset_counts(cumulative_paths["clahe"], image_ext=output_image_ext), "summary": merge_clahe_summary["counts"]},
            }
            write_dataset_regeneration_summary(
                dataset_path=cumulative_paths["wb"],
                dataset_name=dataset_name,
                stage_name="wb_alif",
                dataset_plan=dataset_plan,
                artifact_names=cumulative_artifacts,
                counts=manifest["materialized_datasets"][dataset_name]["wb"]["counts"],
                source_summary=merge_wb_summary["counts"],
            )
            write_dataset_regeneration_summary(
                dataset_path=cumulative_paths["clahe"],
                dataset_name=dataset_name,
                stage_name="wb_alif_clahe",
                dataset_plan=dataset_plan,
                artifact_names=cumulative_artifacts,
                counts=manifest["materialized_datasets"][dataset_name]["clahe"]["counts"],
                source_summary=merge_clahe_summary["counts"],
            )
            for split_name in ("raw", "margin", "wb", "clahe"):
                cleanup_dataset_path(chunk_paths[split_name])
            manifest["materialized_datasets"].pop("background_chunk", None)
        else:
            if dependency_stage_name is None:
                raise RegenerationValidationError("Missing dependency stage for incremental merge")
            chunk_artifacts, chunk_paths = ensure_stage_chunk(dependency_stage_name)
            merge_raw_summary = merge_yolo_datasets.merge_yolo_datasets(
                base_dataset_path=previous_paths["raw"],
                update_dataset_path=chunk_paths["raw"],
                result_dataset_path=cumulative_paths["raw"],
                overwrite=overwrite,
            )
            merge_margin_summary = merge_yolo_datasets.merge_yolo_datasets(
                base_dataset_path=previous_paths["margin"],
                update_dataset_path=chunk_paths["margin"],
                result_dataset_path=cumulative_paths["margin"],
                overwrite=overwrite,
            )
            merge_wb_summary = merge_yolo_datasets.merge_yolo_datasets(
                base_dataset_path=previous_paths["wb"],
                update_dataset_path=chunk_paths["wb"],
                result_dataset_path=cumulative_paths["wb"],
                overwrite=overwrite,
            )
            merge_clahe_summary = merge_yolo_datasets.merge_yolo_datasets(
                base_dataset_path=previous_paths["clahe"],
                update_dataset_path=chunk_paths["clahe"],
                result_dataset_path=cumulative_paths["clahe"],
                overwrite=overwrite,
            )
            for split_name in ("raw", "margin", "wb", "clahe"):
                validate_counts = validate_dataset(cumulative_paths[split_name], expected_ids, image_ext=output_image_ext)
                log_dataset_stage_summary(
                    dataset_name=dataset_name,
                    stage_name=split_name,
                    included_batches=dataset_plan.included_batches,
                    dataset_path=cumulative_paths[split_name],
                    counts=validate_counts,
                )
            manifest["materialized_datasets"][dataset_name] = {
                "kind": "incremental_merge",
                "included_batches": list(dataset_plan.included_batches),
                "raw": {"dataset_path": str(cumulative_paths["raw"]), "counts": current_dataset_counts(cumulative_paths["raw"], image_ext=output_image_ext), "summary": merge_raw_summary["counts"]},
                "margin": {"dataset_path": str(cumulative_paths["margin"]), "counts": current_dataset_counts(cumulative_paths["margin"], image_ext=output_image_ext), "summary": merge_margin_summary["counts"]},
                "wb": {"dataset_path": str(cumulative_paths["wb"]), "counts": current_dataset_counts(cumulative_paths["wb"], image_ext=output_image_ext), "summary": merge_wb_summary["counts"]},
                "clahe": {"dataset_path": str(cumulative_paths["clahe"]), "counts": current_dataset_counts(cumulative_paths["clahe"], image_ext=output_image_ext), "summary": merge_clahe_summary["counts"]},
            }
            write_dataset_regeneration_summary(
                dataset_path=cumulative_paths["wb"],
                dataset_name=dataset_name,
                stage_name="wb_alif",
                dataset_plan=dataset_plan,
                artifact_names=cumulative_artifacts,
                counts=manifest["materialized_datasets"][dataset_name]["wb"]["counts"],
                source_summary=merge_wb_summary["counts"],
            )
            write_dataset_regeneration_summary(
                dataset_path=cumulative_paths["clahe"],
                dataset_name=dataset_name,
                stage_name="wb_alif_clahe",
                dataset_plan=dataset_plan,
                artifact_names=cumulative_artifacts,
                counts=manifest["materialized_datasets"][dataset_name]["clahe"]["counts"],
                source_summary=merge_clahe_summary["counts"],
            )
            for split_name in ("raw", "margin", "wb", "clahe"):
                cleanup_dataset_path(chunk_paths[split_name])
            manifest["materialized_datasets"].pop(dependency_stage_name, None)

        cleanup_dataset_path(previous_paths["raw"])
        cleanup_dataset_path(previous_paths["margin"])
        previous_cumulative_artifacts = cumulative_artifacts
        write_json(subsets_output_dir / "regeneration_manifest.json", manifest)

    if previous_cumulative_artifacts is not None:
        final_previous_paths = artifact_paths(previous_cumulative_artifacts)
        cleanup_dataset_path(final_previous_paths["raw"])
        cleanup_dataset_path(final_previous_paths["margin"])


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    root_path = Path(args.root_pth).resolve(strict=False)
    yolo_output_dir = root_path / args.yolo_output_dirname
    subsets_output_dir = root_path / args.subsets_output_dirname
    holdout_json_path = root_path / args.holdout_json_name
    background_dir = root_path / args.background_dirname

    stage_specs = build_stage_specs(root_path)
    dataset_specs = build_dataset_specs()
    ensure_inputs_exist(stage_specs, holdout_json_path, background_dir)

    protected_holdout_groups, _source_to_groups, malformed_holdout_paths = antileakage_split.collect_protected_holdout_groups(
        [str(holdout_json_path)]
    )

    stage_plans: dict[str, StagePlan] = {}
    for spec in stage_specs:
        stage_plans[spec.stage_name] = build_stage_plan(
            spec=spec,
            original_holdout_json=holdout_json_path,
            protected_holdout_groups=protected_holdout_groups,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    background_paths = discover_background_paths(background_dir)
    dataset_plans = {
        dataset_spec.dataset_name: compose_dataset_plan(dataset_spec, stage_plans, background_paths)
        for dataset_spec in dataset_specs
    }

    ensure_output_targets_available(
        yolo_output_dir=yolo_output_dir,
        subsets_output_dir=subsets_output_dir,
        stage_specs=stage_specs,
        dataset_specs=dataset_specs,
        margin=args.margin,
        overwrite=args.overwrite,
    )

    manifest = build_manifest(
        root_path=root_path,
        yolo_output_dir=yolo_output_dir,
        subsets_output_dir=subsets_output_dir,
        holdout_json_path=holdout_json_path,
        background_dir=background_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        margin=args.margin,
        min_inside_ratio=args.min_inside_ratio,
        dry_run=args.dry_run,
    )
    if malformed_holdout_paths:
        warning_message = (
            "Original holdout JSON contains malformed filenames that remain in batch1_3 holdout but "
            "cannot participate in group-based protected-holdout matching for later stages."
        )
        LOGGER.warning("%s sample=%s", warning_message, malformed_holdout_paths[:10])
        manifest["warnings"].append(
            {
                "message": warning_message,
                "malformed_protected_holdout_paths": malformed_holdout_paths,
            }
        )

    manifest["stage_plans"] = {
        stage_name: {
            "batch_ids": list(stage_plan.spec.batch_ids),
            "subset_json_name": stage_plan.spec.subset_json_name,
            "historical_subset_match": stage_plan.historical_subset_match,
            "validation": stage_plan.validation,
            "counts": {split: len(paths) for split, paths in stage_plan.subsets.items()},
        }
        for stage_name, stage_plan in stage_plans.items()
    }
    manifest["dataset_plans"] = {
        dataset_name: {
            "included_batches": list(dataset_plan.included_batches),
            "background_included": dataset_plan.spec.include_background,
            "artifact_names": {
                "raw": build_artifact_names(dataset_name, args.margin).raw,
                "margin": build_artifact_names(dataset_name, args.margin).margin,
                "wb": build_artifact_names(dataset_name, args.margin).wb,
                "clahe": build_artifact_names(dataset_name, args.margin).clahe,
            },
            "counts": {
                "train": len(dataset_plan.train_sources) + len(dataset_plan.background_sources),
                "val": len(dataset_plan.val_sources),
                "holdout": len(dataset_plan.holdout_sources),
                "holdout_v2": len(dataset_plan.holdout_v2_sources),
            },
        }
        for dataset_name, dataset_plan in dataset_plans.items()
    }

    LOGGER.info("dry validation completed successfully")
    for dataset_name in (
        "batch1_3",
        "batch1_3_bg",
        "batch1_5_bg",
        "batch1_7_bg",
        "batch1_10_bg",
        "batch1_11_bg",
        "batch1_12_bg",
    ):
        dataset_plan = dataset_plans[dataset_name]
        artifact_names = build_artifact_names(dataset_name, args.margin)
        LOGGER.info(
            "validated dataset=%s raw=%s wb=%s clahe=%s batches=%s background=%s train=%s val=%s holdout=%s holdout_v2=%s",
            dataset_name,
            artifact_names.raw,
            artifact_names.wb,
            artifact_names.clahe,
            ",".join(str(batch_id) for batch_id in dataset_plan.included_batches),
            "yes" if dataset_plan.spec.include_background else "no",
            len(dataset_plan.train_sources) + len(dataset_plan.background_sources),
            len(dataset_plan.val_sources),
            len(dataset_plan.holdout_sources),
            len(dataset_plan.holdout_v2_sources),
        )

    if args.dry_run:
        manifest["status"] = "validated_dry_run"
        LOGGER.info("dry run requested; no files were written")
        return

    subsets_output_dir.mkdir(parents=True, exist_ok=True)
    yolo_output_dir.mkdir(parents=True, exist_ok=True)
    manifest["written_subset_jsons"] = write_stage_subset_jsons(stage_plans, subsets_output_dir)
    manifest["status"] = "materializing"
    write_json(subsets_output_dir / "regeneration_manifest.json", manifest)

    materialize_pipeline(
        root_path=root_path,
        yolo_output_dir=yolo_output_dir,
        subsets_output_dir=subsets_output_dir,
        stage_plans=stage_plans,
        dataset_plans=dataset_plans,
        background_paths=background_paths,
        margin=args.margin,
        min_inside_ratio=args.min_inside_ratio,
        output_image_ext=args.output_image_ext,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
        allow_annotation_issues=args.allow_annotation_issues,
        manifest=manifest,
    )
    manifest["status"] = "completed"
    write_json(subsets_output_dir / "regeneration_manifest.json", manifest)
    LOGGER.info("regeneration completed successfully")


if __name__ == "__main__":
    main()
