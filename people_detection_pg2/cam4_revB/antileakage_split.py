#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


NEW_STYLE_RAW_NAME_RE = re.compile(r"^(4\.2_H_cam_\d+_.*)_(\d+)$")
LEGACY_STYLE_RAW_NAME_RE = re.compile(r"^(4\.2_H\d+_.+)_(\d+)$")


def _stem_without_ext(path_or_name: str) -> str:
    return Path(path_or_name).stem


class GroupIdParseError(ValueError):
    pass


def _extract_group_id_and_frame_idx(raw_stem: str, original_path: str) -> tuple[str, str]:
    for pattern in (NEW_STYLE_RAW_NAME_RE, LEGACY_STYLE_RAW_NAME_RE):
        match = pattern.match(raw_stem)
        if match is not None:
            return match.group(1), match.group(2)

    raise GroupIdParseError(
        "Cannot extract raw_scene_group_id from "
        f"{original_path!r}: expected one of "
        "'4.2_H_cam_<id>_<scene>_<frame>' or '4.2_H<id>_<scene>_<frame>'"
    )


def extract_raw_scene_group_id(path_or_name: str) -> str:
    """
    Derive the raw scene group id from either:
    - raw batch path: /.../4.2_H_cam_229_Kitchen_1045.tif
    - yolo dataset path: /.../4.2_batch10____4.2_H_cam_229_Kitchen_1045.jpg

    Returns a normalized lowercase group key, for example:
    - 4.2_h_cam_229_kitchen
    - 4.2_h_cam_282_menivimos_4ho
    """
    stem = _stem_without_ext(path_or_name)
    raw_stem = stem.split("____", 1)[1] if "____" in stem else stem
    group_id, _frame_idx = _extract_group_id_and_frame_idx(raw_stem, path_or_name)

    return group_id.lower()


def collect_group_ids(paths: list[str]) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    groups: dict[str, list[str]] = defaultdict(list)
    malformed: list[dict[str, str]] = []

    for path in paths:
        try:
            group_id = extract_raw_scene_group_id(path)
        except GroupIdParseError as exc:
            malformed.append({"path": path, "error": str(exc)})
            continue

        groups[group_id].append(path)

    return (
        {group_id: sorted(items) for group_id, items in sorted(groups.items())},
        malformed,
    )


def group_paths_by_raw_scene(paths: list[str]) -> dict[str, list[str]]:
    groups, malformed = collect_group_ids(paths)
    if malformed:
        raise GroupIdParseError(f"Encountered malformed paths while grouping: {malformed[:3]}")
    return groups


def load_subset_json(path: str) -> dict[str, list[str]]:
    with open(path, "rt") as f:
        data = json.load(f)
    return {key: list(value) for key, value in data.items()}


def collect_protected_holdout_groups(
    subset_json_paths: list[str],
) -> tuple[set[str], dict[str, list[str]], list[dict[str, str]]]:
    protected_groups: set[str] = set()
    source_to_groups: dict[str, list[str]] = {}
    malformed: list[dict[str, str]] = []

    for subset_json_path in subset_json_paths:
        subsets = load_subset_json(subset_json_path)
        holdout_paths = subsets.get("holdout", [])
        grouped, bad_paths = collect_group_ids(holdout_paths)
        groups = sorted(grouped)
        protected_groups.update(groups)
        source_to_groups[subset_json_path] = groups
        malformed.extend({"source": subset_json_path, **item} for item in bad_paths)

    return protected_groups, source_to_groups, malformed


def split_new_batch(
    new_paths: list[str],
    protected_holdout_groups: set[str],
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    holdout_paths: list[str] = []
    working_paths: list[str] = []
    malformed: list[dict[str, str]] = []

    for path in sorted(new_paths):
        try:
            raw_scene_group_id = extract_raw_scene_group_id(path)
        except GroupIdParseError as exc:
            malformed.append({"path": path, "error": str(exc)})
            continue

        if raw_scene_group_id in protected_holdout_groups:
            holdout_paths.append(path)
        else:
            working_paths.append(path)

    working_groups = group_paths_by_raw_scene(working_paths)
    group_ids = sorted(working_groups)

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total_working_frames = len(working_paths)
    target_val_frames = int(total_working_frames * val_ratio)

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

    return (
        {
            "train": sorted(train_paths),
            "val": sorted(val_paths),
            "holdout": sorted(holdout_paths),
        },
        malformed,
    )


def validate_split(
    split_data: dict[str, list[str]],
    protected_holdout_groups: set[str],
    malformed_paths: list[dict[str, str]] | None = None,
    target_val_ratio: float | None = None,
) -> dict[str, Any]:
    split_to_paths = {split: sorted(paths) for split, paths in split_data.items()}
    split_to_groups = {
        split: {extract_raw_scene_group_id(path) for path in paths}
        for split, paths in split_to_paths.items()
    }

    duplicate_paths = (
        (set(split_to_paths.get("train", [])) & set(split_to_paths.get("val", [])))
        | (set(split_to_paths.get("train", [])) & set(split_to_paths.get("holdout", [])))
        | (set(split_to_paths.get("val", [])) & set(split_to_paths.get("holdout", [])))
    )

    train_val_group_overlap = split_to_groups.get("train", set()) & split_to_groups.get("val", set())
    train_holdout_group_overlap = split_to_groups.get("train", set()) & split_to_groups.get("holdout", set())
    val_holdout_group_overlap = split_to_groups.get("val", set()) & split_to_groups.get("holdout", set())

    leaked_protected_groups = (
        (split_to_groups.get("train", set()) & protected_holdout_groups)
        | (split_to_groups.get("val", set()) & protected_holdout_groups)
    )
    malformed_paths = malformed_paths or []
    working_frame_count = len(split_to_paths.get("train", [])) + len(split_to_paths.get("val", []))
    actual_val_ratio = (
        len(split_to_paths.get("val", [])) / working_frame_count if working_frame_count else 0.0
    )

    is_valid = not any(
        [
            duplicate_paths,
            train_val_group_overlap,
            train_holdout_group_overlap,
            val_holdout_group_overlap,
            leaked_protected_groups,
            malformed_paths,
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
    }


def write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def discover_paths(batch_roots: list[str], extension: str = "*.tif") -> list[str]:
    discovered: list[str] = []
    for batch_root in batch_roots:
        discovered.extend(str(path) for path in sorted(Path(batch_root).glob(extension)))
    return discovered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate leakage-safe train/val/holdout subsets using raw_scene_group_id extracted from filenames."
    )
    parser.add_argument(
        "--new-batch-root",
        dest="new_batch_roots",
        action="append",
        required=True,
        help="Path to a new raw batch folder, e.g. /home/klymy/data/cam4_revB/4.2_batch12. Repeatable.",
    )
    parser.add_argument(
        "--protected-subsets-json",
        dest="protected_subset_jsons",
        action="append",
        required=True,
        help="Path to a historical subsets JSON whose holdout split must remain protected. Repeatable.",
    )
    parser.add_argument(
        "--output-subsets-json",
        required=True,
        help="Path to write the new train/val/holdout subsets JSON.",
    )
    parser.add_argument(
        "--output-overlap-json",
        required=True,
        help="Path to write the holdout intersection report JSON.",
    )
    parser.add_argument(
        "--output-summary-json",
        required=True,
        help="Path to write the split validation summary JSON.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Target fraction of working frames allocated to val after protected holdout removal.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic group shuffling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    new_paths = discover_paths(args.new_batch_roots)
    protected_groups, source_to_groups, malformed_protected = collect_protected_holdout_groups(
        args.protected_subset_jsons
    )
    split_data, malformed_new = split_new_batch(
        new_paths=new_paths,
        protected_holdout_groups=protected_groups,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    malformed_paths = malformed_protected + malformed_new
    validation = validate_split(
        split_data,
        protected_groups,
        malformed_paths=malformed_paths,
        target_val_ratio=args.val_ratio,
    )

    overlap_report = {
        "new_batch_roots": args.new_batch_roots,
        "protected_subset_jsons": args.protected_subset_jsons,
        "protected_holdout_group_count": len(protected_groups),
        "protected_holdout_groups_sample": sorted(protected_groups)[:50],
        "protected_groups_by_source": source_to_groups,
        "matched_new_holdout_groups": sorted({extract_raw_scene_group_id(path) for path in split_data["holdout"]}),
        "matched_new_holdout_frame_count": len(split_data["holdout"]),
        "malformed_protected_paths": malformed_protected,
        "malformed_new_paths": malformed_new,
    }

    summary = {
        "config": {
            "new_batch_roots": args.new_batch_roots,
            "protected_subset_jsons": args.protected_subset_jsons,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
        },
        "validation": validation,
    }

    write_json(args.output_subsets_json, split_data)
    write_json(args.output_overlap_json, overlap_report)
    write_json(args.output_summary_json, summary)

    if not validation["is_valid"]:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
