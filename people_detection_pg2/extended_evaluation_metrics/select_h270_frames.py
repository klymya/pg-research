#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

import pandas as pd


FRAME_INDEX_RE = re.compile(r"_(\d+)(?:\.[^.]+)?$")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SelectionConfig:
    selection_budget: int
    max_per_group: int
    min_per_group: int
    val_group_count: int | None
    val_fraction_groups: float | None
    weight_f1: float
    weight_fn: float
    weight_fp: float
    weight_loc: float
    weight_conf: float
    pr_tolerance: float
    drop_inconsistent_frames: bool
    posture_cap: int | None
    position_cap: int | None
    density_cap: int | None
    materialize_dataset: bool
    materialize_output_root: str | None


def _require_path(path_str: str, name: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise ValidationError(f"{name} not found: {path}")
    return path


def _load_csv(path: Path, required_columns: list[str], name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValidationError(f"{name} is missing required columns {missing}: {path}")
    return df


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _frame_index_from_name(name: str) -> int:
    raw_name = Path(name).name
    if "____" in raw_name:
        raw_name = raw_name.split("____", 1)[1]
    match = FRAME_INDEX_RE.search(raw_name)
    if match is None:
        raise ValidationError(f"Could not parse frame index from name: {name}")
    return int(match.group(1))


def _yolo_id_from_img_id(img_id: str) -> str:
    name = Path(str(img_id)).name
    stem = Path(name).stem
    if "____" in stem:
        return stem
    return stem.replace("/", "____")


def _yolo_id_from_subset_path(path_str: str) -> str:
    path = Path(path_str)
    batch_name = path.parent.name
    stem = path.stem
    return f"{batch_name}____{stem}"


def _img_id_from_yolo_id(yolo_id: str) -> str:
    return f"{yolo_id}.jpg"


def _normalize_boolish(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    value_s = str(value).strip().lower()
    if value_s in {"true", "false"}:
        return value_s
    if not value_s:
        return "missing"
    return value_s


def _mode_or_missing(series: pd.Series) -> str:
    cleaned = [str(v).strip() for v in series.tolist() if pd.notna(v) and str(v).strip()]
    if not cleaned:
        return "missing"
    counts = pd.Series(cleaned).value_counts()
    return str(sorted(counts[counts == counts.max()].index)[0])


def _load_subset_manifest(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"Subsets JSON must be an object: {path}")

    rows: list[dict[str, Any]] = []
    for subset_name, items in data.items():
        if not isinstance(items, list):
            raise ValidationError(f"Subset '{subset_name}' must be a list in {path}")
        for item in items:
            rows.append(
                {
                    "subset_name": subset_name,
                    "source_path": str(item),
                    "yolo_id": _yolo_id_from_subset_path(str(item)),
                }
            )
    subsets_df = pd.DataFrame(rows)
    if subsets_df.empty:
        raise ValidationError(f"No subset entries found in {path}")
    dup = subsets_df["yolo_id"].duplicated()
    if dup.any():
        duplicated_ids = sorted(subsets_df.loc[dup, "yolo_id"].unique().tolist())
        raise ValidationError(f"Subset manifest contains duplicate yolo_id values: {duplicated_ids[:10]}")
    return subsets_df


def _load_groups(path: Path) -> pd.DataFrame:
    groups_df = pd.read_csv(path)
    rename_map = {}
    for column in groups_df.columns:
        normalized = column.strip().lower()
        if normalized in {"group_id", "grpoup_id"}:
            rename_map[column] = "group_id"
        elif normalized == "start_frame_included":
            rename_map[column] = "start_frame_included"
        elif normalized == "end_frame_included":
            rename_map[column] = "end_frame_included"
    groups_df = groups_df.rename(columns=rename_map)
    required = {"group_id", "start_frame_included", "end_frame_included"}
    missing = required.difference(groups_df.columns)
    if missing:
        raise ValidationError(f"Groups CSV is missing required columns {sorted(missing)}: {path}")

    groups_df = groups_df[list(required)].copy()
    groups_df["group_id"] = groups_df["group_id"].astype(int)
    groups_df["start_frame_included"] = groups_df["start_frame_included"].astype(int)
    groups_df["end_frame_included"] = groups_df["end_frame_included"].astype(int)
    groups_df = groups_df.sort_values(["start_frame_included", "end_frame_included", "group_id"]).reset_index(drop=True)
    return groups_df


def _expand_groups(groups_df: pd.DataFrame, expected_frames: set[int]) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    seen: dict[int, int] = {}
    overlaps: list[tuple[int, int, int]] = []

    for row in groups_df.itertuples(index=False):
        if row.start_frame_included > row.end_frame_included:
            raise ValidationError(
                f"Invalid group range: group {row.group_id} has start > end "
                f"({row.start_frame_included} > {row.end_frame_included})"
            )
        for frame_index in range(row.start_frame_included, row.end_frame_included + 1):
            if frame_index in seen:
                overlaps.append((frame_index, seen[frame_index], row.group_id))
            seen[frame_index] = row.group_id
            rows.append({"frame_index": frame_index, "group_id": row.group_id})

    if overlaps:
        preview = ", ".join(f"{frame}:{a}/{b}" for frame, a, b in overlaps[:10])
        raise ValidationError(f"Groups CSV assigns frames to multiple groups: {preview}")

    covered = set(seen)
    missing = sorted(expected_frames.difference(covered))
    extra = sorted(covered.difference(expected_frames))
    if missing:
        raise ValidationError(f"Groups CSV does not cover all expected frames. Missing: {missing[:20]}")
    if extra:
        raise ValidationError(f"Groups CSV covers unexpected frames outside dataset: {extra[:20]}")

    return pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)


def _aggregate_gt_metrics(metrics_per_gt: pd.DataFrame) -> pd.DataFrame:
    gt = metrics_per_gt.copy()
    gt["match_type"] = gt["match_type"].fillna("")
    gt["is_tp"] = (gt["match_type"] == "TP").astype(int)
    gt["is_fn"] = (gt["match_type"] == "FN").astype(int)
    gt["iou_num"] = pd.to_numeric(gt["iou"], errors="coerce")

    grouped = gt.groupby("yolo_id", as_index=False).agg(
        gt_count_eval=("gt_det_index", "count"),
        tp_count=("is_tp", "sum"),
        fn_count=("is_fn", "sum"),
        matched_iou_mean=("iou_num", "mean"),
        matched_iou_min=("iou_num", "min"),
    )
    grouped["matched_iou_mean"] = grouped["matched_iou_mean"].fillna(0.0)
    grouped["matched_iou_min"] = grouped["matched_iou_min"].fillna(0.0)
    return grouped


def _aggregate_pred_metrics(metrics_per_pred: pd.DataFrame) -> pd.DataFrame:
    pred = metrics_per_pred.copy()
    pred["match_type"] = pred["match_type"].fillna("")
    pred["is_tp_pred"] = (pred["match_type"] == "TP").astype(int)
    pred["is_fp"] = (pred["match_type"] == "FP").astype(int)
    pred["score_num"] = pd.to_numeric(pred["score"], errors="coerce")
    pred["pred_iou_num"] = pd.to_numeric(pred["iou"], errors="coerce")

    grouped = pred.groupby("yolo_id", as_index=False).agg(
        pred_count=("pred_det_index", "count"),
        matched_pred_count=("is_tp_pred", "sum"),
        fp_count=("is_fp", "sum"),
        pred_score_mean=("score_num", "mean"),
        pred_score_min=("score_num", "min"),
        pred_score_max=("score_num", "max"),
        matched_pred_iou_mean=("pred_iou_num", "mean"),
    )
    for column in ["pred_score_mean", "pred_score_min", "pred_score_max", "matched_pred_iou_mean"]:
        grouped[column] = grouped[column].fillna(0.0)
    return grouped


def _aggregate_detection_attrs(detection_attrs: pd.DataFrame) -> pd.DataFrame:
    det = detection_attrs.copy()
    if "occluded_by_object" in det.columns:
        det["occluded_by_object_norm"] = det["occluded_by_object"].apply(_normalize_boolish)
    if "occluded_by_person" in det.columns:
        det["occluded_by_person_norm"] = det["occluded_by_person"].apply(_normalize_boolish)
    if "position" in det.columns:
        det["position_norm"] = det["position"].apply(lambda value: str(value).strip().lower() if pd.notna(value) and str(value).strip() else "missing")

    rows: list[dict[str, Any]] = []
    for yolo_id, group in det.groupby("yolo_id"):
        row: dict[str, Any] = {"yolo_id": yolo_id}
        row["dominant_posture"] = _mode_or_missing(group["posture"]) if "posture" in group.columns else "missing"
        row["dominant_position"] = _mode_or_missing(group["position_norm"]) if "position_norm" in group.columns else "missing"
        row["occluded_by_object_mode"] = (
            _mode_or_missing(group["occluded_by_object_norm"]) if "occluded_by_object_norm" in group.columns else "missing"
        )
        row["occluded_by_person_mode"] = (
            _mode_or_missing(group["occluded_by_person_norm"]) if "occluded_by_person_norm" in group.columns else "missing"
        )
        row["position_missing_count"] = int((group["position_norm"] == "missing").sum()) if "position_norm" in group.columns else len(group)
        row["posture_missing_count"] = int(group["posture"].isna().sum()) if "posture" in group.columns else len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_frame_table(
    metrics_per_img: pd.DataFrame,
    metrics_per_gt: pd.DataFrame,
    metrics_per_pred: pd.DataFrame,
    frames_attrs: pd.DataFrame,
    detections_attrs: pd.DataFrame,
    groups_expanded: pd.DataFrame,
    subsets_df: pd.DataFrame,
    dataset_root: Path,
) -> pd.DataFrame:
    frame_df = metrics_per_img.copy()
    frame_df["img_id"] = frame_df["img_id"].astype(str).map(lambda value: Path(value).name)
    frame_df["yolo_id"] = frame_df["img_id"].map(_yolo_id_from_img_id)
    frame_df["frame_index"] = frame_df["img_id"].map(_frame_index_from_name)

    for metric in ["P", "R", "F1", "AP"]:
        frame_df[metric] = pd.to_numeric(frame_df[metric], errors="raise")

    frame_df = frame_df.merge(groups_expanded, on="frame_index", how="left", validate="one_to_one")
    if frame_df["group_id"].isna().any():
        missing = frame_df.loc[frame_df["group_id"].isna(), "img_id"].tolist()[:10]
        raise ValidationError(f"Failed to assign group_id to some frames: {missing}")

    frame_df = frame_df.merge(subsets_df, on="yolo_id", how="left", validate="one_to_one")
    if frame_df["subset_name"].isna().any():
        missing = frame_df.loc[frame_df["subset_name"].isna(), "yolo_id"].tolist()[:10]
        raise ValidationError(f"Some evaluation frames are missing from subsets manifest: {missing}")

    gt_agg = _aggregate_gt_metrics(metrics_per_gt)
    pred_agg = _aggregate_pred_metrics(metrics_per_pred)
    det_agg = _aggregate_detection_attrs(detections_attrs)

    frame_df = frame_df.merge(frames_attrs, on="yolo_id", how="left", validate="one_to_one")
    frame_df = frame_df.merge(gt_agg, on="yolo_id", how="left", validate="one_to_one")
    frame_df = frame_df.merge(pred_agg, on="yolo_id", how="left", validate="one_to_one")
    frame_df = frame_df.merge(det_agg, on="yolo_id", how="left", validate="one_to_one")

    required_sources = {
        "frames_attributes": ["num_gt_detections"],
        "metrics_per_gt": ["gt_count_eval", "tp_count", "fn_count", "matched_iou_mean", "matched_iou_min"],
        "metrics_per_pred": ["pred_count", "matched_pred_count", "fp_count", "pred_score_mean", "pred_score_min", "pred_score_max"],
    }
    for source_name, columns in required_sources.items():
        missing_mask = frame_df[columns].isna().any(axis=1)
        if missing_mask.any():
            sample = frame_df.loc[missing_mask, ["yolo_id", "img_id"]].head(10).to_dict("records")
            raise ValidationError(
                f"Required join from {source_name} is missing for {int(missing_mask.sum())} frames. "
                f"Sample: {sample}"
            )

    numeric_fill_zero = [
        "gt_count_eval",
        "tp_count",
        "fn_count",
        "pred_count",
        "matched_pred_count",
        "fp_count",
        "matched_iou_mean",
        "matched_iou_min",
        "matched_pred_iou_mean",
        "pred_score_mean",
        "pred_score_min",
        "pred_score_max",
        "num_gt_detections",
    ]
    for column in numeric_fill_zero:
        if column in frame_df.columns:
            frame_df[column] = pd.to_numeric(frame_df[column], errors="coerce").fillna(0.0)

    frame_df["scene_density"] = frame_df.get("scene_density", pd.Series(["missing"] * len(frame_df))).fillna("missing").astype(str)
    frame_df["dominant_posture"] = frame_df.get("dominant_posture", pd.Series(["missing"] * len(frame_df))).fillna("missing").astype(str)
    frame_df["dominant_position"] = frame_df.get("dominant_position", pd.Series(["missing"] * len(frame_df))).fillna("missing").astype(str)
    frame_df["precision_eval"] = frame_df["P"]
    frame_df["recall_eval"] = frame_df["R"]
    frame_df["gt_count_metadata"] = frame_df["num_gt_detections"]
    frame_df["gt_count_authoritative"] = frame_df["gt_count_eval"]

    frame_df["num_gt_detections"] = frame_df["num_gt_detections"].astype(int)
    frame_df["gt_count_eval"] = frame_df["gt_count_eval"].astype(int)
    frame_df["tp_count"] = frame_df["tp_count"].astype(int)
    frame_df["fn_count"] = frame_df["fn_count"].astype(int)
    frame_df["pred_count"] = frame_df["pred_count"].astype(int)
    frame_df["matched_pred_count"] = frame_df["matched_pred_count"].astype(int)
    frame_df["fp_count"] = frame_df["fp_count"].astype(int)
    frame_df["group_id"] = frame_df["group_id"].astype(int)

    holdout_yolo_ids = set(subsets_df.loc[subsets_df["subset_name"] == "holdout", "yolo_id"])
    frame_df["eligible_holdout"] = frame_df["yolo_id"].isin(holdout_yolo_ids)

    holdout_image_dir = dataset_root / "images" / "holdout"
    holdout_label_dir = dataset_root / "labels" / "holdout"
    frame_df["holdout_image_path"] = frame_df["img_id"].map(lambda img_id: str((holdout_image_dir / img_id).resolve()))
    frame_df["holdout_label_path"] = frame_df["yolo_id"].map(lambda yolo_id: str((holdout_label_dir / f"{yolo_id}.txt").resolve()))
    frame_df["source_subset_path"] = frame_df["source_path"].fillna("")

    for row in frame_df.itertuples(index=False):
        if row.eligible_holdout:
            if not Path(row.holdout_image_path).exists():
                raise ValidationError(f"Holdout image does not exist for selected frame source: {row.holdout_image_path}")
            if not Path(row.holdout_label_path).exists():
                raise ValidationError(f"Holdout label does not exist for selected frame source: {row.holdout_label_path}")

    return frame_df.sort_values(["group_id", "frame_index"]).reset_index(drop=True)


def validate_canonical_frame_table(frame_df: pd.DataFrame, config: SelectionConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    validated = frame_df.copy()
    validated["precision_derived"] = validated.apply(
        lambda row: _safe_div(row.tp_count, max(row.tp_count + row.fp_count, 1)),
        axis=1,
    )
    validated["recall_derived"] = validated.apply(
        lambda row: _safe_div(row.tp_count, max(row.tp_count + row.fn_count, 1)),
        axis=1,
    )
    validated["gt_count_mismatch"] = validated["gt_count_metadata"].astype(int) != validated["gt_count_eval"].astype(int)

    precision_mismatch = (validated["precision_eval"] - validated["precision_derived"]).abs() > config.pr_tolerance
    recall_mismatch = (validated["recall_eval"] - validated["recall_derived"]).abs() > config.pr_tolerance
    pr_mismatch = precision_mismatch | recall_mismatch
    dropped_inconsistent: list[str] = []
    if pr_mismatch.any() and not config.drop_inconsistent_frames:
        sample_df = validated.loc[
            pr_mismatch,
            ["yolo_id", "img_id", "precision_eval", "precision_derived", "recall_eval", "recall_derived", "tp_count", "fp_count", "fn_count"],
        ].head(10)
        raise ValidationError(
            "Per-image metrics are inconsistent with derived detailed counts for "
            f"{int(pr_mismatch.sum())} frames (tolerance={config.pr_tolerance}). Sample: "
            f"{sample_df.to_dict('records')}"
        )
    if pr_mismatch.any() and config.drop_inconsistent_frames:
        dropped_inconsistent = sorted(validated.loc[pr_mismatch, "yolo_id"].astype(str).tolist())
        validated = validated.loc[~pr_mismatch].copy()

    summary = {
        "validation_passed": True,
        "pr_inconsistency_count": int(pr_mismatch.sum()),
        "dropped_inconsistent_frames_count": len(dropped_inconsistent),
        "dropped_inconsistent_frame_ids": dropped_inconsistent,
        "drop_inconsistent_frames_enabled": config.drop_inconsistent_frames,
        "gt_count_mismatch_count": int(validated["gt_count_mismatch"].sum()),
        "gt_count_authoritative_source": "metrics_per_gt.csv",
        "ranked_output_sorted_by": ["usefulness_score_desc", "group_id_asc", "frame_index_asc", "img_id_asc"],
    }
    return validated, summary


def _compute_scores(frame_df: pd.DataFrame, config: SelectionConfig) -> pd.DataFrame:
    scored = frame_df.copy()
    scored["f1_penalty"] = 1.0 - scored["F1"].clip(lower=0.0, upper=1.0)
    scored["fn_signal"] = scored.apply(lambda row: _safe_div(row.fn_count, max(row.gt_count_authoritative, 1)), axis=1)
    scored["fp_signal"] = scored.apply(lambda row: _safe_div(row.fp_count, max(row.pred_count, 1)), axis=1)
    scored["loc_signal"] = 1.0 - scored["matched_iou_mean"].clip(lower=0.0, upper=1.0)
    scored["conf_signal"] = 1.0 - scored["pred_score_mean"].clip(lower=0.0, upper=1.0)

    scored["usefulness_score"] = (
        config.weight_f1 * scored["f1_penalty"]
        + config.weight_fn * scored["fn_signal"]
        + config.weight_fp * scored["fp_signal"]
        + config.weight_loc * scored["loc_signal"]
        + config.weight_conf * scored["conf_signal"]
    )
    return scored


def _cap_allowed(counter: dict[str, int], key: str, cap: int | None) -> bool:
    if cap is None:
        return True
    return counter.get(key, 0) < cap


def _select_frames(scored: pd.DataFrame, config: SelectionConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    eligible = scored[scored["eligible_holdout"]].copy()
    eligible["source_index"] = eligible.index
    eligible = eligible.sort_values(
        ["usefulness_score", "fn_count", "fp_count", "f1_penalty", "frame_index"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    selected_indices: list[int] = []
    selected_by_group: dict[int, int] = {}
    posture_counter: dict[str, int] = {}
    position_counter: dict[str, int] = {}
    density_counter: dict[str, int] = {}
    exclusions: list[dict[str, Any]] = []

    for row in eligible.itertuples():
        if len(selected_indices) >= config.selection_budget:
            exclusions.append({"yolo_id": row.yolo_id, "reason": "budget_exhausted"})
            continue

        group_count = selected_by_group.get(int(row.group_id), 0)
        if group_count >= config.max_per_group:
            exclusions.append({"yolo_id": row.yolo_id, "reason": "max_per_group"})
            continue

        if not _cap_allowed(posture_counter, row.dominant_posture, config.posture_cap):
            exclusions.append({"yolo_id": row.yolo_id, "reason": f"posture_cap:{row.dominant_posture}"})
            continue
        if not _cap_allowed(position_counter, row.dominant_position, config.position_cap):
            exclusions.append({"yolo_id": row.yolo_id, "reason": f"position_cap:{row.dominant_position}"})
            continue
        if not _cap_allowed(density_counter, row.scene_density, config.density_cap):
            exclusions.append({"yolo_id": row.yolo_id, "reason": f"density_cap:{row.scene_density}"})
            continue

        selected_indices.append(row.Index)
        selected_by_group[int(row.group_id)] = group_count + 1
        posture_counter[row.dominant_posture] = posture_counter.get(row.dominant_posture, 0) + 1
        position_counter[row.dominant_position] = position_counter.get(row.dominant_position, 0) + 1
        density_counter[row.scene_density] = density_counter.get(row.scene_density, 0) + 1

    selected = eligible.loc[selected_indices].copy()
    selected_group_counts = selected.groupby("group_id").size().to_dict()
    bad_groups = [int(group_id) for group_id, count in selected_group_counts.items() if count < config.min_per_group]
    if bad_groups:
        raise ValidationError(
            "Selection does not satisfy min_per_group for groups "
            f"{bad_groups}. Lower --min-per-group or increase --selection-budget."
        )

    scored["is_selected"] = False
    scored["selection_reason"] = "not_selected"
    scored.loc[selected["source_index"].tolist(), "is_selected"] = True
    scored.loc[selected["source_index"].tolist(), "selection_reason"] = "selected"
    if exclusions:
        exclusion_df = pd.DataFrame(exclusions).drop_duplicates(subset=["yolo_id"], keep="first")
        reason_map = dict(zip(exclusion_df["yolo_id"], exclusion_df["reason"]))
        scored.loc[~scored["is_selected"], "selection_reason"] = scored.loc[~scored["is_selected"], "yolo_id"].map(reason_map).fillna("not_selected")
    return scored, exclusions


def _assign_partitions(selected_df: pd.DataFrame, config: SelectionConfig) -> tuple[set[int], set[int]]:
    selected_groups = sorted(selected_df["group_id"].unique().tolist())
    if not selected_groups:
        raise ValidationError("No frames were selected; cannot assign train/val partitions.")

    if config.val_group_count is not None and config.val_fraction_groups is not None:
        raise ValidationError("Use either --val-group-count or --val-fraction-groups, not both.")

    if config.val_group_count is not None:
        val_group_count = config.val_group_count
    elif config.val_fraction_groups is not None:
        if not (0.0 < config.val_fraction_groups < 1.0):
            raise ValidationError("--val-fraction-groups must be between 0 and 1.")
        val_group_count = math.ceil(len(selected_groups) * config.val_fraction_groups)
    else:
        raise ValidationError("One of --val-group-count or --val-fraction-groups is required.")

    if val_group_count <= 0:
        raise ValidationError("Validation must receive at least one group.")
    if val_group_count >= len(selected_groups):
        raise ValidationError(
            f"Requested {val_group_count} val groups, but only {len(selected_groups)} groups were selected. "
            "At least one group must remain for train."
        )

    group_summary = (
        selected_df.groupby("group_id", as_index=False)
        .agg(group_score_mean=("usefulness_score", "mean"), selected_count=("yolo_id", "count"), first_frame=("frame_index", "min"))
        .sort_values(["group_score_mean", "selected_count", "group_id"], ascending=[False, False, True])
    )
    val_groups = set(group_summary.head(val_group_count)["group_id"].astype(int).tolist())
    train_groups = set(selected_groups).difference(val_groups)
    if not train_groups or not val_groups:
        raise ValidationError("Failed to produce non-empty train and val group assignments.")
    return train_groups, val_groups


def _materialize_dataset(
    *,
    output_dir: Path,
    materialize_output_root: Path | None,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:
    materialized_root = materialize_output_root if materialize_output_root is not None else (output_dir / "materialized_dataset")
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        image_dst = materialized_root / "images" / split_name
        label_dst = materialized_root / "labels" / split_name
        image_dst.mkdir(parents=True, exist_ok=True)
        label_dst.mkdir(parents=True, exist_ok=True)
        for row in split_df.itertuples(index=False):
            shutil.copy2(row.holdout_image_path, image_dst / Path(row.holdout_image_path).name)
            shutil.copy2(row.holdout_label_path, label_dst / Path(row.holdout_label_path).name)

    project_yaml = materialized_root / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                f"path: {materialized_root}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [1, 2]",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_report(
    output_path: Path,
    *,
    config: SelectionConfig,
    ranked_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    exclusions: list[dict[str, Any]],
    train_groups: set[int],
    val_groups: set[int],
    validation_summary: dict[str, Any],
) -> None:
    selected = ranked_df[ranked_df["partition"].isin(["train", "val"])].copy()
    worst_slices = (
        ranked_df.groupby("dominant_posture", as_index=False)
        .agg(frame_count=("yolo_id", "count"), mean_f1=("F1", "mean"), mean_score=("usefulness_score", "mean"))
        .sort_values(["mean_f1", "mean_score"], ascending=[True, False])
        .head(5)
    )
    lines = [
        "# H270 Selection Report",
        "",
        "## Summary",
        f"- Eligible holdout frames: {int(ranked_df['eligible_holdout'].sum())}",
        f"- Selected train frames: {len(train_df)}",
        f"- Selected val frames: {len(val_df)}",
        f"- Train groups: {sorted(train_groups)}",
        f"- Val groups: {sorted(val_groups)}",
        f"- Validation passed: {validation_summary['validation_passed']}",
        f"- Precision/recall inconsistencies: {validation_summary['pr_inconsistency_count']}",
        f"- Dropped inconsistent frames: {validation_summary['dropped_inconsistent_frames_count']}",
        f"- GT count mismatches (metadata vs eval): {validation_summary['gt_count_mismatch_count']}",
        f"- Authoritative GT source for scoring: {validation_summary['gt_count_authoritative_source']}",
        "- Ranked output sort: usefulness_score desc, then group_id/frame_index/img_id.",
        "",
        "## Score Composition",
        f"- weight_f1={config.weight_f1}",
        f"- weight_fn={config.weight_fn}",
        f"- weight_fp={config.weight_fp}",
        f"- weight_loc={config.weight_loc}",
        f"- weight_conf={config.weight_conf}",
        f"- pr_tolerance={config.pr_tolerance}",
        "",
        "## Leakage Safeguards Applied",
        "- Selection was restricted to current holdout frames only.",
        "- Train/val assignment was performed strictly at group_id level.",
        "- No group appears in both train and val.",
        "- Original H270 dataset was not modified unless --materialize-dataset was explicitly enabled.",
        "",
        "## Top Failure Slices",
    ]
    for row in worst_slices.itertuples(index=False):
        lines.append(
            f"- posture={row.dominant_posture}: frames={int(row.frame_count)}, "
            f"mean_f1={row.mean_f1:.4f}, mean_score={row.mean_score:.4f}"
        )
    lines.extend(
        [
            "",
            "## Group Allocation",
        ]
    )
    for group_id, group_df in selected.groupby("group_id"):
        partition = str(group_df["partition"].iloc[0])
        lines.append(
            f"- group {int(group_id)}: partition={partition}, frames={len(group_df)}, "
            f"score_mean={group_df['usefulness_score'].mean():.4f}"
        )
    lines.extend(
        [
            "",
            "## Exclusion Summary",
            f"- Non-selected entries with explicit exclusion reason tracked: {len(exclusions)}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select useful H270 frames for future YOLO expansion without modifying the original holdout dataset.")
    parser.add_argument("--metrics-per-img", required=True)
    parser.add_argument("--metrics-per-gt", required=True)
    parser.add_argument("--metrics-per-pred", required=True)
    parser.add_argument("--frames-attributes", required=True)
    parser.add_argument("--detections-attributes", required=True)
    parser.add_argument("--groups-csv", required=True)
    parser.add_argument("--subsets-json", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-budget", type=int, required=True)
    parser.add_argument("--max-per-group", type=int, required=True)
    parser.add_argument("--min-per-group", type=int, default=1)
    parser.add_argument("--val-group-count", type=int, default=None)
    parser.add_argument("--val-fraction-groups", type=float, default=None)
    parser.add_argument("--weight-f1", type=float, default=0.35)
    parser.add_argument("--weight-fn", type=float, default=0.35)
    parser.add_argument("--weight-fp", type=float, default=0.15)
    parser.add_argument("--weight-loc", type=float, default=0.10)
    parser.add_argument("--weight-conf", type=float, default=0.05)
    parser.add_argument("--pr-tolerance", type=float, default=1e-6)
    parser.add_argument("--drop-inconsistent-frames", action="store_true")
    parser.add_argument("--posture-cap", type=int, default=None)
    parser.add_argument("--position-cap", type=int, default=None)
    parser.add_argument("--density-cap", type=int, default=None)
    parser.add_argument("--materialize-dataset", action="store_true")
    parser.add_argument("--materialize-output-root", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = SelectionConfig(
        selection_budget=args.selection_budget,
        max_per_group=args.max_per_group,
        min_per_group=args.min_per_group,
        val_group_count=args.val_group_count,
        val_fraction_groups=args.val_fraction_groups,
        weight_f1=args.weight_f1,
        weight_fn=args.weight_fn,
        weight_fp=args.weight_fp,
        weight_loc=args.weight_loc,
        weight_conf=args.weight_conf,
        pr_tolerance=args.pr_tolerance,
        drop_inconsistent_frames=args.drop_inconsistent_frames,
        posture_cap=args.posture_cap,
        position_cap=args.position_cap,
        density_cap=args.density_cap,
        materialize_dataset=args.materialize_dataset,
        materialize_output_root=args.materialize_output_root,
    )

    if config.selection_budget <= 0:
        raise ValidationError("--selection-budget must be > 0")
    if config.max_per_group <= 0:
        raise ValidationError("--max-per-group must be > 0")
    if config.min_per_group < 0:
        raise ValidationError("--min-per-group must be >= 0")
    if config.min_per_group > config.max_per_group:
        raise ValidationError("--min-per-group cannot exceed --max-per-group")

    metrics_per_img_path = _require_path(args.metrics_per_img, "metrics_per_img.csv")
    metrics_per_gt_path = _require_path(args.metrics_per_gt, "metrics_per_gt.csv")
    metrics_per_pred_path = _require_path(args.metrics_per_pred, "metrics_per_pred.csv")
    frames_attrs_path = _require_path(args.frames_attributes, "frames_attributes.csv")
    detections_attrs_path = _require_path(args.detections_attributes, "detections_attributes.csv")
    groups_csv_path = _require_path(args.groups_csv, "groups.csv")
    subsets_json_path = _require_path(args.subsets_json, "subsets.json")
    dataset_root = _require_path(args.dataset_root, "dataset root")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_per_img = _load_csv(metrics_per_img_path, ["img_id", "P", "R", "F1", "AP"], "metrics_per_img.csv")
    metrics_per_gt = _load_csv(metrics_per_gt_path, ["img_id", "yolo_id", "match_type", "gt_det_index", "iou"], "metrics_per_gt.csv")
    metrics_per_pred = _load_csv(metrics_per_pred_path, ["img_id", "yolo_id", "match_type", "pred_det_index", "score"], "metrics_per_pred.csv")
    frames_attrs = _load_csv(
        frames_attrs_path,
        ["yolo_id", "xml_image_name", "num_gt_detections", "scene_density"],
        "frames_attributes.csv",
    )
    detections_attrs = _load_csv(
        detections_attrs_path,
        ["yolo_id", "det_index", "position", "posture", "occluded_by_object", "occluded_by_person"],
        "detections_attributes.csv",
    )
    groups_df = _load_groups(groups_csv_path)
    subsets_df = _load_subset_manifest(subsets_json_path)

    expected_frames = {_frame_index_from_name(yolo_id) for yolo_id in subsets_df["yolo_id"]}
    groups_expanded = _expand_groups(groups_df, expected_frames=expected_frames)

    ranked_df = _build_frame_table(
        metrics_per_img=metrics_per_img,
        metrics_per_gt=metrics_per_gt,
        metrics_per_pred=metrics_per_pred,
        frames_attrs=frames_attrs,
        detections_attrs=detections_attrs,
        groups_expanded=groups_expanded,
        subsets_df=subsets_df,
        dataset_root=dataset_root,
    )
    ranked_df, validation_summary = validate_canonical_frame_table(ranked_df, config=config)
    ranked_df = _compute_scores(ranked_df, config=config)
    ranked_df["partition"] = "not_selected"

    ranked_df, exclusions = _select_frames(ranked_df, config=config)
    selected_df = ranked_df[ranked_df["is_selected"]].copy()
    train_groups, val_groups = _assign_partitions(selected_df, config=config)
    ranked_df.loc[ranked_df["group_id"].isin(train_groups) & ranked_df["is_selected"], "partition"] = "train"
    ranked_df.loc[ranked_df["group_id"].isin(val_groups) & ranked_df["is_selected"], "partition"] = "val"

    train_df = ranked_df[ranked_df["partition"] == "train"].copy()
    val_df = ranked_df[ranked_df["partition"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise ValidationError("Selection produced an empty train or val manifest.")

    ranked_output = ranked_df.sort_values(
        ["usefulness_score", "group_id", "frame_index", "img_id"],
        ascending=[False, True, True, True],
    )
    ranked_output.to_csv(output_dir / "ranked_frames.csv", index=False)
    train_df.sort_values(["group_id", "frame_index"]).to_csv(output_dir / "selected_train_manifest.csv", index=False)
    val_df.sort_values(["group_id", "frame_index"]).to_csv(output_dir / "selected_val_manifest.csv", index=False)

    summary = {
        "config": asdict(config),
        "input_files": {
            "metrics_per_img": str(metrics_per_img_path),
            "metrics_per_gt": str(metrics_per_gt_path),
            "metrics_per_pred": str(metrics_per_pred_path),
            "frames_attributes": str(frames_attrs_path),
            "detections_attributes": str(detections_attrs_path),
            "groups_csv": str(groups_csv_path),
            "subsets_json": str(subsets_json_path),
            "dataset_root": str(dataset_root),
        },
        "total_frames": int(len(ranked_df)),
        "eligible_frames": int(ranked_df["eligible_holdout"].sum()),
        "selected_frames": int(selected_df.shape[0]),
        "train_frames": int(train_df.shape[0]),
        "val_frames": int(val_df.shape[0]),
        "selected_groups": sorted(selected_df["group_id"].unique().tolist()),
        "rejected_groups": sorted(set(ranked_df["group_id"].unique().tolist()).difference(set(selected_df["group_id"].unique().tolist()))),
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "score_stats": {
            "min": float(ranked_df["usefulness_score"].min()),
            "max": float(ranked_df["usefulness_score"].max()),
            "mean": float(ranked_df["usefulness_score"].mean()),
            "selected_mean": float(selected_df["usefulness_score"].mean()),
        },
        "validation": validation_summary,
        "exclusions": exclusions[:500],
        "excluded_reason_counts": pd.DataFrame(exclusions)["reason"].value_counts().to_dict() if exclusions else {},
    }
    with (output_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _write_report(
        output_path=output_dir / "selection_report.md",
        config=config,
        ranked_df=ranked_df,
        train_df=train_df,
        val_df=val_df,
        exclusions=exclusions,
        train_groups=train_groups,
        val_groups=val_groups,
        validation_summary=validation_summary,
    )

    if config.materialize_dataset:
        materialize_output_root = (
            Path(config.materialize_output_root).expanduser().resolve()
            if config.materialize_output_root is not None
            else None
        )
        _materialize_dataset(
            output_dir=output_dir,
            materialize_output_root=materialize_output_root,
            train_df=train_df,
            val_df=val_df,
        )

    print(f"Wrote selection outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
