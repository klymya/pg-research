#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[2] / "objectdetectionmetrics"))
from metricsevaluator import MetricsEvaluator


SELECTION_METRICS = {"f1", "recall", "precision"}


@dataclass(frozen=True)
class EvalParams:
    conf_th: float
    iou_th: float
    run_name: str


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    return value if value is not None else default


def _parse_device(value: str | None):
    if not value:
        return None
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    try:
        return int(value)
    except ValueError:
        return value


def _parse_bool_env(key: str, default: bool = False) -> bool:
    value = (_env(key, "1" if default else "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _parse_float_list(raw: str | None) -> list[float]:
    if raw is None:
        return []
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def _normalize_selection_metric(value: str | None) -> str:
    metric = (value or "f1").strip().lower()
    if metric not in SELECTION_METRICS:
        raise ValueError(
            f"Unsupported SELECTION_METRIC={value!r}. "
            f"Expected one of: {', '.join(sorted(SELECTION_METRICS))}."
        )
    return metric


def _selection_metric_key(metric: str) -> str:
    return f"selection/{metric}"


def _compute_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall + 1e-9)


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value_f):
        return None
    return value_f


def _derive_selection_summary(summary: dict[str, Any]) -> dict[str, Any]:
    pose_precision = _to_float_or_none(summary.get("metrics/precision(P)"))
    pose_recall = _to_float_or_none(summary.get("metrics/recall(P)"))
    if pose_precision is not None and pose_recall is not None:
        return {
            "selection_source": "validator_pose_metrics",
            "selection/precision": pose_precision,
            "selection/recall": pose_recall,
            "selection/f1": _compute_f1(pose_precision, pose_recall),
        }

    detection_precision = float(summary["detection/precision"])
    detection_recall = float(summary["detection/recall"])
    return {
        "selection_source": "local_detection_metrics",
        "selection/precision": detection_precision,
        "selection/recall": detection_recall,
        "selection/f1": float(summary["detection/f1"]),
    }


def _format_float_tag(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def _build_run_name(base_name: str, conf_th: float, iou_th: float) -> str:
    return (
        f"{base_name}"
        f"__conf_{_format_float_tag(conf_th)}"
        f"__iou_{_format_float_tag(iou_th)}"
    )


def _yolo_id_from_img_id(img_id: str) -> str:
    return str(Path(img_id).with_suffix("")).replace("/", "____")


def _resolve_path(path_str: str) -> str:
    if not path_str:
        return path_str
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / path
    return str(candidate) if candidate.exists() else str(path)


def _get_image_size(img_path: str) -> tuple[int, int]:
    try:
        import cv2

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"cv2.imread returned None for {img_path}")
        height, width = image.shape[:2]
        return height, width
    except Exception:
        from PIL import Image

        with Image.open(img_path) as image:
            width, height = image.size
        return height, width


def _xywh_to_xyxy(x_c, y_c, w, h):
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return x1, y1, x2, y2


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _greedy_match(iou_mat: np.ndarray, iou_th: float) -> list[tuple[int, int, float]]:
    matches: list[tuple[int, int, float]] = []
    if iou_mat.size == 0:
        return matches
    mat = iou_mat.copy()
    while True:
        idx = np.unravel_index(np.argmax(mat), mat.shape)
        max_iou = mat[idx]
        if max_iou < iou_th:
            break
        pred_idx, gt_idx = int(idx[0]), int(idx[1])
        matches.append((pred_idx, gt_idx, float(max_iou)))
        mat[pred_idx, :] = -1
        mat[:, gt_idx] = -1
    return matches


def _load_env() -> None:
    env_path = os.getenv("ENV_PATH", ".env")
    load_dotenv(env_path)


def _normalize_validator_device(device):
    if isinstance(device, list):
        return device
    if device is None:
        return ["cpu"]
    return [device]


def _resolve_image_filename(raw_image_id: str, imgs_pth: str, cache: dict[str, str]) -> str:
    if raw_image_id in cache:
        return cache[raw_image_id]

    image_path = Path(imgs_pth) / raw_image_id
    if image_path.exists():
        cache[raw_image_id] = raw_image_id
        return raw_image_id

    stem = Path(raw_image_id).stem
    matches = sorted(Path(imgs_pth).glob(f"{stem}.*"))
    if matches:
        resolved = matches[0].name
        cache[raw_image_id] = resolved
        return resolved

    guessed = f"{raw_image_id}.jpg"
    cache[raw_image_id] = guessed
    return guessed


def _list_image_files(imgs_pth: str) -> list[str]:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    path = Path(imgs_pth)
    if not path.exists():
        return []
    return sorted([item.name for item in path.iterdir() if item.is_file() and item.suffix.lower() in img_exts])


def _run_pt_validation(
    model: YOLO,
    dataset_pth: str,
    device,
    batch: int,
    conf_th: float,
    iou_th: float,
    project_dir: str,
    project_name: str,
):
    return model.val(
        data=dataset_pth,
        task="pose",
        device=device,
        batch=batch,
        verbose=True,
        conf=conf_th,
        iou=iou_th,
        project=project_dir,
        name=project_name,
        exist_ok=True,
        save_json=True,
    )


def _build_tflite_inferer(
    backend: str,
    conf_th: float,
    iou_th: float,
    input_w: int,
    tflite_backbone_pth: str,
    tflite_head_pth: str,
    tflite_output_correction_model_pth: str,
):
    inferer_cls = None
    emulator_src = "/home/klymy/projects/algo_system_emulator/src"
    if Path(emulator_src).exists():
        if emulator_src not in sys.path:
            sys.path.append(emulator_src)
        try:
            from alif.tflite_inference import TFLiteInferer as _Inferer

            inferer_cls = _Inferer
        except Exception as exc:
            print(f"Warning: failed to import emulator TFLiteInferer ({exc}), using local inferer.")

    if inferer_cls is None:
        people_det_root = str(Path(__file__).resolve().parents[1])
        if people_det_root not in sys.path:
            sys.path.append(people_det_root)
        from tflite_inference import TFLiteInferer as _Inferer

        inferer_cls = _Inferer

    head_pth = tflite_head_pth if backend == "tflite_mixed" else None
    kwargs = dict(
        yolo_pth=tflite_backbone_pth,
        head_pth=head_pth,
        conf_th=conf_th,
        iou_th=iou_th,
        imgsz=input_w,
        separated_head=False,
        head_int8=False,
    )

    if backend == "tflite_backbone_only" and tflite_output_correction_model_pth:
        signature = inspect.signature(inferer_cls.__init__)
        if "output_correction_model_path" in signature.parameters:
            kwargs["output_correction_model_path"] = tflite_output_correction_model_pth
        else:
            raise ValueError(
                "Selected TFLite inferer does not support output_correction_model_path, "
                "but TFLITE_OUTPUT_CORRECTION_MODEL_PTH is set."
            )

    return inferer_cls(**kwargs)


def _run_tflite_validation(
    model: YOLO,
    dataset_pth: str,
    project_dir: str,
    project_name: str,
    batch: int,
    device,
    input_w: int,
    conf_th: float,
    iou_th: float,
    backend: str,
    tflite_backbone_pth: str,
    tflite_head_pth: str,
    tflite_output_correction_model_pth: str,
):
    people_det_root = str(Path(__file__).resolve().parents[1])
    if people_det_root not in sys.path:
        sys.path.append(people_det_root)
    import ultralytics_validator_wrapper

    inferer = _build_tflite_inferer(
        backend=backend,
        conf_th=conf_th,
        iou_th=iou_th,
        input_w=input_w,
        tflite_backbone_pth=tflite_backbone_pth,
        tflite_head_pth=tflite_head_pth,
        tflite_output_correction_model_pth=tflite_output_correction_model_pth,
    )

    val_args = dict(
        task="pose",
        device=_normalize_validator_device(device),
        batch=batch,
        verbose=True,
        conf=conf_th,
        iou=iou_th,
        imgsz=input_w,
        model="val",
        rect=False,
        single_cls=False,
        save_json=True,
        save_txt=False,
        plots=False,
        split="val",
        project=project_dir,
        name=project_name,
        exist_ok=True,
        data=dataset_pth,
        val=True,
    )

    stats = ultralytics_validator_wrapper.validate_model(
        val_args,
        model,
        functools.partial(ultralytics_validator_wrapper.predict_tflite, inferer, val_args),
        draw=False,
    )
    return stats if isinstance(stats, dict) else {}


def _load_gt_and_images(imgs_pth: str, labels_pth: str) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, tuple[int, int]]]:
    all_img_ids = _list_image_files(imgs_pth)
    gt_by_img: dict[str, list[dict[str, Any]]] = {}
    img_sizes: dict[str, tuple[int, int]] = {}

    for img_id in tqdm(all_img_ids, desc="Loading GT", leave=False):
        img_h, img_w = _get_image_size(os.path.join(imgs_pth, img_id))
        img_sizes[img_id] = (img_h, img_w)

        label_path = os.path.join(labels_pth, str(Path(img_id).with_suffix(".txt")))
        gt_entries: list[dict[str, Any]] = []
        if os.path.exists(label_path):
            with open(label_path, "rt") as handle:
                for index, line in enumerate(handle):
                    parts = [float(item) for item in line.strip().split()]
                    if len(parts) < 5:
                        continue
                    cls, x_c, y_c, w, h = parts[:5]
                    if cls > 1 or np.isnan(x_c):
                        continue
                    x1, y1, x2, y2 = _xywh_to_xyxy(x_c, y_c, w, h)
                    gt_entries.append(
                        {
                            "gt_index": index,
                            "x_c": x_c,
                            "y_c": y_c,
                            "w": w,
                            "h": h,
                            "x1": x1 * img_w,
                            "y1": y1 * img_h,
                            "x2": x2 * img_w,
                            "y2": y2 * img_h,
                        }
                    )
        gt_by_img[img_id] = gt_entries

    return all_img_ids, gt_by_img, img_sizes


def _load_det_attrs(detections_attr_csv: str) -> pd.DataFrame | None:
    if detections_attr_csv and os.path.exists(detections_attr_csv):
        return pd.read_csv(detections_attr_csv)
    return None


def _normalize_predictions(predictions: list[dict[str, Any]], imgs_pth: str) -> dict[str, list[dict[str, Any]]]:
    predictions_by_id: dict[str, list[dict[str, Any]]] = {}
    image_id_cache: dict[str, str] = {}

    for prediction in predictions:
        raw_img_id = str(prediction["image_id"])
        img_id = _resolve_image_filename(raw_img_id, imgs_pth, image_id_cache)
        normalized = dict(prediction)
        normalized["image_id"] = img_id
        predictions_by_id.setdefault(img_id, []).append(normalized)

    return predictions_by_id


def _prepare_pred_boxes(
    preds: list[dict[str, Any]],
    img_h: int,
    img_w: int,
    conf_th: float,
) -> list[dict[str, Any]]:
    pred_boxes: list[dict[str, Any]] = []
    for prediction in preds:
        score = float(prediction["score"])
        if score < conf_th:
            continue
        x, y, w, h = np.array(prediction["bbox"], dtype=float)
        keypoints = prediction.get("keypoints") or []
        if isinstance(keypoints, (list, tuple)) and len(keypoints) >= 2:
            pred_x = float(keypoints[0]) / img_w
            pred_y = float(keypoints[1]) / img_h
        else:
            pred_x = None
            pred_y = None
        pred_boxes.append(
            {
                "pred_index": len(pred_boxes),
                "score": score,
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "x_c": (x + w / 2) / img_w,
                "y_c": (y + h / 2) / img_h,
                "w": w / img_w,
                "h": h / img_h,
                "x": pred_x,
                "y": pred_y,
            }
        )
    return pred_boxes


def _compute_iou_matrix(pred_boxes: list[dict[str, Any]], gt_boxes: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for pred_idx, pred in enumerate(pred_boxes):
        for gt_idx, gt in enumerate(gt_boxes):
            matrix[pred_idx, gt_idx] = _iou_xyxy(
                (pred["x1"], pred["y1"], pred["x2"], pred["y2"]),
                (gt["x1"], gt["y1"], gt["x2"], gt["y2"]),
            )
    return matrix


def _eval_per_image_metrics(
    img_id: str,
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    input_w: int,
    input_h: int,
    match_iou_th: float,
) -> dict[str, float | str]:
    if not gt_boxes and not pred_boxes:
        # A pure background frame predicted as background is a perfect frame-level outcome.
        return {
            "img_id": img_id,
            "P": 1.0,
            "R": 1.0,
            "F1": 1.0,
            "AP": 0.0,
        }

    evaluator = MetricsEvaluator(
        is_relative_coordinates=True,
        is_xywh_coordinates=True,
        img_size=(input_w, input_h),
        outpath=".",
    )

    for gt in gt_boxes:
        evaluator.add_gt_box(
            img_path=img_id,
            class_name="person",
            bbox=[gt["x_c"], gt["y_c"], gt["w"], gt["h"]],
        )

    for pred in pred_boxes:
        evaluator.add_det_box(
            img_path=img_id,
            class_name="person",
            bbox=[pred["x_c"], pred["y_c"], pred["w"], pred["h"]],
            confidence=pred["score"],
        )

    results = evaluator.eval_conf_vals(match_iou_th)
    if results:
        result = results[0]
        tp = float(result["total TP"])
        fp = float(result["total FP"])
        total_pos = float(result["total positives"])
        ap = float(result["AP"])
    else:
        tp = 0.0
        fp = float(len(pred_boxes))
        total_pos = float(len(gt_boxes))
        ap = 0.0

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (total_pos + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "img_id": img_id,
        "P": precision,
        "R": recall,
        "F1": f1,
        "AP": ap,
    }


def _gt_attrs_lookup(det_attrs: pd.DataFrame | None, yolo_id: str, gt_index: int) -> dict[str, Any]:
    if det_attrs is None:
        return {}
    subset = det_attrs[(det_attrs["yolo_id"] == yolo_id) & (det_attrs["det_index"] == gt_index)]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _evaluate_predictions(
    *,
    predictions_by_id: dict[str, list[dict[str, Any]]],
    all_img_ids: list[str],
    gt_by_img: dict[str, list[dict[str, Any]]],
    img_sizes: dict[str, tuple[int, int]],
    input_w: int,
    input_h: int,
    conf_th: float,
    match_iou_th: float,
    det_attrs: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    metrics_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    n_missing_preds = sum(1 for img_id in all_img_ids if img_id not in predictions_by_id)
    if n_missing_preds:
        print(
            f"Warning: {n_missing_preds}/{len(all_img_ids)} images have no entries in predictions.json; "
            "treating them as zero-detection frames."
        )

    for img_id in tqdm(all_img_ids, total=len(all_img_ids), desc="Scoring predictions"):
        preds = predictions_by_id.get(img_id, [])
        gt_boxes = gt_by_img.get(img_id, [])
        img_h, img_w = img_sizes[img_id]
        pred_boxes = _prepare_pred_boxes(preds, img_h=img_h, img_w=img_w, conf_th=conf_th)

        metrics_rows.append(
            _eval_per_image_metrics(
                img_id=img_id,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                input_w=input_w,
                input_h=input_h,
                match_iou_th=match_iou_th,
            )
        )

        yolo_id = _yolo_id_from_img_id(img_id)
        iou_mat = _compute_iou_matrix(pred_boxes, gt_boxes)
        matches = _greedy_match(iou_mat, match_iou_th)
        matched_pred = {match[0] for match in matches}
        matched_gt = {match[1] for match in matches}

        total_tp += len(matches)
        total_fp += len(pred_boxes) - len(matches)
        total_fn += len(gt_boxes) - len(matches)

        for match_id, (pred_i, gt_j, iou_value) in enumerate(matches):
            pred = pred_boxes[pred_i]
            gt = gt_boxes[gt_j]
            gt_row = {
                "img_id": img_id,
                "yolo_id": yolo_id,
                "match_type": "TP",
                "match_id": f"{yolo_id}::m::{match_id}",
                "iou": iou_value,
                "gt_det_index": gt["gt_index"],
                "gt_x_c": gt["x_c"],
                "gt_y_c": gt["y_c"],
                "gt_w": gt["w"],
                "gt_h": gt["h"],
                "matched_pred_index": pred["pred_index"],
            }
            gt_row.update(_gt_attrs_lookup(det_attrs, yolo_id, gt["gt_index"]))
            gt_rows.append(gt_row)

            pred_rows.append(
                {
                    "img_id": img_id,
                    "yolo_id": yolo_id,
                    "match_type": "TP",
                    "match_id": f"{yolo_id}::m::{match_id}",
                    "iou": iou_value,
                    "score": pred["score"],
                    "pred_det_index": pred["pred_index"],
                    "pred_x_c": pred["x_c"],
                    "pred_y_c": pred["y_c"],
                    "pred_w": pred["w"],
                    "pred_h": pred["h"],
                    "pred_x": "" if pred["x"] is None else pred["x"],
                    "pred_y": "" if pred["y"] is None else pred["y"],
                    "matched_gt_index": gt["gt_index"],
                }
            )

        for pred in pred_boxes:
            if pred["pred_index"] in matched_pred:
                continue
            pred_rows.append(
                {
                    "img_id": img_id,
                    "yolo_id": yolo_id,
                    "match_type": "FP",
                    "match_id": "",
                    "iou": "",
                    "score": pred["score"],
                    "pred_det_index": pred["pred_index"],
                    "pred_x_c": pred["x_c"],
                    "pred_y_c": pred["y_c"],
                    "pred_w": pred["w"],
                    "pred_h": pred["h"],
                    "pred_x": "" if pred["x"] is None else pred["x"],
                    "pred_y": "" if pred["y"] is None else pred["y"],
                    "matched_gt_index": "",
                }
            )

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            gt_row = {
                "img_id": img_id,
                "yolo_id": yolo_id,
                "match_type": "FN",
                "match_id": "",
                "iou": "",
                "gt_det_index": gt["gt_index"],
                "gt_x_c": gt["x_c"],
                "gt_y_c": gt["y_c"],
                "gt_w": gt["w"],
                "gt_h": gt["h"],
                "matched_pred_index": "",
            }
            gt_row.update(_gt_attrs_lookup(det_attrs, yolo_id, gt["gt_index"]))
            gt_rows.append(gt_row)

    metrics_df = pd.DataFrame(metrics_rows).fillna(0.0)
    gt_df = pd.DataFrame(gt_rows)
    pred_df = pd.DataFrame(pred_rows)

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    summary: dict[str, float | int] = {
        "detection/tp": int(total_tp),
        "detection/fp": int(total_fp),
        "detection/fn": int(total_fn),
        "detection/precision": float(precision),
        "detection/recall": float(recall),
        "detection/f1": float(f1),
        "frame/mean_precision": float(metrics_df["P"].mean()) if not metrics_df.empty else 0.0,
        "frame/mean_recall": float(metrics_df["R"].mean()) if not metrics_df.empty else 0.0,
        "frame/mean_f1": float(metrics_df["F1"].mean()) if not metrics_df.empty else 0.0,
        "frame/mean_ap": float(metrics_df["AP"].mean()) if not metrics_df.empty else 0.0,
        "frame/count": int(len(metrics_df)),
    }
    return metrics_df, gt_df, pred_df, summary


def _ensure_parent(path_str: str) -> None:
    if not path_str:
        return
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def _write_outputs(
    *,
    metrics_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    metrics_csv_path: str,
    gt_csv_path: str,
    pred_csv_path: str,
) -> None:
    _ensure_parent(metrics_csv_path)
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Wrote per-image metrics: {metrics_csv_path}")
    print(metrics_df[["P", "R", "F1", "AP"]].mean())

    if gt_csv_path:
        _ensure_parent(gt_csv_path)
        gt_df.to_csv(gt_csv_path, index=False)
        print(f"Wrote GT-level CSV: {gt_csv_path}")

    if pred_csv_path:
        _ensure_parent(pred_csv_path)
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"Wrote prediction-level CSV: {pred_csv_path}")


def _copy_file(src: str, dst: str) -> None:
    source = Path(src)
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve(strict=False) == target.resolve(strict=False):
        return
    shutil.copy2(str(source), str(target))


def _run_validation_for_params(
    *,
    model: YOLO,
    dataset_pth: str,
    device,
    batch: int,
    input_w: int,
    params: EvalParams,
    project_dir: str,
    eval_backend: str,
    weights_pth: str,
    tflite_backbone_pth: str,
    tflite_head_pth: str,
    tflite_output_correction_model_pth: str,
) -> tuple[str, dict[str, Any], str]:
    print(
        f"Running validation from scratch for {params.run_name} "
        f"(conf={params.conf_th:.4f}, iou={params.iou_th:.4f})"
    )

    if not os.path.exists(weights_pth):
        raise FileNotFoundError(
            f"Validator PT weights not found: {weights_pth}. "
            "Set BASE_EXPERIMENT_DIR correctly or override WEIGHTS_PTH."
        )

    if eval_backend == "ultralytics_pt":
        val_result = _run_pt_validation(
            model=model,
            dataset_pth=dataset_pth,
            device=device,
            batch=batch,
            conf_th=params.conf_th,
            iou_th=params.iou_th,
            project_dir=project_dir,
            project_name=params.run_name,
        )
        global_metrics = val_result.results_dict if hasattr(val_result, "results_dict") else {}
    elif eval_backend in {"tflite_mixed", "tflite_backbone_only"}:
        if not tflite_backbone_pth:
            raise ValueError("TFLITE_BACKBONE_PTH is required for TFLite backends.")
        if not os.path.exists(tflite_backbone_pth):
            raise FileNotFoundError(f"TFLite backbone model not found: {tflite_backbone_pth}")
        if eval_backend == "tflite_mixed" and not tflite_head_pth:
            raise ValueError("TFLITE_HEAD_PTH is required for tflite_mixed backend.")
        if eval_backend == "tflite_mixed" and not os.path.exists(tflite_head_pth):
            raise FileNotFoundError(f"TFLite head model not found: {tflite_head_pth}")
        if eval_backend == "tflite_backbone_only":
            if tflite_output_correction_model_pth:
                if not os.path.exists(tflite_output_correction_model_pth):
                    raise FileNotFoundError(
                        f"Output correction model not found: {tflite_output_correction_model_pth}"
                    )
                print(f"Using output correction model: {tflite_output_correction_model_pth}")
            else:
                print("No output correction model provided; running without correction.")

        global_metrics = _run_tflite_validation(
            model=model,
            dataset_pth=dataset_pth,
            project_dir=project_dir,
            project_name=params.run_name,
            batch=batch,
            device=device,
            input_w=input_w,
            conf_th=params.conf_th,
            iou_th=params.iou_th,
            backend=eval_backend,
            tflite_backbone_pth=tflite_backbone_pth,
            tflite_head_pth=tflite_head_pth,
            tflite_output_correction_model_pth=tflite_output_correction_model_pth,
        )
    else:
        raise ValueError(f"Unsupported EVAL_BACKEND: {eval_backend}")

    generated_predictions_path = os.path.join(project_dir, params.run_name, "predictions.json")
    if not os.path.exists(generated_predictions_path):
        raise FileNotFoundError(f"Validation completed but predictions.json was not found: {generated_predictions_path}")

    return generated_predictions_path, global_metrics, os.path.join(project_dir, params.run_name)


def _build_eval_plan(
    *,
    base_project_name: str,
    base_conf_th: float,
    base_iou_th: float,
    enable_sweep: bool,
    sweep_conf_ths: list[float],
    sweep_iou_ths: list[float],
) -> list[EvalParams]:
    if not enable_sweep:
        return [EvalParams(conf_th=base_conf_th, iou_th=base_iou_th, run_name=base_project_name)]

    conf_values = sweep_conf_ths or [base_conf_th]
    iou_values = sweep_iou_ths or [base_iou_th]
    plan: list[EvalParams] = []
    seen: set[tuple[float, float]] = set()
    for conf_th in conf_values:
        for iou_th in iou_values:
            pair = (float(conf_th), float(iou_th))
            if pair in seen:
                continue
            seen.add(pair)
            plan.append(
                EvalParams(
                    conf_th=pair[0],
                    iou_th=pair[1],
                    run_name=_build_run_name(base_project_name, conf_th=pair[0], iou_th=pair[1]),
                )
            )
    if not plan:
        raise ValueError("ENABLE_SWEEP=1 but no parameter combinations were produced.")
    return plan


def _build_best_sort_key(row: dict[str, Any], selection_metric: str) -> tuple[float, ...]:
    primary = float(row[_selection_metric_key(selection_metric)])
    recall = float(row["selection/recall"])
    precision = float(row["selection/precision"])
    f1 = float(row["selection/f1"])
    conf_th = float(row["conf_th"])
    iou_th = float(row["iou_th"])

    if selection_metric == "recall":
        return (primary, f1, precision, conf_th, iou_th)
    if selection_metric == "precision":
        return (primary, f1, recall, conf_th, iou_th)
    return (primary, recall, precision, conf_th, iou_th)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO validation and export per-frame / per-detection metrics.")
    parser.add_argument("--env-file", default="", help="Optional path to .env file. If empty, uses ENV_PATH.")
    parser.add_argument("--out-csv", default="", help="Output CSV path (default: from env METRICS_PER_IMG).")
    parser.add_argument("--out-gt-csv", default="", help="Output GT CSV path (default: from env METRICS_PER_GT).")
    parser.add_argument("--out-pred-csv", default="", help="Output prediction CSV path (default: from env METRICS_PER_PRED).")
    parser.add_argument("--out-global-json", default="", help="Output global metrics JSON path (default: from env GLOBAL_METRICS_PTH).")
    args, _unknown = parser.parse_known_args()

    if args.env_file:
        load_dotenv(args.env_file)
    else:
        _load_env()

    input_w = int(_env("INPUT_WIDTH", "320"))
    input_h = int(_env("INPUT_HEIGHT", "320"))

    base_dataset_pth = _env(
        "BASE_DATASET_PTH",
        "/home/klymy/data/cam4_revB/yolo_dataset/batch1_3_margin_15_wb_alif_clahe",
    )
    subset_name = _env("SUBSET_NAME", "holdout")
    imgs_pth = os.path.join(base_dataset_pth, "images", subset_name)
    labels_pth = os.path.join(base_dataset_pth, "labels", subset_name)
    dataset_pth = _env("DATASET_PTH", os.path.join(base_dataset_pth, "project_holdout.yaml"))

    base_experiment_dir = _env(
        "BASE_EXPERIMENT_DIR",
        "../results/cam3_revB_batch1_12_margin_15_wb_alif_clahe_box_01_dfl_01_2nd",
    )
    eval_backend = (_env("EVAL_BACKEND", "ultralytics_pt") or "ultralytics_pt").strip().lower()
    base_weights_dir = os.path.join(base_experiment_dir, "weights")
    pt_weights_default = os.path.join(base_weights_dir, "best.pt")
    tflite_backbone_only_default = os.path.join(
        base_weights_dir, "best_non_separated_saved_model", "best_full_integer_quant.tflite"
    )
    tflite_mixed_backbone_default = os.path.join(
        base_weights_dir, "best_saved_model", "best_full_integer_quant.tflite"
    )
    tflite_mixed_head_default = os.path.join(base_weights_dir, "head_saved_model", "head_float32.tflite")

    weights_pth = _env("WEIGHTS_PTH", pt_weights_default)
    use_inference_params = _parse_bool_env("USE_INFERENCE_PARAMS", default=True)
    enable_sweep = _parse_bool_env("ENABLE_SWEEP", default=False)
    selection_metric = _normalize_selection_metric(_env("SELECTION_METRIC", "f1"))

    if use_inference_params and os.path.exists(os.path.join(base_experiment_dir, "inference_params.json")) and not enable_sweep:
        with open(os.path.join(base_experiment_dir, "inference_params.json"), "r") as handle:
            inference_params = json.load(handle)
        base_conf_th = float(inference_params["conf_th"])
        base_iou_th = float(inference_params["iou_th"])
        print(
            "Using thresholds from inference_params.json for single-run mode: "
            f"conf={base_conf_th:.4f}, iou={base_iou_th:.4f}"
        )
    else:
        base_conf_th = float(_env("CONF_TH", "0.35"))
        base_iou_th = float(_env("IOU_TH", "0.50"))

    project_dir = _env("PROJECT", os.path.join(os.getcwd(), "val"))
    base_project_name = _env("PROJECT_NAME", f"{os.path.basename(base_experiment_dir)}__{eval_backend}")
    project_pth = _env("PROJECT_PTH", os.path.join(project_dir, base_project_name))
    predictions_pth = _env("PREDICTIONS_PTH", os.path.join(project_pth, "predictions.json"))
    global_metrics_pth = args.out_global_json or _env("GLOBAL_METRICS_PTH", os.path.join(project_pth, "global_metrics.json"))

    detections_attr_csv = _resolve_path(
        _env(
            "DETECTIONS_ATTRS",
            "people_detection_pg2/extended_evaluation_metrics/holdout_data/detections_attributes.csv",
        )
    )
    out_csv = args.out_csv or _env("METRICS_PER_IMG", os.path.join(project_pth, "metrics_per_img.csv"))
    out_gt_csv = args.out_gt_csv or _env("METRICS_PER_GT", os.path.join(project_pth, "metrics_per_gt.csv"))
    out_pred_csv = args.out_pred_csv or _env("METRICS_PER_PRED", os.path.join(project_pth, "metrics_per_pred.csv"))
    sweep_results_pth = _env("SWEEP_RESULTS_PTH", os.path.join(project_pth, "sweep_results.csv"))
    best_params_pth = _env("BEST_PARAMS_PTH", os.path.join(project_pth, "best_params.json"))

    if eval_backend == "tflite_mixed":
        tflite_backbone_pth = _env("TFLITE_BACKBONE_PTH", tflite_mixed_backbone_default)
        tflite_head_pth = _env("TFLITE_HEAD_PTH", tflite_mixed_head_default)
    elif eval_backend == "tflite_backbone_only":
        tflite_backbone_pth = _env("TFLITE_BACKBONE_PTH", tflite_backbone_only_default)
        tflite_head_pth = ""
    else:
        tflite_backbone_pth = _env("TFLITE_BACKBONE_PTH", "")
        tflite_head_pth = _env("TFLITE_HEAD_PTH", "")
    tflite_output_correction_model_pth = _env("TFLITE_OUTPUT_CORRECTION_MODEL_PTH", "")

    device = _parse_device(_env("DEVICE", None))
    batch = int(_env("BATCH", "64"))

    sweep_conf_ths = _parse_float_list(_env("SWEEP_CONF_THS", ""))
    sweep_iou_ths = _parse_float_list(_env("SWEEP_IOU_THS", ""))
    eval_plan = _build_eval_plan(
        base_project_name=base_project_name,
        base_conf_th=base_conf_th,
        base_iou_th=base_iou_th,
        enable_sweep=enable_sweep,
        sweep_conf_ths=sweep_conf_ths,
        sweep_iou_ths=sweep_iou_ths,
    )

    all_img_ids, gt_by_img, img_sizes = _load_gt_and_images(imgs_pth=imgs_pth, labels_pth=labels_pth)
    det_attrs = _load_det_attrs(detections_attr_csv)

    model = YOLO(weights_pth, task="pose")
    project_dir_for_runs = os.path.join(project_pth, "runs")

    best_candidate: dict[str, Any] | None = None
    sweep_rows: list[dict[str, Any]] = []

    for params in eval_plan:
        predictions_run_path, validator_global_metrics, run_dir = _run_validation_for_params(
            model=model,
            dataset_pth=dataset_pth,
            device=device,
            batch=batch,
            input_w=input_w,
            params=params,
            project_dir=project_dir_for_runs,
            eval_backend=eval_backend,
            weights_pth=weights_pth,
            tflite_backbone_pth=tflite_backbone_pth,
            tflite_head_pth=tflite_head_pth,
            tflite_output_correction_model_pth=tflite_output_correction_model_pth,
        )

        with open(predictions_run_path, "rt") as handle:
            predictions = json.load(handle)
        predictions_by_id = _normalize_predictions(predictions, imgs_pth=imgs_pth)

        metrics_df, gt_df, pred_df, local_summary = _evaluate_predictions(
            predictions_by_id=predictions_by_id,
            all_img_ids=all_img_ids,
            gt_by_img=gt_by_img,
            img_sizes=img_sizes,
            input_w=input_w,
            input_h=input_h,
            conf_th=params.conf_th,
            match_iou_th=params.iou_th,
            det_attrs=det_attrs,
        )

        combined_summary: dict[str, Any] = dict(validator_global_metrics)
        combined_summary.update(local_summary)
        combined_summary.update(_derive_selection_summary(combined_summary))
        combined_summary.update(
            {
                "conf_th": params.conf_th,
                "iou_th": params.iou_th,
                "run_name": params.run_name,
                "run_dir": run_dir,
                "selection_metric": selection_metric,
                "selected_metric_value": float(combined_summary[_selection_metric_key(selection_metric)]),
            }
        )
        sweep_rows.append(combined_summary)

        candidate = {
            "summary": combined_summary,
            "metrics_df": metrics_df,
            "gt_df": gt_df,
            "pred_df": pred_df,
            "predictions_run_path": predictions_run_path,
        }
        if best_candidate is None or _build_best_sort_key(combined_summary, selection_metric) > _build_best_sort_key(
            best_candidate["summary"], selection_metric
        ):
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("No evaluation candidates were executed.")

    sweep_df = pd.DataFrame(sweep_rows)
    if not sweep_df.empty:
        sort_columns = [
            _selection_metric_key(selection_metric),
            "selection/f1",
            "selection/recall",
            "selection/precision",
            "conf_th",
            "iou_th",
        ]
        ascending = [False] * len(sort_columns)
        sweep_df = sweep_df.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        _ensure_parent(sweep_results_pth)
        sweep_df.to_csv(sweep_results_pth, index=False)
        print(f"Wrote sweep results: {sweep_results_pth}")

    best_summary = dict(best_candidate["summary"])
    best_summary["num_candidates"] = len(eval_plan)
    best_summary["mode"] = "sweep" if enable_sweep else "single_run"

    _ensure_parent(best_params_pth)
    with open(best_params_pth, "w") as handle:
        json.dump(best_summary, handle, indent=2)
    print(f"Wrote best params summary: {best_params_pth}")

    _copy_file(best_candidate["predictions_run_path"], predictions_pth)
    print(f"Copied best predictions to: {predictions_pth}")

    _write_outputs(
        metrics_df=best_candidate["metrics_df"],
        gt_df=best_candidate["gt_df"],
        pred_df=best_candidate["pred_df"],
        metrics_csv_path=out_csv,
        gt_csv_path=out_gt_csv,
        pred_csv_path=out_pred_csv,
    )

    _ensure_parent(global_metrics_pth)
    with open(global_metrics_pth, "w") as handle:
        json.dump(best_summary, handle, indent=2)
    print(f"Wrote global metrics: {global_metrics_pth}")

    print(
        "Selected best parameters: "
        f"conf={best_summary['conf_th']:.4f}, "
        f"iou={best_summary['iou_th']:.4f}, "
        f"{selection_metric}={best_summary['selected_metric_value']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
