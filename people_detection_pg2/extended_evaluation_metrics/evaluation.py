#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "objectdetectionmetrics"))
from metricsevaluator import MetricsEvaluator


def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None else default


def _parse_device(value: str | None):
    if not value:
        return None
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip() != ""]
    try:
        return int(value)
    except ValueError:
        return value


def _yolo_id_from_img_id(img_id: str) -> str:
    return str(Path(img_id).with_suffix("")).replace("/", "____")


def _resolve_path(p: str) -> str:
    if not p:
        return p
    path = Path(p)
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

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"cv2.imread returned None for {img_path}")
        h, w = img.shape[:2]
        return h, w
    except Exception:
        from PIL import Image

        with Image.open(img_path) as im:
            w, h = im.size
        return h, w


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
    matches = []
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Run YOLO validation and per-frame metrics export.")
    ap.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file with evaluation parameters.",
    )
    ap.add_argument(
        "--out-csv",
        default="metrics_per_img.csv",
        help="Output CSV path (default: %(default)s)",
    )
    ap.add_argument(
        "--out-gt-csv",
        default="metrics_per_gt.csv",
        help="Optional GT-level CSV path (default: from env METRICS_PER_GT_CSV or none).",
    )
    ap.add_argument(
        "--out-pred-csv",
        default="metrics_per_pred.csv",
        help="Optional prediction-level CSV path (default: from env METRICS_PER_PRED_CSV or none).",
    )
    args, _unknown = ap.parse_known_args()

    load_dotenv(args.env_file)

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
    base_weights_dir = os.path.join(base_experiment_dir, "weights")
    weights_pth = _env("WEIGHTS_PTH", os.path.join(base_weights_dir, "best.pt"))

    if os.path.exists(os.path.join(base_experiment_dir, "inference_params.json")):
        with open(os.path.join(base_experiment_dir, "inference_params.json"), "r") as f:
            data = json.load(f)
        tuned_conf = float(data["conf_th"])
        tuned_iou = float(data["iou_th"])
    else:
        tuned_conf = float(_env("CONF_TH", "0.35"))
        tuned_iou = float(_env("IOU_TH", "0.50"))

    project_dir = _env("PROJECT", os.path.join(os.getcwd(), "val"))
    exp_name = _env("EXP_NAME", os.path.basename(base_experiment_dir))
    predictions_pth = _env("PREDICTIONS_PTH", os.path.join(project_dir, exp_name, "predictions.json"))

    detections_attr_csv = _env(
        "DETECTIONS_ATTRIBUTES_CSV",
        "people_detection_pg2/extended_evaluation_metrics/holdout_data/detections_attributes.csv",
    )
    detections_attr_csv = _resolve_path(detections_attr_csv) if detections_attr_csv else detections_attr_csv
    out_gt_csv = args.out_gt_csv or _env("METRICS_PER_GT_CSV", "")
    out_pred_csv = args.out_pred_csv or _env("METRICS_PER_PRED_CSV", "")

    device = _parse_device(_env("DEVICE", None))
    batch = int(_env("BATCH", "64"))

    model = YOLO(weights_pth, task="pose")

    if not os.path.exists(predictions_pth):
        model.val(
            data=dataset_pth,
            task="pose",
            device=device,
            batch=batch,
            verbose=True,
            conf=tuned_conf,
            iou=tuned_iou,
            project=project_dir,
            name=exp_name,
            exist_ok=True,
            save_json=True,
        )

    with open(predictions_pth, "rt") as f:
        predictions = json.load(f)

    predictions_by_id: dict[str, list] = {}
    for pred in predictions:
        img_id = pred["image_id"]
        predictions_by_id.setdefault(img_id, []).append(pred)

    frames_attrs = None

    metrics_rows = []
    img_sizes: dict[str, tuple[int, int]] = {}

    for img_id, preds in tqdm(predictions_by_id.items(), total=len(predictions_by_id)):
        evaluator = MetricsEvaluator(
            is_relative_coordinates=True,
            is_xywh_coordinates=True,
            img_size=(input_w, input_h),
            outpath=".",
        )

        img_size = img_sizes.get(img_id)
        if img_size is None:
            img_h, img_w = _get_image_size(os.path.join(imgs_pth, img_id))
            img_sizes[img_id] = (img_h, img_w)
        else:
            img_h, img_w = img_size

        label_pth = os.path.join(labels_pth, img_id.replace(".jpg", ".txt"))
        with open(label_pth, "rt") as f:
            lines = f.readlines()
        for line in lines:
            parts = [float(i) for i in line.strip().split()]
            if len(parts) < 5:
                continue
            c, x_c, y_c, w, h = parts[:5]
            if c > 1 or np.isnan(x_c):
                continue
            evaluator.add_gt_box(
                img_path=img_id,
                class_name="person",
                bbox=[x_c, y_c, w, h],
            )

        for prediction in preds:
            x, y, w, h = np.array(prediction["bbox"]) / [img_w, img_h, img_w, img_h]
            evaluator.add_det_box(
                img_path=img_id,
                class_name="person",
                bbox=[x + w / 2, y + h / 2, w, h],
                confidence=prediction["score"],
            )

        result = evaluator.eval_conf_vals(0.5)[0]
        tp = result["total TP"]
        fp = result["total FP"]
        total_pos = result["total positives"]

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (total_pos + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        row = {
            "img_id": img_id,
            "P": precision,
            "R": recall,
            "F1": f1,
            "AP": result["AP"],
        }

        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.fillna(0.0, inplace=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.out_csv, index=False)

    print(f"Wrote per-image metrics: {args.out_csv}")
    print(metrics_df[["P", "R", "F1", "AP"]].mean())

    if out_gt_csv or out_pred_csv:
        det_attrs = None
        if detections_attr_csv and os.path.exists(detections_attr_csv):
            det_attrs = pd.read_csv(detections_attr_csv)

        gt_rows = []
        pred_rows = []
        for img_id, preds in tqdm(predictions_by_id.items(), total=len(predictions_by_id), desc="Det-level"):
            yolo_id = _yolo_id_from_img_id(img_id)
            label_pth = os.path.join(labels_pth, img_id.replace(".jpg", ".txt"))
            if not os.path.exists(label_pth):
                continue

            img_h, img_w = img_sizes.get(img_id) or _get_image_size(os.path.join(imgs_pth, img_id))

            gts = []
            with open(label_pth, "rt") as f:
                for idx, line in enumerate(f):
                    parts = [float(i) for i in line.strip().split()]
                    if len(parts) < 5:
                        continue
                    c, x_c, y_c, w, h = parts[:5]
                    if c > 1 or np.isnan(x_c):
                        continue
                    gts.append({"gt_index": idx, "x_c": x_c, "y_c": y_c, "w": w, "h": h})

            pred_boxes = []
            for p in preds:
                x, y, w, h = np.array(p["bbox"])
                kp = p.get("keypoints") or []
                if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    pred_x = float(kp[0]) / img_w
                    pred_y = float(kp[1]) / img_h
                else:
                    pred_x = None
                    pred_y = None
                pred_boxes.append(
                    {
                        "pred_index": len(pred_boxes),
                        "score": float(p["score"]),
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

            gt_boxes = []
            for gt in gts:
                x1, y1, x2, y2 = _xywh_to_xyxy(gt["x_c"], gt["y_c"], gt["w"], gt["h"])
                gt_boxes.append(
                    {
                        "gt_index": gt["gt_index"],
                        "x1": x1 * img_w,
                        "y1": y1 * img_h,
                        "x2": x2 * img_w,
                        "y2": y2 * img_h,
                        "x_c": gt["x_c"],
                        "y_c": gt["y_c"],
                        "w": gt["w"],
                        "h": gt["h"],
                    }
                )

            iou_mat = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
            for i, p in enumerate(pred_boxes):
                for j, g in enumerate(gt_boxes):
                    iou_mat[i, j] = _iou_xyxy((p["x1"], p["y1"], p["x2"], p["y2"]),
                                              (g["x1"], g["y1"], g["x2"], g["y2"]))

            matches = _greedy_match(iou_mat, float(tuned_iou))
            matched_pred = {m[0] for m in matches}
            matched_gt = {m[1] for m in matches}

            def _gt_attrs(gt_index: int):
                if det_attrs is None:
                    return {}
                sub = det_attrs[(det_attrs["yolo_id"] == yolo_id) & (det_attrs["det_index"] == gt_index)]
                if sub.empty:
                    return {}
                return sub.iloc[0].to_dict()

            for match_id, (pred_i, gt_j, iou) in enumerate(matches):
                p = pred_boxes[pred_i]
                g = gt_boxes[gt_j]
                gt_row = {
                    "img_id": img_id,
                    "yolo_id": yolo_id,
                    "match_type": "TP",
                    "match_id": f"{yolo_id}::m::{match_id}",
                    "iou": iou,
                    "gt_det_index": g["gt_index"],
                    "gt_x_c": g["x_c"],
                    "gt_y_c": g["y_c"],
                    "gt_w": g["w"],
                    "gt_h": g["h"],
                    "matched_pred_index": p["pred_index"],
                }
                gt_row.update(_gt_attrs(g["gt_index"]))
                gt_rows.append(gt_row)

                pred_row = {
                    "img_id": img_id,
                    "yolo_id": yolo_id,
                    "match_type": "TP",
                    "match_id": f"{yolo_id}::m::{match_id}",
                    "iou": iou,
                    "score": p["score"],
                    "pred_det_index": p["pred_index"],
                    "pred_x_c": p["x_c"],
                    "pred_y_c": p["y_c"],
                    "pred_w": p["w"],
                    "pred_h": p["h"],
                    "pred_x": "" if p["x"] is None else p["x"],
                    "pred_y": "" if p["y"] is None else p["y"],
                    "matched_gt_index": g["gt_index"],
                }
                pred_rows.append(pred_row)

            for p in pred_boxes:
                if p["pred_index"] in matched_pred:
                    continue
                pred_row = {
                    "img_id": img_id,
                    "yolo_id": yolo_id,
                    "match_type": "FP",
                    "match_id": "",
                    "iou": "",
                    "score": p["score"],
                    "pred_det_index": p["pred_index"],
                    "pred_x_c": p["x_c"],
                    "pred_y_c": p["y_c"],
                    "pred_w": p["w"],
                    "pred_h": p["h"],
                    "pred_x": "" if p["x"] is None else p["x"],
                    "pred_y": "" if p["y"] is None else p["y"],
                    "matched_gt_index": "",
                }
                pred_rows.append(pred_row)

            for g in gt_boxes:
                if g["gt_index"] in matched_gt:
                    continue
                gt_row = {
                    "img_id": img_id,
                    "yolo_id": yolo_id,
                    "match_type": "FN",
                    "match_id": "",
                    "iou": "",
                    "gt_det_index": g["gt_index"],
                    "gt_x_c": g["x_c"],
                    "gt_y_c": g["y_c"],
                    "gt_w": g["w"],
                    "gt_h": g["h"],
                    "matched_pred_index": "",
                }
                gt_row.update(_gt_attrs(g["gt_index"]))
                gt_rows.append(gt_row)

        if out_gt_csv:
            gt_df = pd.DataFrame(gt_rows)
            Path(out_gt_csv).parent.mkdir(parents=True, exist_ok=True)
            gt_df.to_csv(out_gt_csv, index=False)
            print(f"Wrote GT-level CSV: {out_gt_csv}")
        if out_pred_csv:
            pred_df = pd.DataFrame(pred_rows)
            Path(out_pred_csv).parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(out_pred_csv, index=False)
            print(f"Wrote prediction-level CSV: {out_pred_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
