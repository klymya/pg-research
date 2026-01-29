#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def read_predictions(pred_path: Path):
    preds = json.loads(pred_path.read_text())
    by_id = {}
    for p in preds:
        img_id = p.get("image_id")
        if img_id is None:
            continue
        by_id.setdefault(img_id, []).append(p)
    return by_id


def read_metrics(metrics_path: Path):
    import pandas as pd

    df = pd.read_csv(metrics_path)
    metrics = {}
    for _, row in df.iterrows():
        img_id = row["img_id"]
        metrics[img_id] = {
            "P": float(row.get("P", 0.0)),
            "R": float(row.get("R", 0.0)),
            "F1": float(row.get("F1", 0.0)),
        }
    return metrics


def draw_gt(img_bgr, label_path: Path):
    img = img_bgr.copy()
    h, w = img.shape[:2]
    if not label_path.exists():
        return img
    with label_path.open("rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            _, x_c, y_c, bw, bh = [float(x) for x in parts[:5]]
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if len(parts) >= 7:
                x_kp, y_kp = [float(x) for x in parts[5:7]]
                cv2.circle(img, (int(x_kp * w), int(y_kp * h)), 6, (0, 255, 0), 2)
    return img


def draw_predictions(img_bgr, preds, show_scores=True):
    img = img_bgr.copy()
    for p in preds:
        bbox = p.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, bw, bh = [int(v) for v in bbox[:4]]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        if show_scores and "score" in p:
            score = float(p["score"])
            label = f"{score:.2f}"
            cv2.rectangle(img, (x, y), (x + 60, y + 22), (0, 0, 255), -1)
            cv2.putText(img, label, (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if "keypoints" in p and isinstance(p["keypoints"], (list, tuple)) and len(p["keypoints"]) >= 2:
            kx, ky = p["keypoints"][:2]
            cv2.circle(img, (int(kx), int(ky)), 6, (0, 0, 255), 2)
    return img


def short_model_name(path: Path):
    name = path.name
    m = re.search(r"batch1_\d+", name)
    return m.group(0) if m else name


def main():
    parser = argparse.ArgumentParser(description="Side-by-side comparison of 3 model predictions on holdout set.")
    parser.add_argument(
        "--val-root",
        default="/home/klymy/projects/researches/people_detection_pg2/cam4_revB/val",
        help="Root folder containing model eval subfolders with predictions.json and metrics_per_img.csv",
    )
    parser.add_argument(
        "--models",
        nargs=3,
        default=[
            "cam3_revB_batch1_10_margin_15_rq_bg_wb_alif_clahe_box_01_dfl_01_2nd",
            "cam3_revB_batch1_11_margin_15_rq_wb_alif_clahe_box_01_dfl_01_2nd",
            "cam3_revB_batch1_12_margin_15_wb_alif_clahe_box_01_dfl_01_2nd",
        ],
        help="Three model folder names under --val-root",
    )
    parser.add_argument(
        "--images-dir",
        default="/home/klymy/data/cam4_revB/yolo_dataset/batch1_3_margin_15_wb_alif_clahe/images/holdout",
        help="Holdout images directory",
    )
    parser.add_argument(
        "--labels-dir",
        default="/home/klymy/data/cam4_revB/yolo_dataset/batch1_3_margin_15_wb_alif_clahe/labels/holdout",
        help="Holdout labels directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/klymy/projects/researches/people_detection_pg2/yolo_dataset_review/batch/side_by_side",
        help="Output folder for side-by-side images",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images processed (0 = all)",
    )
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Disable drawing confidence scores on prediction boxes",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [Path(args.val_root) / m for m in args.models]

    pred_by_model = []
    metrics_by_model = []
    model_labels = []

    for mpath in models:
        pred_path = mpath / "predictions.json"
        metrics_path = mpath / "metrics_per_img.csv"
        if not pred_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Missing predictions or metrics in {mpath}")
        pred_by_model.append(read_predictions(pred_path))
        metrics_by_model.append(read_metrics(metrics_path))
        model_labels.append(short_model_name(mpath))

    img_paths = sorted(images_dir.glob("*.jpg"))
    if args.max_images and args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    for img_path in tqdm(img_paths, desc="Rendering"):
        img_id = img_path.name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        gt_img = draw_gt(img_bgr, labels_dir / img_id.replace(".jpg", ".txt"))
        pred_imgs = []
        for p_by_id in pred_by_model:
            pred_imgs.append(draw_predictions(img_bgr, p_by_id.get(img_id, []), show_scores=not args.no_scores))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        axes[0].imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("GT")
        axes[0].axis("off")

        for idx, pred_img in enumerate(pred_imgs):
            metrics = metrics_by_model[idx].get(img_id)
            if metrics:
                title = f"{model_labels[idx]} | F1={metrics['F1']:.2f} P={metrics['P']:.2f} R={metrics['R']:.2f}"
            else:
                title = f"{model_labels[idx]} | F1=NA P=NA R=NA"
            axes[idx + 1].imshow(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))
            axes[idx + 1].set_title(title, fontsize=9)
            axes[idx + 1].axis("off")

        plt.tight_layout()
        out_path = out_dir / img_id
        plt.savefig(out_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
