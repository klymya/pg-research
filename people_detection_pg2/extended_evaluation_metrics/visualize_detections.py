#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from tqdm import tqdm

import matplotlib

# This script renders and saves images; using a GUI backend (TkAgg/QtAgg/...) can be
# dramatically slower depending on the environment. Default to the non-interactive
# Agg backend unless the user explicitly selects a backend via MPLBACKEND.
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from dotenv import load_dotenv


COLOR_GT_TP = "blue"
COLOR_PRED_TP = "green"
COLOR_GT_FN = "red"
COLOR_PRED_FP = "orange"


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _img_size(img) -> tuple[int, int]:
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    return h, w


def _draw_box(ax, x_c, y_c, w, h, img_w, img_h, color, label=None):
    if any(v is None for v in [x_c, y_c, w, h]):
        return
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    rect = patches.Rectangle((x1, y1), w * img_w, h * img_h, linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    if label:
        x2 = (x_c + w / 2) * img_w
        tx = max(x2 - 2, 0)
        ty = max(y1 + 2, 0)
        ax.text(
            tx,
            ty,
            label,
            color=color,
            fontsize=7,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )


def _draw_point(ax, x, y, img_w, img_h, color, label=None):
    if x is None or y is None:
        return
    ax.plot([x * img_w], [y * img_h], marker="o", markersize=5, color=color)
    if label:
        ax.text(
            x * img_w + 3,
            y * img_h + 3,
            label,
            color=color,
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )


def main() -> int:
    env_path = os.getenv("ENV_PATH", ".env")
    load_dotenv(env_path)

    ap = argparse.ArgumentParser(description="Visualize GT/Pred detections with TP/FP/FN coloring.")
    ap.add_argument("--yolo-holdout", default=os.getenv("YOLO_HOLDOUT", ""), help="Path to YOLO dataset root (with images/holdout & labels/holdout).")
    ap.add_argument("--metrics-per-img", default=os.getenv("METRICS_PER_IMG", ""), help="Path to metrics_per_img.csv.")
    ap.add_argument("--metrics-per-gt", default=os.getenv("METRICS_PER_GT", ""), help="Path to metrics_per_gt_detection.csv.")
    ap.add_argument("--metrics-per-pred", default=os.getenv("METRICS_PER_PRED", ""), help="Path to metrics_per_pred_detections.csv.")
    ap.add_argument(
        "--frames-attrs",
        default=os.getenv("FRAMES_ATTRS", "frames_attributes.csv"),
        help="Optional frames_attributes.csv for camera_height (default: %(default)s)",
    )
    # leg position is expected directly in metrics_per_gt CSV (columns x,y)
    ap.add_argument("--out-dir", default=os.getenv("VIZ_OUT_DIR", ""), help="Output folder for rendered images.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of images to render.")
    args, _unknown = ap.parse_known_args()

    if not args.yolo_holdout or not args.metrics_per_img or not args.metrics_per_gt or not args.metrics_per_pred or not args.out_dir:
        raise SystemExit("Missing required paths. Set them in .env or pass as args.")

    yolo_root = Path(args.yolo_holdout)
    imgs_dir = yolo_root / "images" / "holdout"
    labels_dir = yolo_root / "labels" / "holdout"
    if not imgs_dir.exists():
        raise SystemExit(f"Images dir not found: {imgs_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")

    metrics_img = pd.read_csv(args.metrics_per_img)
    metrics_gt = pd.read_csv(args.metrics_per_gt)
    metrics_pred = pd.read_csv(args.metrics_per_pred)

    metrics_img = metrics_img.set_index("img_id")
    gt_by_img = {k: v for k, v in metrics_gt.groupby("img_id")}
    pred_by_img = {k: v for k, v in metrics_pred.groupby("img_id")}

    frames_attrs = None
    if args.frames_attrs and Path(args.frames_attrs).exists():
        frames_attrs = pd.read_csv(args.frames_attrs)
        if "yolo_id" not in frames_attrs.columns and "xml_image_name" in frames_attrs.columns:
            frames_attrs["yolo_id"] = frames_attrs["xml_image_name"].apply(
                lambda s: str(Path(str(s)).with_suffix("")).replace("/", "____")
            )
        if "yolo_id" in frames_attrs.columns:
            frames_attrs = frames_attrs.set_index("yolo_id")

    _ensure_dir(args.out_dir)

    img_ids = list(metrics_img.index)
    if args.limit and args.limit > 0:
        img_ids = img_ids[: args.limit]

    for img_id in tqdm(img_ids, desc="Rendering images"):
        img_path = imgs_dir / img_id
        if not img_path.exists():
            continue
        img = plt.imread(str(img_path))
        img_h, img_w = _img_size(img)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        ax.axis("off")

        gt_df = gt_by_img.get(img_id)
        if gt_df is not None:
            for _, row in gt_df.iterrows():
                if row.get("match_type") == "TP":
                    color = COLOR_GT_TP
                else:
                    color = COLOR_GT_FN
                _draw_box(ax, row.get("gt_x_c"), row.get("gt_y_c"), row.get("gt_w"), row.get("gt_h"), img_w, img_h, color)
                x = row.get("x")
                y = row.get("y")
                if pd.notna(x) and pd.notna(y):
                    ax.plot([x * img_w], [y * img_h], marker="o", markersize=5, color=color)

        pred_df = pred_by_img.get(img_id)
        if pred_df is not None:
            for _, row in pred_df.iterrows():
                if row.get("match_type") == "TP":
                    color = COLOR_PRED_TP
                else:
                    color = COLOR_PRED_FP
                score_label = f"{row.get('score'):.2f}" if pd.notna(row.get("score")) else None
                _draw_box(
                    ax,
                    row.get("pred_x_c"),
                    row.get("pred_y_c"),
                    row.get("pred_w"),
                    row.get("pred_h"),
                    img_w,
                    img_h,
                    color,
                    label=score_label,
                )
                px = row.get("pred_x") if "pred_x" in row else row.get("x")
                py = row.get("pred_y") if "pred_y" in row else row.get("y")
                _draw_point(
                    ax,
                    px,
                    py,
                    img_w,
                    img_h,
                    color,
                    label=None,
                )

        m = metrics_img.loc[img_id]
        yolo_id = str(Path(str(img_id)).with_suffix("")).replace("/", "____")
        height = ""
        if frames_attrs is not None and yolo_id in frames_attrs.index and "camera_height" in frames_attrs.columns:
            height = frames_attrs.loc[yolo_id]["camera_height"]
        title = f"{img_id} | H={height} | P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f}"
        ax.set_title(title, fontsize=9)

        legend_patches = [
            patches.Patch(color=COLOR_GT_TP, label="GT TP"),
            patches.Patch(color=COLOR_PRED_TP, label="Pred TP"),
            patches.Patch(color=COLOR_GT_FN, label="GT FN"),
            patches.Patch(color=COLOR_PRED_FP, label="Pred FP"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.7)

        out_path = Path(args.out_dir) / img_id
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Saved images to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
