from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None
try:
    from IPython.display import display, Image, Markdown
except Exception:  # pragma: no cover
    def display(x=None):
        print(x)

    class Image:  # noqa: D401
        """Fallback placeholder when IPython is unavailable."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def Markdown(text: str):
        return text

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False

# ===== CONFIG =====
env_path = os.getenv("ENV_PATH", ".env")
load_dotenv(env_path)
if os.getenv("REPORT_UTILS_VERBOSE", "0") == "1":
    print(f"Loaded env from: {env_path}")

BASE = Path(os.getenv("REPORT_BASE", "."))
HOLDOUT = BASE / "holdout_data"
VIZ_DIR = Path(os.getenv("VIZ_DIR", str(BASE / "val" / "viz")))
METRICS_IMG = Path(os.getenv("METRICS_PER_IMG", str(BASE / "metrics_per_img.csv")))
FRAMES_ATTRS = Path(os.getenv("FRAMES_ATTRS", str(HOLDOUT / "frames_attributes.csv")))
GT_METRICS = Path(os.getenv("METRICS_PER_GT", str(BASE / "metrics_per_gt.csv")))
PRED_METRICS = Path(os.getenv("METRICS_PER_PRED", str(BASE / "metrics_per_pred.csv")))
DET_ATTRS = Path(os.getenv("DETECTIONS_ATTRS", str(HOLDOUT / "detections_attributes.csv")))
GLOBAL_METRICS = Path(os.getenv("GLOBAL_METRICS_PTH", str(BASE / "val" / "global_metrics.json")))
BEST_PARAMS = Path(os.getenv("BEST_PARAMS_PTH", str(BASE / "val" / "best_params.json")))

TOP_N = int(os.getenv("TOP_N", "20"))
F1_BINS = [float(x) for x in os.getenv("F1_BINS", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0").split(",")]

# ===== LOADERS =====

def load_data():
    metrics_img = pd.read_csv(METRICS_IMG)
    frames = pd.read_csv(FRAMES_ATTRS) if FRAMES_ATTRS.exists() else pd.DataFrame()
    gt = pd.read_csv(GT_METRICS) if GT_METRICS.exists() else None
    pred = pd.read_csv(PRED_METRICS) if PRED_METRICS.exists() else None
    det_attrs = pd.read_csv(DET_ATTRS) if DET_ATTRS.exists() else None
    return metrics_img, frames, gt, pred, det_attrs

# ===== SECTION A =====

def section_a_global_metrics():
    gm = None
    if GLOBAL_METRICS.exists():
        with open(GLOBAL_METRICS) as f:
            gm = json.load(f)
    best_params = None
    if BEST_PARAMS.exists():
        with open(BEST_PARAMS) as f:
            best_params = json.load(f)
    if gm is None:
        print('Global metrics not found; falling back to per-image mean')
        metrics_img = pd.read_csv(METRICS_IMG)
        display(metrics_img[['P','R','F1','AP']].mean().to_frame('mean'))
        if best_params is not None:
            best_df = pd.DataFrame(
                [
                    {
                        "selection_metric": best_params.get("selection_metric", "f1"),
                        "selected_metric_value": best_params.get("selected_metric_value"),
                        "selection_source": best_params.get("selection_source"),
                        "conf_th": best_params.get("conf_th"),
                        "iou_th": best_params.get("iou_th"),
                        "num_candidates": best_params.get("num_candidates"),
                        "mode": best_params.get("mode"),
                    }
                ]
            )
            display(best_df)
        return
    def _get(k):
        return gm.get(k, None)
    def _f1(p, r):
        if p is None or r is None: return None
        return 2 * p * r / (p + r + 1e-9)
    box_p = _get('metrics/precision(B)')
    box_r = _get('metrics/recall(B)')
    pos_p = _get('metrics/precision(P)')
    pos_r = _get('metrics/recall(P)')
    data = {
        'box': {'Precision': box_p, 'Recall': box_r, 'F1': _f1(box_p, box_r), 'mAP': _get('metrics/mAP50-95(B)')},
        'pos': {'Precision': pos_p, 'Recall': pos_r, 'F1': _f1(pos_p, pos_r), 'mAP': _get('metrics/mAP50-95(P)')},
    }
    df = pd.DataFrame(data).T
    display(df.round(3))
    if best_params is not None:
        best_df = pd.DataFrame(
            [
                {
                    "selection_metric": best_params.get("selection_metric", "f1"),
                    "selected_metric_value": best_params.get("selected_metric_value"),
                    "selection_source": best_params.get("selection_source"),
                    "conf_th": best_params.get("conf_th"),
                    "iou_th": best_params.get("iou_th"),
                    "num_candidates": best_params.get("num_candidates"),
                    "mode": best_params.get("mode"),
                }
            ]
        )
        display(best_df.round(4))

# ===== SECTION B =====

def section_b_f1_distribution():
    metrics_img, _, _, _, _ = load_data()
    if px is not None:
        fig = px.histogram(metrics_img, x="F1", nbins=20, title="Frame-level F1 distribution")
        fig.show()
    else:
        plt.figure(figsize=(8, 4))
        plt.hist(metrics_img["F1"].dropna(), bins=20)
        plt.title("Frame-level F1 distribution")
        plt.xlabel("F1")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

    bins = np.array(F1_BINS)
    labels = [f'({bins[i]:.1f}, {bins[i+1]:.1f}]' for i in range(len(bins)-1)]
    metrics_img['F1_bin'] = pd.cut(metrics_img['F1'], bins=bins, labels=labels, include_lowest=True)
    summary = metrics_img['F1_bin'].value_counts().sort_index().to_frame('count')
    summary['percent'] = summary['count'] / summary['count'].sum() * 100
    display(summary)

# ===== SECTION C =====

def section_c_top_n():
    metrics_img, frames, _, _, _ = load_data()
    metrics_with_frames = metrics_img.copy()
    metrics_with_frames['yolo_id'] = metrics_with_frames['img_id'].apply(lambda s: str(Path(str(s)).with_suffix('')).replace('/', '____'))
    metrics_with_frames = metrics_with_frames.merge(frames, on='yolo_id', how='left')
    worst = metrics_with_frames.sort_values('F1').head(TOP_N)
    display(worst[['img_id','P','R','F1','camera_height','num_gt_detections','scene_density']].head(TOP_N))

    imgs = [VIZ_DIR / img_id for img_id in worst['img_id'].head(TOP_N)]
    imgs = [p for p in imgs if p.exists()]
    m = int(np.ceil(len(imgs) / 2))
    fig, axes = plt.subplots(m, 2, figsize=(12, 6*m))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax, p in zip(axes, imgs):
        ax.imshow(plt.imread(str(p)))
        ax.set_title(p.name, fontsize=8)
        ax.axis('off')
    for ax in axes[len(imgs):]:
        ax.axis('off')
    plt.tight_layout()

# ===== SECTION D =====

def section_d1_gt_recall():
    _, _, gt, _, det_attrs = load_data()
    if det_attrs is None or gt is None:
        print('Missing gt/detections attributes for attribute breakdown')
        return
    gt_m = gt.copy()
    if 'posture' not in gt_m.columns:
        gt_m = gt.merge(det_attrs, left_on=['yolo_id','gt_det_index'], right_on=['yolo_id','det_index'], how='left', suffixes=('', '_det'))

    def _col(df, name):
        if name in df.columns: return name
        if f'{name}_det' in df.columns: return f'{name}_det'
        if f'{name}_x' in df.columns: return f'{name}_x'
        if f'{name}_y' in df.columns: return f'{name}_y'
        return None
    posture_col = _col(gt_m, 'posture')
    occ_obj_col = _col(gt_m, 'occluded_by_object')
    occ_per_col = _col(gt_m, 'occluded_by_person')

    def _recall_table(df, attr_col):
        if not attr_col: return None
        recall = df.groupby(attr_col)['match_type'].apply(lambda s: (s=='TP').mean())
        return recall.to_frame('recall')

    if posture_col: display(_recall_table(gt_m, posture_col))
    else: print('posture column not found')
    if occ_obj_col: display(_recall_table(gt_m, occ_obj_col))
    else: print('occluded_by_object column not found')
    if occ_per_col: display(_recall_table(gt_m, occ_per_col))
    else: print('occluded_by_person column not found')

    if posture_col and occ_obj_col and occ_per_col:
        gt_c = gt_m.copy()
        gt_c = gt_c[gt_c[posture_col].notna() & gt_c[occ_obj_col].notna() & gt_c[occ_per_col].notna()]
        comb = gt_c.groupby([posture_col, occ_obj_col, occ_per_col])['match_type'].apply(lambda s: (s=='TP').mean())
        display(comb.to_frame('recall'))
    else:
        print('One or more columns missing for combined recall table')

# ===== SECTION E =====

def section_e_fn_composition():
    _, _, gt, _, det_attrs = load_data()
    if gt is None:
        print('Missing gt/detections attributes for error composition')
        return
    fn = gt[gt['match_type']=='FN']
    if det_attrs is not None:
        fn_m = fn.copy()
        if 'posture' not in fn_m.columns:
            fn_m = fn.merge(det_attrs, left_on=['yolo_id','gt_det_index'], right_on=['yolo_id','det_index'], how='left', suffixes=('', '_det'))

        def _col(df, name):
            if name in df.columns: return name
            if f'{name}_det' in df.columns: return f'{name}_det'
            if f'{name}_x' in df.columns: return f'{name}_x'
            if f'{name}_y' in df.columns: return f'{name}_y'
            return None
        posture_col = _col(fn_m, 'posture')
        occ_obj_col = _col(fn_m, 'occluded_by_object')
        occ_per_col = _col(fn_m, 'occluded_by_person')

        def _count_by(df, cols):
            df = df.copy()
            for col in cols:
                df = df[df[col].notna()]
            return df.groupby(cols).size().to_frame('fn_count')

        if posture_col: display(_count_by(fn_m, [posture_col]))
        else: print('posture column not found')
        if occ_per_col: display(_count_by(fn_m, [occ_per_col]))
        else: print('occluded_by_person column not found')
        if occ_obj_col: display(_count_by(fn_m, [occ_obj_col]))
        else: print('occluded_by_object column not found')
        if posture_col and occ_obj_col and occ_per_col:
            display(_count_by(fn_m, [posture_col, occ_obj_col, occ_per_col]))
        else:
            print('One or more columns missing for combined table')

# ===== SECTION F =====

def section_f_visual_summary():
    metrics_img, frames, gt, pred, _ = load_data()
    if 'camera_height' in frames.columns:
        metrics_with_frames = metrics_img.copy()
        metrics_with_frames['yolo_id'] = metrics_with_frames['img_id'].apply(lambda s: str(Path(str(s)).with_suffix('')).replace('/', '____'))
        h = metrics_with_frames.merge(frames, on='yolo_id', how='left')
        if px is not None:
            fig = px.scatter(h, x="camera_height", y="F1", title="F1 vs camera height")
            fig.show()
        else:
            plt.figure(figsize=(7, 4))
            plt.scatter(h["camera_height"], h["F1"], s=12)
            plt.title("F1 vs camera height")
            plt.xlabel("camera_height")
            plt.ylabel("F1")
            plt.tight_layout()
            plt.show()
        if 'num_gt_detections' in h.columns:
            if px is not None:
                fig = px.scatter(h, x="num_gt_detections", y="F1", title="Num GT detections vs F1")
                fig.show()
            else:
                plt.figure(figsize=(7, 4))
                plt.scatter(h["num_gt_detections"], h["F1"], s=12)
                plt.title("Num GT detections vs F1")
                plt.xlabel("num_gt_detections")
                plt.ylabel("F1")
                plt.tight_layout()
                plt.show()

    if gt is not None and pred is not None:
        if 'leg_distance_to_center' in gt.columns:
            fn = gt[gt['match_type']=='FN']
            fp = pred[pred['match_type']=='FP']
            if 'pred_x' in pred.columns and 'pred_y' in pred.columns:
                fp = fp.copy()
                fp['leg_distance_to_center'] = ((fp['pred_x']-0.5)**2 + (fp['pred_y']-0.5)**2) ** 0.5
            err = pd.concat([fn[['leg_distance_to_center']], fp[['leg_distance_to_center']]], ignore_index=True)
            err = err.dropna()
            if not err.empty:
                err['dist_bin'] = pd.cut(err['leg_distance_to_center'], bins=np.linspace(0, 1, 11))
                counts = err['dist_bin'].value_counts().sort_index().to_frame('num_errors')
                counts = counts.reset_index()
                counts['dist_bin'] = counts['dist_bin'].astype(str)
                display(counts)
                if px is not None:
                    fig = px.bar(counts, x="dist_bin", y="num_errors", title="Distance from center vs num errors")
                    fig.show()
                else:
                    plt.figure(figsize=(10, 4))
                    plt.bar(counts["dist_bin"], counts["num_errors"])
                    plt.title("Distance from center vs num errors")
                    plt.xlabel("dist_bin")
                    plt.ylabel("num_errors")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    plt.show()


# ===== SECTION G =====

def section_g_conclusions():
    """
    Prints a short report-driven conclusion block:
    - strengths
    - weaknesses
    - overall conclusion
    - recommendations for next iteration
    """
    metrics_img, _frames, gt, pred, _det_attrs = load_data()

    # Global (YOLO) metrics
    gm = None
    if GLOBAL_METRICS.exists():
        with open(GLOBAL_METRICS) as f:
            gm = json.load(f)

    def _f1(p, r):
        if p is None or r is None:
            return None
        return float(2 * p * r / (p + r + 1e-9))

    box_p = (gm or {}).get("metrics/precision(B)")
    box_r = (gm or {}).get("metrics/recall(B)")
    pos_p = (gm or {}).get("metrics/precision(P)")
    pos_r = (gm or {}).get("metrics/recall(P)")
    box_f1 = _f1(box_p, box_r) if gm else None
    pos_f1 = _f1(pos_p, pos_r) if gm else None

    # Frame-level tail
    median_f1 = float(metrics_img["F1"].median()) if "F1" in metrics_img.columns else None
    pct_f1_lt_09 = float((metrics_img["F1"] < 0.9).mean()) if "F1" in metrics_img.columns else None
    pct_f1_lt_05 = float((metrics_img["F1"] < 0.5).mean()) if "F1" in metrics_img.columns else None

    # Detection-level
    tp = fn = fp = None
    det_precision = det_recall = None
    if gt is not None and pred is not None and "match_type" in gt.columns and "match_type" in pred.columns:
        tp = int((gt["match_type"] == "TP").sum())
        fn = int((gt["match_type"] == "FN").sum())
        fp = int((pred["match_type"] == "FP").sum())
        det_precision = tp / (tp + fp + 1e-9)
        det_recall = tp / (tp + fn + 1e-9)

    # FN patterns (GT-only)
    fn_edge_pct = fn_occ_obj_pct = None
    worst_posture = worst_posture_recall = None
    if gt is not None and "match_type" in gt.columns:
        fn_df = gt[gt["match_type"] == "FN"].copy()
        if len(fn_df) > 0:
            if "position" in fn_df.columns:
                fn_edge_pct = float((fn_df["position"] == "edge").mean())
            if "occluded_by_object" in fn_df.columns:
                fn_occ_obj_pct = float((fn_df["occluded_by_object"] == True).mean())  # noqa: E712

        if "posture" in gt.columns:
            posture_grp = gt.groupby(gt["posture"].fillna("None"))
            stats = pd.DataFrame(
                {
                    "tp": posture_grp.apply(lambda x: int((x["match_type"] == "TP").sum())),
                    "fn": posture_grp.apply(lambda x: int((x["match_type"] == "FN").sum())),
                }
            )
            stats["support"] = stats["tp"] + stats["fn"]
            stats["recall"] = stats["tp"] / (stats["support"] + 1e-9)
            stats2 = stats[stats["support"] >= 10].sort_values(["recall", "support"], ascending=[True, False])
            if not stats2.empty:
                worst_posture = str(stats2.index[0])
                worst_posture_recall = float(stats2.iloc[0]["recall"])

    def _fmt(x):
        if x is None:
            return "N/A"
        return f"{float(x):.3f}"

    def _pct(x):
        if x is None:
            return "N/A"
        return f"{100*float(x):.1f}%"

    lines = []
    lines.append("## Section G — Conclusions & recommendations")
    lines.append("")
    lines.append("### Strengths")
    if gm:
        lines.append(
            f"- Strong global performance: **box** P={_fmt(box_p)}, R={_fmt(box_r)}, F1={_fmt(box_f1)}; "
            f"**pose** P={_fmt(pos_p)}, R={_fmt(pos_r)}, F1={_fmt(pos_f1)}."
        )
    if median_f1 is not None:
        lines.append(
            f"- Most frames are near-perfect: median frame F1={_fmt(median_f1)} "
            f"(frames with F1<0.9: {_pct(pct_f1_lt_09)}, F1<0.5: {_pct(pct_f1_lt_05)})."
        )

    lines.append("")
    lines.append("### Weaknesses")
    if det_precision is not None and det_recall is not None:
        lines.append(
            f"- Remaining errors are concentrated in hard cases: TP={tp}, FN={fn}, FP={fp} "
            f"(precision={_fmt(det_precision)}, recall={_fmt(det_recall)})."
        )
    parts = []
    if fn_edge_pct is not None:
        parts.append(f"{_pct(fn_edge_pct)} of FN at the **image edge**")
    if fn_occ_obj_pct is not None:
        parts.append(f"{_pct(fn_occ_obj_pct)} of FN **occluded by objects**")
    if parts:
        lines.append(f"- FN patterns: {', '.join(parts)}.")
    if worst_posture is not None:
        lines.append(
            f"- Worst recall posture slice (support≥10): **{worst_posture}** (recall={_fmt(worst_posture_recall)})."
        )

    lines.append("")
    lines.append("### Overall conclusion")
    lines.append(
        "- The model is strong on typical frames, but misses cluster around **partial visibility** (edge truncation, "
        "object occlusion) and certain **posture** patterns."
    )

    lines.append("")
    lines.append("### Recommendations (next data/annotation iteration)")
    lines.append(
        "- **Data collection:** oversample people at frame borders and behind common occluders (tables/counters/poles)."
    )
    lines.append(
        "- **Annotation/QC:** enforce consistent posture/occlusion labels and verify leg/foot point correctness."
    )
    lines.append(
        "- **Training:** emphasize truncation/occlusion augmentations (crop/pad, cutout/erase) + review worst-frame FP backgrounds for hard negatives."
    )

    display(Markdown("\n".join(lines)))
