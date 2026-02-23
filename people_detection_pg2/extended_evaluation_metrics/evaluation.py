#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import inspect
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

    p = Path(imgs_pth) / raw_image_id
    if p.exists():
        cache[raw_image_id] = raw_image_id
        return raw_image_id

    stem = Path(raw_image_id).stem
    matches = sorted(Path(imgs_pth).glob(f"{stem}.*"))
    if matches:
        rel = matches[0].name
        cache[raw_image_id] = rel
        return rel

    # Fallback for common case from validator where image_id is stem without extension.
    guess = f"{raw_image_id}.jpg"
    cache[raw_image_id] = guess
    return guess


def _list_image_files(imgs_pth: str) -> list[str]:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    p = Path(imgs_pth)
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_file() and x.suffix.lower() in img_exts])


def _run_pt_validation(
    model: YOLO,
    dataset_pth: str,
    device,
    batch: int,
    tuned_conf: float,
    tuned_iou: float,
    project_dir: str,
    project_name: str,
):
    return model.val(
        data=dataset_pth,
        task="pose",
        device=device,
        batch=batch,
        verbose=True,
        conf=tuned_conf,
        iou=tuned_iou,
        project=project_dir,
        name=project_name,
        exist_ok=True,
        save_json=True,
    )


def _build_tflite_inferer(
    backend: str,
    tuned_conf: float,
    tuned_iou: float,
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
        except Exception as e:
            print(f"Warning: failed to import emulator TFLiteInferer ({e}), using local inferer.")

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
        conf_th=tuned_conf,
        iou_th=tuned_iou,
        imgsz=input_w,
        separated_head=False,
        head_int8=False,
    )

    if backend == "tflite_backbone_only" and tflite_output_correction_model_pth:
        sig = inspect.signature(inferer_cls.__init__)
        if "output_correction_model_path" in sig.parameters:
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
    tuned_conf: float,
    tuned_iou: float,
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
        tuned_conf=tuned_conf,
        tuned_iou=tuned_iou,
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
        conf=tuned_conf,
        iou=tuned_iou,
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Run YOLO validation and per-frame metrics export.")
    ap.add_argument(
        "--env-file",
        default="",
        help="(Optional) Path to .env file. If empty, uses ENV_PATH.",
    )
    ap.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path (default: from env METRICS_PER_IMG).",
    )
    ap.add_argument(
        "--out-gt-csv",
        default="",
        help="Optional GT-level CSV path (default: from env METRICS_PER_GT).",
    )
    ap.add_argument(
        "--out-pred-csv",
        default="",
        help="Optional prediction-level CSV path (default: from env METRICS_PER_PRED).",
    )
    ap.add_argument(
        "--out-global-json",
        default="",
        help="Optional global metrics JSON path (default: from env GLOBAL_METRICS_PTH).",
    )
    args, _unknown = ap.parse_known_args()

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
    tflite_mixed_head_default = os.path.join(
        base_weights_dir, "head_saved_model", "head_float32.tflite"
    )

    # Keep WEIGHTS_PTH optional as an override for validator metadata model.
    weights_pth = _env("WEIGHTS_PTH", pt_weights_default)

    if os.path.exists(os.path.join(base_experiment_dir, "inference_params.json")):
        with open(os.path.join(base_experiment_dir, "inference_params.json"), "r") as f:
            data = json.load(f)
        tuned_conf = float(data["conf_th"])
        tuned_iou = float(data["iou_th"])
    else:
        tuned_conf = float(_env("CONF_TH", "0.35"))
        tuned_iou = float(_env("IOU_TH", "0.50"))

    project_dir = _env("PROJECT", os.path.join(os.getcwd(), "val"))
    project_name = _env("PROJECT_NAME", f"{os.path.basename(base_experiment_dir)}__{eval_backend}")
    exp_name = _env("EXP_NAME", os.path.basename(base_experiment_dir))
    predictions_pth = _env("PREDICTIONS_PTH", os.path.join(project_dir, project_name, "predictions.json"))
    global_metrics_pth = args.out_global_json or _env("GLOBAL_METRICS_PTH", "")

    detections_attr_csv = _env(
        "DETECTIONS_ATTRS",
        "people_detection_pg2/extended_evaluation_metrics/holdout_data/detections_attributes.csv",
    )
    detections_attr_csv = _resolve_path(detections_attr_csv) if detections_attr_csv else detections_attr_csv
    out_csv = args.out_csv or _env("METRICS_PER_IMG", "metrics_per_img.csv")
    out_gt_csv = args.out_gt_csv or _env("METRICS_PER_GT", "")
    out_pred_csv = args.out_pred_csv or _env("METRICS_PER_PRED", "")

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

    if not os.path.exists(weights_pth):
        raise FileNotFoundError(
            f"Validator PT weights not found: {weights_pth}. "
            "Set BASE_EXPERIMENT_DIR correctly or override WEIGHTS_PTH."
        )
    model = YOLO(weights_pth, task="pose")

    force_val = _env("FORCE_VAL", "0") == "1"
    print(f"Force validation: {force_val}; backend: {eval_backend}")
    val_result = None
    gm = {}
    if force_val or not os.path.exists(predictions_pth):
        if eval_backend == "ultralytics_pt":
            val_result = _run_pt_validation(
                model=model,
                dataset_pth=dataset_pth,
                device=device,
                batch=batch,
                tuned_conf=tuned_conf,
                tuned_iou=tuned_iou,
                project_dir=project_dir,
                project_name=project_name,
            )
            gm = val_result.results_dict if hasattr(val_result, "results_dict") else {}
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

            gm = _run_tflite_validation(
                model=model,
                dataset_pth=dataset_pth,
                project_dir=project_dir,
                project_name=project_name,
                batch=batch,
                device=device,
                input_w=input_w,
                tuned_conf=tuned_conf,
                tuned_iou=tuned_iou,
                backend=eval_backend,
                tflite_backbone_pth=tflite_backbone_pth,
                tflite_head_pth=tflite_head_pth,
                tflite_output_correction_model_pth=tflite_output_correction_model_pth,
            )
        else:
            raise ValueError(f"Unsupported EVAL_BACKEND: {eval_backend}")

        generated_predictions_pth = os.path.join(project_dir, project_name, "predictions.json")
        if os.path.exists(generated_predictions_pth):
            src = Path(generated_predictions_pth)
            dst = Path(predictions_pth)
            same = False
            try:
                if dst.exists():
                    same = os.path.samefile(src, dst)
                else:
                    same = src.resolve(strict=False) == dst.resolve(strict=False)
            except OSError:
                same = src.resolve(strict=False) == dst.resolve(strict=False)

            if not same:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                print(f"Copied predictions to: {predictions_pth}")

        print(f"Validation completed. Predictions saved to: {predictions_pth}")
    if gm:
        try:
            Path(global_metrics_pth).parent.mkdir(parents=True, exist_ok=True)
            with open(global_metrics_pth, "w") as f:
                json.dump(gm, f, indent=2)
            print(f"Wrote global metrics: {global_metrics_pth}")
        except Exception as e:
            print(f"Warning: failed to save global metrics: {e}")

    with open(predictions_pth, "rt") as f:
        predictions = json.load(f)

    predictions_by_id: dict[str, list] = {}
    image_id_cache: dict[str, str] = {}
    for pred in predictions:
        raw_img_id = str(pred["image_id"])
        img_id = _resolve_image_filename(raw_img_id, imgs_pth, image_id_cache)
        pred_norm = dict(pred)
        pred_norm["image_id"] = img_id
        predictions_by_id.setdefault(img_id, []).append(pred_norm)

    metrics_rows = []
    img_sizes: dict[str, tuple[int, int]] = {}

    all_img_ids = _list_image_files(imgs_pth)
    if not all_img_ids:
        all_img_ids = sorted(predictions_by_id.keys())

    n_missing_preds = sum(1 for img_id in all_img_ids if img_id not in predictions_by_id)
    if n_missing_preds:
        print(
            f"Warning: {n_missing_preds}/{len(all_img_ids)} images have no entries in predictions.json; "
            "treating them as zero-detection frames."
        )

    for img_id in tqdm(all_img_ids, total=len(all_img_ids)):
        preds = predictions_by_id.get(img_id, [])
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

        label_pth = os.path.join(labels_pth, str(Path(img_id).with_suffix(".txt")))
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
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_csv, index=False)

    print(f"Wrote per-image metrics: {out_csv}")
    print(metrics_df[["P", "R", "F1", "AP"]].mean())

    if out_gt_csv or out_pred_csv:
        det_attrs = None
        if detections_attr_csv and os.path.exists(detections_attr_csv):
            det_attrs = pd.read_csv(detections_attr_csv)

        gt_rows = []
        pred_rows = []
        for img_id in tqdm(all_img_ids, total=len(all_img_ids), desc="Det-level"):
            preds = predictions_by_id.get(img_id, [])
            yolo_id = _yolo_id_from_img_id(img_id)
            label_pth = os.path.join(labels_pth, str(Path(img_id).with_suffix(".txt")))
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
