"""End-to-end pipeline: collect FP32 vs INT8 TFLite outputs and fit a Ridge
regression that maps INT8 boxes back onto FP32 boxes.

Consolidates the work previously split across `collect_outputs.ipynb` and
`regr_tiny.ipynb` / `regr.ipynb` into a single command.

Example:
    python train_quant_regression.py \
        --model-name cam3_revB_batch1_12_margin_15_wb_alif_clahe_box_01_dfl_01_2nd \
        --int8-path  /path/to/best_full_integer_quant.tflite \
        --fp32-path  /path/to/best_float32.tflite \
        --data-path  /path/to/yolo_dataset/batch1_12_margin_15_bg_wb_alif_clahe/images
"""

import argparse
import glob
import json
import os
import sys

# Disable XNNPACK before TF is imported (matches notebook setup).
os.environ.setdefault("TF_TFLITE_USE_XNNPACK", "0")

import cv2
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tflite_inference_debug import TFLiteInference


COORD_NAMES = ["dx", "dy", "dw", "dh", "dx_coord", "dy_coord"]
SPLITS = ("train", "val", "holdout")


def sample_image_paths(data_path: str, calibration_n: int, seed: int) -> list[str]:
    """Sample up to `calibration_n` jpgs per split, with a fixed shuffle seed."""
    rng = np.random.RandomState(seed)
    sampled: list[str] = []
    for split in SPLITS:
        paths = glob.glob(os.path.join(data_path, split, "*.jpg"))
        print(f"  {split}: found {len(paths)} images")
        rng.shuffle(paths)
        sampled.extend(paths[:calibration_n])
    print(f"  total sampled: {len(sampled)}")
    return sampled


def collect_outputs(
    img_paths: list[str],
    fp32_inference: TFLiteInference,
    uint8_inference: TFLiteInference,
    score_threshold: float,
    out_dir: str,
) -> dict[str, np.ndarray]:
    """Run paired FP32/INT8 inference, filter by INT8 score and dump npy arrays."""
    fp32_scores, fp32_boxes, uint8_scores, uint8_boxes = [], [], [], []

    for img_path in tqdm(img_paths, desc="Collecting outputs"):
        img = cv2.imread(img_path)
        if img is None:
            tqdm.write(f"  [warn] unreadable image, skipping: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fp32_out = fp32_inference.infer_image(img.astype(np.float32) / 255.0)
        uint8_out = uint8_inference.infer_image(img.astype(np.uint8))

        # NB: output ordering differs between the two graphs (see notebook).
        u_scores = uint8_out["outputs"][0]
        u_boxes = uint8_out["outputs"][1]
        f_scores = fp32_out["outputs"][1]
        f_boxes = fp32_out["outputs"][0]

        keep = np.where(u_scores[0] > score_threshold)[0]
        if keep.size == 0:
            continue

        fp32_scores.append(f_scores[0, keep])
        fp32_boxes.append(f_boxes[0, keep])
        uint8_scores.append(u_scores[0, keep])
        uint8_boxes.append(u_boxes[0, keep])

    if not fp32_scores:
        raise RuntimeError(
            f"No detections passed score_threshold={score_threshold}. "
            "Lower the threshold or check the model outputs."
        )

    arrays = {
        "fp32_scores": np.concatenate(fp32_scores),
        "fp32_boxes": np.concatenate(fp32_boxes),
        "uint8_scores": np.concatenate(uint8_scores),
        "uint8_boxes": np.concatenate(uint8_boxes),
    }

    os.makedirs(out_dir, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)
        print(f"  saved {name}.npy shape={arr.shape}")

    return arrays


def load_collected(out_dir: str) -> dict[str, np.ndarray]:
    return {
        name: np.load(os.path.join(out_dir, f"{name}.npy"))
        for name in ("fp32_scores", "fp32_boxes", "uint8_scores", "uint8_boxes")
    }


def fit_regression(
    fp32_boxes: np.ndarray,
    uint8_boxes: np.ndarray,
    params_path: str,
    alphas: np.ndarray,
) -> dict:
    """Fit one RidgeCV per box coordinate on the full 6D INT8 input."""
    params: dict[str, dict] = {}
    predictions = np.empty_like(fp32_boxes)

    for i, name in enumerate(COORD_NAMES):
        model = RidgeCV(fit_intercept=True, alphas=alphas)
        model.fit(uint8_boxes, fp32_boxes[:, i])
        predictions[:, i] = model.predict(uint8_boxes)

        params[name] = {
            "coef": [float(c) for c in model.coef_],
            "intercept": float(model.intercept_),
            "alpha": float(model.alpha_),
        }
        print(
            f"  {name:9s} intercept={model.intercept_:+.4f}  alpha={model.alpha_:.4g}  "
            f"coef={np.array2string(model.coef_, precision=4, suppress_small=True)}"
        )

    # MSE comparison: direct INT8 vs regression-corrected.
    mse_uint8 = mean_squared_error(fp32_boxes, uint8_boxes, multioutput="raw_values")
    mse_regr = mean_squared_error(fp32_boxes, predictions, multioutput="raw_values")

    print("\nMSE (FP32 vs INT8 direct)        | (FP32 vs Regression) | improvement")
    print("-" * 74)
    for i, name in enumerate(COORD_NAMES):
        impr = (mse_uint8[i] - mse_regr[i]) / mse_uint8[i] * 100
        print(f"  {name:9s} {mse_uint8[i]:10.4f}            {mse_regr[i]:10.4f}      {impr:+6.2f}%")
    overall_impr = (mse_uint8.mean() - mse_regr.mean()) / mse_uint8.mean() * 100
    print(f"  {'overall':9s} {mse_uint8.mean():10.4f}            {mse_regr.mean():10.4f}      {overall_impr:+6.2f}%")

    os.makedirs(os.path.dirname(params_path) or ".", exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"\nRegression parameters saved to: {params_path}")

    return params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect paired FP32/INT8 TFLite outputs and fit a Ridge regression "
                    "that recovers FP32 boxes from INT8 boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", required=True,
                   help="Identifier used for output sub-folder and default params filename.")
    p.add_argument("--int8-path", help="Path to the full-integer-quant TFLite model.")
    p.add_argument("--fp32-path", help="Path to the float32 TFLite model.")
    p.add_argument("--data-path",
                   help="Image dataset root containing train/, val/, holdout/ sub-folders of jpgs.")
    p.add_argument("--output-dir", default="collected_outputs",
                   help="Where collected npy arrays are written (under <output-dir>/<model-name>/).")
    p.add_argument("--params-path", default=None,
                   help="Where the regression params JSON is written. "
                        "Default: regr_models/<model-name>_ridge_regression_params.json")
    p.add_argument("--imgsz", type=int, default=320, help="Model input size.")
    p.add_argument("--calibration-n", type=int, default=100,
                   help="Max images sampled per split (train/val/holdout).")
    p.add_argument("--score-threshold", type=float, default=0.2,
                   help="INT8 score threshold for keeping detections.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for sampling.")
    p.add_argument("--skip-collect", action="store_true",
                   help="Skip inference; load existing npy arrays from --output-dir/<model-name>/.")
    p.add_argument("--skip-regression", action="store_true",
                   help="Only run the collection step.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    collected_dir = os.path.join(args.output_dir, args.model_name)
    params_path = args.params_path or os.path.join(
        "regr_models", f"{args.model_name}_ridge_regression_params.json"
    )

    print(f"\n{'#' * 20} Model: {args.model_name} {'#' * 20}")

    if args.skip_collect:
        print(f"\n[1/2] Loading cached outputs from {collected_dir}")
        arrays = load_collected(collected_dir)
        for name, arr in arrays.items():
            print(f"  {name}.npy shape={arr.shape}")
    else:
        missing = [name for name in ("int8_path", "fp32_path", "data_path")
                   if getattr(args, name) is None]
        if missing:
            raise SystemExit(
                f"Missing required arguments for collection step: "
                f"{', '.join('--' + m.replace('_', '-') for m in missing)}. "
                "Pass them, or use --skip-collect to reuse cached npy arrays."
            )

        print(f"\n[1/2] Sampling images from {args.data_path}")
        img_paths = sample_image_paths(args.data_path, args.calibration_n, args.seed)

        print(f"\n[1/2] Loading TFLite interpreters")
        fp32_inf = TFLiteInference(args.fp32_path, is_uint8=False, imgsz=args.imgsz)
        uint8_inf = TFLiteInference(args.int8_path, is_uint8=True, imgsz=args.imgsz)

        print(f"\n[1/2] Running paired inference -> {collected_dir}")
        arrays = collect_outputs(
            img_paths, fp32_inf, uint8_inf, args.score_threshold, collected_dir,
        )

    if args.skip_regression:
        print("\n[2/2] Skipped (--skip-regression).")
        return

    print(f"\n[2/2] Fitting RidgeCV regression")
    fit_regression(
        arrays["fp32_boxes"],
        arrays["uint8_boxes"],
        params_path,
        alphas=np.logspace(-2, 1, 50),
    )


if __name__ == "__main__":
    main()
