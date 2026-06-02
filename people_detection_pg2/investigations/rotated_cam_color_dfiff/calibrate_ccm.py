#!/usr/bin/env python3
"""Fit a 3x3 Color Correction Matrix that maps the new sensor's pixels to the
old sensor's pixels, then write the optimized coefficients into the firmware
file (color_correction.c) as #define macros.

Param vector layout matches the macro order in the firmware:
    index:  0     1     2     3     4     5     6     7     8
    macro:  C_RR  C_RG  C_RB  C_GR  C_GG  C_GB  C_BR  C_BG  C_BB

The firmware stores the CCM column-major: C_xy is the contribution of input
channel x to output channel y. The Red output row is therefore
[C_RR, C_GR, C_BR] (param indices 0, 3, 6).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize


# ----- defaults -----------------------------------------------------------

DEFAULT_IMG_ROOT = Path(__file__).parent / "images"
DEFAULT_FIRMWARE_FILE = Path(
    "/home/klymy/projects/rtos_sensor/projects/"
    "alif_ml-embedded-evaluation-kit_pg/source/hal/source/components/"
    "image/source/ensemble/src/color_correction.c"
)
DEFAULT_JSON_OUT = Path(__file__).parent / "ccm.json"
SCENES = (0, 1, 2, 3)

MACRO_NAMES = (
    "C_RR", "C_RG", "C_RB",
    "C_GR", "C_GG", "C_GB",
    "C_BR", "C_BG", "C_BB",
)

INITIAL_PARAMS = np.array(
    [1.45, 0.00, -0.05, 0.25, 0.95, -0.53, -0.53, 0.07, 2.27],
    dtype=np.float64,
)


# ----- block 1: data ingestion --------------------------------------------

def load_pair_stack(img_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load paired calibration scenes; return (X_new, X_old) in [0, 1] floats.

    Both arrays have shape (N_total_pixels, 3) in RGB order. `X_new` is the
    input manifold; `X_old` is the target (reference) manifold.
    """
    new_stack, old_stack = [], []
    for i in SCENES:
        new_path = img_root / "new" / f"{i}.jpg"
        old_path = img_root / "old" / f"{i}.jpg"
        new_bgr = cv2.imread(str(new_path))
        old_bgr = cv2.imread(str(old_path))
        if new_bgr is None:
            raise FileNotFoundError(new_path)
        if old_bgr is None:
            raise FileNotFoundError(old_path)
        if new_bgr.shape != old_bgr.shape:
            raise ValueError(
                f"Scene {i}: paired shapes differ "
                f"{new_bgr.shape} vs {old_bgr.shape}"
            )
        new_stack.append(cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3))
        old_stack.append(cv2.cvtColor(old_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3))

    X_new = np.vstack(new_stack).astype(np.float64) / 255.0
    X_old = np.vstack(old_stack).astype(np.float64) / 255.0
    return X_new, X_old


# ----- block 2: optimization ----------------------------------------------

def params_to_matrix(params: np.ndarray) -> np.ndarray:
    """Pack the 9-vector into the 3x3 matrix `M` such that `out = X @ M.T`.

    Mirrors color_correction.c exactly:
        R_out = C_RR*R + C_GR*G + C_BR*B
        G_out = C_RG*R + C_GG*G + C_BG*B
        B_out = C_RB*R + C_GB*G + C_BB*B
    """
    p = params
    return np.array(
        [
            [p[0], p[3], p[6]],  # R_out row: C_RR, C_GR, C_BR
            [p[1], p[4], p[7]],  # G_out row: C_RG, C_GG, C_BG
            [p[2], p[5], p[8]],  # B_out row: C_RB, C_GB, C_BB
        ],
        dtype=np.float64,
    )


def make_cost(X_new: np.ndarray, X_old: np.ndarray, row_penalty: float):
    """MSE between the transformed new-sensor pixels and the old-sensor target,
    plus a row-sum regularizer that pulls each output-channel row toward 1.0
    (preserves global luma and protects against dynamic-range clipping).
    """
    def cost(params: np.ndarray) -> float:
        M = params_to_matrix(params)
        pred = X_new @ M.T
        np.clip(pred, 0.0, 1.0, out=pred)
        mse = float(np.mean((pred - X_old) ** 2))
        reg = float(np.sum((M.sum(axis=1) - 1.0) ** 2))
        return mse + row_penalty * reg

    return cost


def optimize_ccm(X_new: np.ndarray, X_old: np.ndarray, row_penalty: float):
    cost = make_cost(X_new, X_old, row_penalty)
    return minimize(
        cost,
        INITIAL_PARAMS.copy(),
        method="Nelder-Mead",
        options={
            "xatol": 1e-5,
            "fatol": 1e-8,
            "maxiter": 20000,
            "adaptive": True,
        },
    )


# ----- block 3: firmware file rewrite -------------------------------------

def fmt_coef(v: float) -> str:
    """Format like the firmware: signed two-decimal float literal, e.g. +1.45f."""
    r = round(float(v), 2)
    if r == 0.0:  # collapse -0.0 to +0.00
        r = 0.0
    return f"{r:+.2f}f"


def rewrite_firmware(
    file_path: Path, params: np.ndarray
) -> dict[str, tuple[str, str]]:
    """Substitute each `#define C_xy` macro's literal with the new value.

    Returns {macro_name: (old_literal, new_literal)} for reporting. Raises if
    any of the 9 macros is missing.
    """
    text = file_path.read_text()
    swaps: dict[str, tuple[str, str]] = {}
    for name, value in zip(MACRO_NAMES, params):
        new_lit = fmt_coef(value)
        pattern = re.compile(
            rf"(^\s*#\s*define\s+{re.escape(name)}\s+)([+-]?\d+(?:\.\d+)?f)\b",
            flags=re.MULTILINE,
        )
        match = pattern.search(text)
        if match is None:
            raise RuntimeError(
                f"Could not locate `#define {name}` in {file_path}"
            )
        old_lit = match.group(2)
        text, n = pattern.subn(rf"\g<1>{new_lit}", text, count=1)
        if n != 1:
            raise RuntimeError(
                f"Expected exactly one replacement for {name}, got {n}"
            )
        swaps[name] = (old_lit, new_lit)
    file_path.write_text(text)
    return swaps


# ----- driver -------------------------------------------------------------

def fmt_matrix(M: np.ndarray) -> str:
    return "\n".join(
        "  " + "  ".join(f"{v:+.4f}" for v in row) for row in M
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--images", type=Path, default=DEFAULT_IMG_ROOT,
        help="Folder containing old/ and new/ subfolders of paired JPEGs.",
    )
    ap.add_argument(
        "--firmware", type=Path, default=DEFAULT_FIRMWARE_FILE,
        help="Path to color_correction.c to rewrite.",
    )
    ap.add_argument(
        "--row-penalty", type=float, default=0.5,
        help="Weight on the (row_sum - 1.0)^2 regularizer.",
    )
    ap.add_argument(
        "--json-out", type=Path, default=DEFAULT_JSON_OUT,
        help="Path to write the optimized CCM as JSON (always written).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print optimized macros without modifying the firmware file.",
    )
    args = ap.parse_args()

    # Block 1
    X_new, X_old = load_pair_stack(args.images)
    print(f"Loaded {X_new.shape[0]:,} paired pixels from {args.images}")

    # Block 2
    cost_fn = make_cost(X_new, X_old, args.row_penalty)
    init_cost = cost_fn(INITIAL_PARAMS)
    print(f"Initial cost (MSE + {args.row_penalty} * row_penalty) = {init_cost:.6f}")

    result = optimize_ccm(X_new, X_old, args.row_penalty)
    print(
        f"Optimizer: success={result.success} iters={result.nit} "
        f"final_cost={result.fun:.6f}"
    )
    if not result.success:
        print(f"Warning: optimizer reported failure: {result.message}")

    fitted = result.x
    M = params_to_matrix(fitted)
    print("\nFitted CCM (rows = output R, G, B):")
    print(fmt_matrix(M))
    rs = M.sum(axis=1)
    print(
        f"\nRow sums (target 1.0):  R={rs[0]:+.4f}  G={rs[1]:+.4f}  B={rs[2]:+.4f}"
    )

    print("\nMacro values:")
    for name, v in zip(MACRO_NAMES, fitted):
        print(f"  #define {name}  {fmt_coef(v)}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images_root": str(args.images.resolve()),
        "row_penalty": args.row_penalty,
        "optimizer": {
            "method": "Nelder-Mead",
            "success": bool(result.success),
            "iterations": int(result.nit),
            "final_cost": float(result.fun),
            "initial_cost": float(init_cost),
        },
        "macros_rounded": {name: round(float(v), 2) for name, v in zip(MACRO_NAMES, fitted)},
        "macros_raw": {name: float(v) for name, v in zip(MACRO_NAMES, fitted)},
        "matrix_rows_RGB": M.tolist(),
        "row_sums_RGB": rs.tolist(),
    }
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote calibration JSON: {args.json_out}")

    # Block 3
    if args.dry_run:
        print("\n--dry-run: firmware file NOT modified.")
        return

    if not args.firmware.is_file():
        raise SystemExit(f"Firmware file not found: {args.firmware}")

    swaps = rewrite_firmware(args.firmware, fitted)
    print(f"\nRewrote {args.firmware}:")
    for name, (old, new) in swaps.items():
        marker = " " if old == new else "*"
        print(f"  {marker} {name}: {old:>8s}  ->  {new}")


if __name__ == "__main__":
    main()
