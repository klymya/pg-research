"""Generate a quantized YOLOv8-pose weight of a target size in MB.

Reuses the project pipeline: build a model config with tunable width and
max_channels, init YOLO from it without training, run export_non_separated.py
for int8 TFLite quantization, then optimize_vela.py for the Ethos-U55
deployable .tflite. The final size is measured on the vela-optimized .tflite.

Key facts that shape the search:
  * Channels are snapped to multiples of 8 (`make_divisible(c*width, 8)`),
    creating discrete plateaus in achievable vela size. Width alone cannot
    reach every target — sweeping `max_channels` in tandem reaches sizes
    inside the gaps between width-only plateaus.
  * PT size predicts vela size with mean ratio ~2.0 (std ±4.4%); PT init+save
    takes ~5 s vs minutes for the full export+vela pipeline. So we use a fast
    PT-proxy search to rank candidates, then validate the top picks with the
    real pipeline.

Example:
    python generate_quantized_weights.py --target-size-mb 2.0
    python generate_quantized_weights.py --target-size-mb 3.5 --max-iter 9
"""

import argparse
import copy
import math
import subprocess
import sys
from pathlib import Path

import yaml

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_CFG = SCRIPT_DIR / "yolov8-pose.yaml"

# Empirical (width, vela_MB) reference points from prior runs with max_channels=1024.
REFERENCE_POINTS = [
    (0.21, 2.2),  # yolov8-pose-tiny.yaml
    (0.25, 2.5),  # yolov8-pose.yaml (scale n)
    (0.29, 4.3),  # yolov8-pose-xs.yaml
]

# Empirical PT/vela ratio from prior 9-iteration run (mean 1.999, std 0.087).
DEFAULT_PT_TO_VELA_RATIO = 1.999

WIDTH_MIN = 0.05
WIDTH_MAX = 1.25
WIDTH_DELTA_STOP = 1e-3
DEFAULT_DEPTH = 0.33
DEFAULT_MAX_CHANNELS = 1024
MAX_CHANNELS_GRID = [1024, 896, 768, 640, 512, 384, 256]


# ---------- model + pipeline primitives ----------

def write_iteration_cfg(out_path: Path, width: float, max_channels: int) -> None:
    with TEMPLATE_CFG.open("r") as fh:
        cfg = yaml.safe_load(fh)
    cfg["scales"] = {"c": [DEFAULT_DEPTH, float(width), int(max_channels)]}
    cfg["scale"] = "c"
    with out_path.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def init_and_save_pt(cfg_path: Path, pt_path: Path) -> int:
    """Init YOLO from yaml (no training) and persist as .pt. Returns PT size in bytes."""
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(cfg_path))
    # Without training, self.ckpt is empty; reload then overwrites model.args from
    # defaults (task='detect'), breaking the pose export. Persist task explicitly.
    model.ckpt = {"train_args": {"task": model.task}}
    model.save(str(pt_path))
    return pt_path.stat().st_size


def run_subprocess(cmd: list, cwd: Path) -> None:
    print(f"[cwd={cwd}] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_export(pt_path: Path, data_yaml: Path, imgsz: int, max_ncalib_imgs: int) -> None:
    run_subprocess(
        [
            sys.executable, "export_non_separated.py",
            "--weights", str(pt_path),
            "--data", str(data_yaml),
            "--max_ncalib_imgs", str(max_ncalib_imgs),
            "--imgsz", str(imgsz),
            "--output-folder-name", "best_saved_model",
        ],
        cwd=SCRIPT_DIR,
    )


def run_vela(weights_dir: Path) -> None:
    run_subprocess(
        [
            sys.executable, "optimize_vela.py",
            str(weights_dir),
            "best_saved_model",
            "--skip-head",
        ],
        cwd=SCRIPT_DIR,
    )


def measure_vela_size_mb(weights_dir: Path) -> float:
    vela_tflite = weights_dir / "best_saved_model" / "vela_he" / "best_full_integer_quant_vela.tflite"
    if not vela_tflite.is_file():
        raise FileNotFoundError(f"Vela output not found: {vela_tflite}")
    return vela_tflite.stat().st_size / (1024 * 1024)


# ---------- search helpers ----------

def clamp_width(w: float) -> float:
    return max(WIDTH_MIN, min(WIDTH_MAX, w))


def initial_width_from_target(target_mb: float) -> float:
    pts = sorted(REFERENCE_POINTS, key=lambda p: p[1])
    if target_mb <= pts[0][1]:
        w0, s0 = pts[0]
        w1, s1 = pts[1]
    elif target_mb >= pts[-1][1]:
        w0, s0 = pts[-2]
        w1, s1 = pts[-1]
    else:
        for i in range(len(pts) - 1):
            if pts[i][1] <= target_mb <= pts[i + 1][1]:
                w0, s0 = pts[i]
                w1, s1 = pts[i + 1]
                break
    width = w0 + (w1 - w0) * (target_mb - s0) / (s1 - s0)
    return clamp_width(width)


def next_width_from_history(history: list, target_mb: float, current_width: float) -> float:
    """Log-linear secant on the last two (width, vela_mb) points; power-law fallback."""
    if len(history) == 1:
        _, size = history[0]
        ratio = target_mb / max(size, 1e-6)
        return clamp_width(current_width * (ratio ** (1.0 / 1.5)))

    (w_a, s_a), (w_b, s_b) = history[-2], history[-1]
    if w_a == w_b or s_a <= 0 or s_b <= 0 or abs(s_b - s_a) < 1e-6:
        ratio = target_mb / max(s_b, 1e-6)
        return clamp_width(w_b * (ratio ** (1.0 / 1.5)))

    log_wa, log_wb = math.log(w_a), math.log(w_b)
    log_sa, log_sb = math.log(s_a), math.log(s_b)
    b = (log_sb - log_sa) / (log_wb - log_wa)
    a = log_sa - b * log_wa
    if abs(b) < 1e-6:
        ratio = target_mb / max(s_b, 1e-6)
        return clamp_width(w_b * (ratio ** (1.0 / 1.5)))
    return clamp_width(math.exp((math.log(target_mb) - a) / b))


class VelaPredictor:
    """vela_MB ≈ pt_MB / ratio. Ratio updates online from observed (pt, vela) pairs."""

    def __init__(self, ratio: float = DEFAULT_PT_TO_VELA_RATIO):
        self.ratio = ratio
        self.samples = []

    def predict_mb(self, pt_bytes: int) -> float:
        return (pt_bytes / (1024 * 1024)) / self.ratio

    def update(self, pt_bytes: int, vela_mb: float) -> None:
        pt_mb = pt_bytes / (1024 * 1024)
        if vela_mb <= 0:
            return
        self.samples.append((pt_mb, vela_mb))
        # Average ratio over all observations (robust to single outliers).
        self.ratio = sum(p / v for p, v in self.samples) / len(self.samples)


# ---------- PT-only fast search ----------

def measure_pt(width: float, max_channels: int, work_dir: Path) -> tuple:
    """Build the cfg, init+save .pt, return (cfg_path, pt_path, pt_bytes)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = work_dir / "yolov8-pose-c.yaml"
    pt_path = work_dir / "weights" / "best.pt"
    write_iteration_cfg(cfg_path, width, max_channels)
    pt_bytes = init_and_save_pt(cfg_path, pt_path)
    return cfg_path, pt_path, pt_bytes


def find_upper_plateau_width(
    target_vela_mb: float,
    predictor: VelaPredictor,
    output_root: Path,
    pt_cache: dict,
    initial_width: float,
    max_pt_iters: int,
) -> float:
    """Find smallest width whose predicted vela ≥ target (PT-only iterations)."""
    width = initial_width
    history = []   # [(width, predicted_vela_mb)]
    last_above = None
    last_below = None

    for it in range(1, max_pt_iters + 1):
        key = (round(width, 5), DEFAULT_MAX_CHANNELS)
        if key in pt_cache:
            pt_bytes = pt_cache[key]
        else:
            work_dir = output_root / f"pt_w{width:.4f}_mc{DEFAULT_MAX_CHANNELS}"
            _, _, pt_bytes = measure_pt(width, DEFAULT_MAX_CHANNELS, work_dir)
            pt_cache[key] = pt_bytes

        pred_vela = predictor.predict_mb(pt_bytes)
        history.append((width, pred_vela))
        print(f"  PT-search iter {it}: width={width:.4f}  pt={pt_bytes/2**20:.3f} MB  "
              f"pred_vela={pred_vela:.3f} MB  (target {target_vela_mb:.3f})")

        if pred_vela >= target_vela_mb:
            last_above = (width, pred_vela)
            # Try to tighten: pick smaller width that still stays above.
            if last_below is not None:
                next_w = clamp_width((width + last_below[0]) / 2)
            else:
                next_w = clamp_width(width * 0.95)
        else:
            last_below = (width, pred_vela)
            if last_above is not None:
                next_w = clamp_width((width + last_above[0]) / 2)
            else:
                next_w = next_width_from_history(history, target_vela_mb, width)

        if abs(next_w - width) < WIDTH_DELTA_STOP:
            break
        width = next_w

    if last_above is not None:
        return last_above[0]
    # No width reached target — return the largest tried.
    return max(history, key=lambda h: h[1])[0]


def sweep_max_channels(
    width: float,
    target_vela_mb: float,
    predictor: VelaPredictor,
    output_root: Path,
    pt_cache: dict,
    mc_grid: list,
) -> list:
    """At fixed width, measure PT for each max_channels. Return ranked candidates."""
    results = []
    for mc in mc_grid:
        key = (round(width, 5), mc)
        if key in pt_cache:
            pt_bytes = pt_cache[key]
        else:
            work_dir = output_root / f"pt_w{width:.4f}_mc{mc}"
            _, _, pt_bytes = measure_pt(width, mc, work_dir)
            pt_cache[key] = pt_bytes
        pred_vela = predictor.predict_mb(pt_bytes)
        results.append({"width": width, "max_channels": mc, "pt_bytes": pt_bytes,
                        "pred_vela_mb": pred_vela})
        print(f"  MC-sweep: width={width:.4f}  mc={mc}  pt={pt_bytes/2**20:.3f} MB  "
              f"pred_vela={pred_vela:.3f} MB")
    return results


def rank_candidates(candidates: list, target_mb: float, predictor: VelaPredictor) -> list:
    out = []
    for c in candidates:
        c = copy.copy(c)
        c["pred_vela_mb"] = predictor.predict_mb(c["pt_bytes"])
        c["abs_err"] = abs(c["pred_vela_mb"] - target_mb)
        out.append(c)
    return sorted(out, key=lambda x: x["abs_err"])


# ---------- full-pipeline validation ----------

def run_full_pipeline(
    width: float, max_channels: int, output_root: Path,
    data_yaml: Path, imgsz: int, max_ncalib_imgs: int,
) -> tuple:
    iter_dir = output_root / f"full_w{width:.4f}_mc{max_channels}"
    weights_dir = iter_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = iter_dir / "yolov8-pose-c.yaml"
    pt_path = weights_dir / "best.pt"
    write_iteration_cfg(cfg_path, width, max_channels)
    pt_bytes = init_and_save_pt(cfg_path, pt_path)
    run_export(pt_path, data_yaml, imgsz, max_ncalib_imgs)
    run_vela(weights_dir)
    vela_mb = measure_vela_size_mb(weights_dir)
    return iter_dir, pt_bytes, vela_mb


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-size-mb", type=float, required=True,
                   help="Target size of the vela-optimized .tflite in MB.")
    p.add_argument("--tolerance-mb", type=float, default=0.05,
                   help="Acceptable absolute size deviation in MB.")
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Root directory for per-iteration artifacts. Defaults to "
             "results/sized_<target>mb under the script directory.",
    )
    p.add_argument(
        "--data", type=str,
        default="/home/klymy/data/people_detection/filtered_v3_margin_15/project.yaml",
        help="project.yaml used for calibration images.",
    )
    p.add_argument("--max-ncalib-imgs", type=int, default=30,
                   help="Calibration set size for int8 quantization.")
    p.add_argument("--imgsz", type=int, default=320, help="Model input size.")
    p.add_argument("--max-iter", type=int, default=9,
                   help="Maximum full export+vela iterations (the slow ones).")
    p.add_argument("--max-pt-iter", type=int, default=12,
                   help="Maximum PT-only width-search iterations (fast).")
    p.add_argument("--initial-width", type=float, default=None,
                   help="Override the auto-picked initial width multiplier.")
    p.add_argument("--max-channels-grid", type=str,
                   default=",".join(str(m) for m in MAX_CHANNELS_GRID),
                   help="Comma-separated max_channels values to sweep (descending).")
    p.add_argument("--no-pt-proxy", action="store_true",
                   help="Disable PT-only fast search; run full pipeline every iteration.")
    return p.parse_args()


def main():
    args = parse_args()
    target = args.target_size_mb
    tol = args.tolerance_mb

    if args.output_dir is None:
        tag = f"{target:.2f}".replace(".", "_")
        output_root = SCRIPT_DIR / "results" / f"sized_{tag}mb"
    else:
        output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Calibration data yaml not found: {data_yaml}")

    mc_grid = [int(x) for x in args.max_channels_grid.split(",") if x.strip()]
    initial_w = args.initial_width if args.initial_width is not None else initial_width_from_target(target)
    initial_w = clamp_width(initial_w)

    predictor = VelaPredictor()
    pt_cache = {}              # (width, max_channels) → pt_bytes
    full_results = []          # list of dicts: width, mc, pt_bytes, vela_mb, iter_dir
    best = None                # (abs_err, dict)

    print(f"Target: {target:.3f} MB  |  tolerance: ±{tol:.3f} MB  |  initial width: {initial_w:.4f}")
    print(f"max_channels grid: {mc_grid}")
    print(f"Output root: {output_root}")
    print(f"PT-proxy ratio (initial): {predictor.ratio:.3f}")

    # ----- Phase 1: PT-only fast search -----
    if not args.no_pt_proxy:
        print("\n=== Phase 1: PT-only proxy search ===")
        print("[Phase 1a] Finding upper-plateau width @ max_channels=1024")
        upper_width = find_upper_plateau_width(
            target, predictor, output_root, pt_cache, initial_w, args.max_pt_iter,
        )
        print(f"[Phase 1a] chosen width: {upper_width:.4f}")

        print("[Phase 1b] Sweeping max_channels at chosen width")
        cands = sweep_max_channels(upper_width, target, predictor, output_root, pt_cache, mc_grid)
        ranked = rank_candidates(cands, target, predictor)
        print(f"[Phase 1] top 5 candidates by predicted vela proximity:")
        for c in ranked[:5]:
            print(f"  width={c['width']:.4f}  mc={c['max_channels']}  "
                  f"pt={c['pt_bytes']/2**20:.3f} MB  pred_vela={c['pred_vela_mb']:.3f} MB  "
                  f"|err|={c['abs_err']:.3f}")
    else:
        # No PT proxy: just init the candidate list with the seed width.
        ranked = [{"width": initial_w, "max_channels": DEFAULT_MAX_CHANNELS,
                   "pt_bytes": 0, "pred_vela_mb": 0.0, "abs_err": float("inf")}]

    # ----- Phase 2: full-pipeline validation of top candidates -----
    print("\n=== Phase 2: full export+vela validation ===")
    tried = set()
    for full_iter in range(1, args.max_iter + 1):
        # Pick the best un-tried candidate.
        ranked = rank_candidates(ranked, target, predictor)
        cand = next((c for c in ranked if (round(c["width"], 5), c["max_channels"]) not in tried), None)
        if cand is None:
            # Exhausted candidates from Phase 1. If running in no-proxy mode, fall back
            # to the original width-only secant search using actual vela measurements.
            if args.no_pt_proxy and full_results:
                hist = [(r["width"], r["vela_mb"]) for r in full_results if r["max_channels"] == DEFAULT_MAX_CHANNELS]
                if hist:
                    new_w = next_width_from_history(hist, target, hist[-1][0])
                    cand = {"width": new_w, "max_channels": DEFAULT_MAX_CHANNELS,
                            "pt_bytes": 0, "pred_vela_mb": 0.0}
            if cand is None:
                print("No more candidates to try. Stopping.")
                break

        w = cand["width"]
        mc = cand["max_channels"]
        tried.add((round(w, 5), mc))

        print(f"\n--- Full iter {full_iter}/{args.max_iter}: width={w:.4f}  max_channels={mc} ---")
        iter_dir, pt_bytes, vela_mb = run_full_pipeline(
            w, mc, output_root, data_yaml, args.imgsz, args.max_ncalib_imgs,
        )
        err = vela_mb - target
        print(f"--- Result: width={w:.4f} mc={mc}  pt={pt_bytes/2**20:.3f} MB  "
              f"vela={vela_mb:.4f} MB  err={err:+.4f} MB ---")

        result = {"width": w, "max_channels": mc, "pt_bytes": pt_bytes,
                  "vela_mb": vela_mb, "iter_dir": iter_dir}
        full_results.append(result)
        predictor.update(pt_bytes, vela_mb)
        print(f"Predictor ratio updated → {predictor.ratio:.3f} "
              f"(n_samples={len(predictor.samples)})")

        if best is None or abs(err) < best[0]:
            best = (abs(err), result)

        if abs(err) <= tol:
            vela_path = iter_dir / "weights" / "best_saved_model" / "vela_he" / "best_full_integer_quant_vela.tflite"
            print(f"\nConverged: size={vela_mb:.4f} MB within ±{tol} MB of target {target:.3f} MB.")
            print(f"Final artifacts: {iter_dir}")
            print(f"Final vela tflite: {vela_path}")
            return 0

    # Fallback to closest match (Option A behavior).
    if best is None:
        print("No iterations succeeded. Exiting.")
        return 2
    abs_err, r = best
    vela_path = r["iter_dir"] / "weights" / "best_saved_model" / "vela_he" / "best_full_integer_quant_vela.tflite"
    print(f"\nTolerance not met. Closest achievable result:")
    print(f"  width={r['width']:.4f}  max_channels={r['max_channels']}  "
          f"vela={r['vela_mb']:.4f} MB  |err|={abs_err:.4f} MB")
    print(f"  artifacts: {r['iter_dir']}")
    print(f"  vela tflite: {vela_path}")
    print("Hint: rerun with a wider --tolerance-mb to mark this as success, "
          "or expand --max-channels-grid to explore finer steps.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
