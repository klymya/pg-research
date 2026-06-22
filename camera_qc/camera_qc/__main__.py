"""CLI entry point: `python -m camera_qc` or `camera-qc`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .evaluate import run_evaluation
from .narrative import narrate
from .report import write_pdf


EXIT_PASS = 0
EXIT_MARGINAL = 1
EXIT_FAIL = 2
EXIT_RUNTIME_ERROR = 10


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camera-qc",
        description=(
            "Compare a DUT camera module against a reference and emit a PDF "
            "verdict. Provide one image in --ref and N images in --dut, "
            "captured in the same controlled environment."
        ),
    )
    p.add_argument("--ref", required=True, type=Path,
                   help="Folder containing exactly one reference image.")
    p.add_argument("--dut", required=True, type=Path,
                   help="Folder containing one or more DUT images.")
    p.add_argument("--scene", required=True, choices=("chessboard", "ceiling"),
                   help="Scene preset (selects the tolerance band).")
    p.add_argument("--out", type=Path, default=Path("./report.pdf"),
                   help="Output PDF path (default: ./report.pdf).")
    p.add_argument("--config", type=Path, default=None,
                   help="Override the bundled tolerance config.yaml.")
    p.add_argument("--keep-md", action="store_true",
                   help="Keep the intermediate Markdown + image assets next "
                        "to the PDF for debugging.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose logging (DEBUG level).")
    return p


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence chatty third-party libraries unless --verbose.
    if not verbose:
        for noisy in ("weasyprint", "weasyprint.progress",
                      "fontTools", "fontTools.subset", "fontTools.ttLib",
                      "PIL", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_per_image(result) -> None:
    """Per-DUT-image block: filename, verdict, narrative, per-metric table."""
    n = len(result.dut_metrics)
    for i, (path, dut_m, piv) in enumerate(zip(
        result.inputs.dut_paths, result.dut_metrics, result.per_image_verdicts
    )):
        print()
        print(f"--- DUT {i+1}/{n}: {path.name} ---")
        print(f"  Verdict: {piv.verdict.upper()}  "
              f"({piv.n_outside_tol}/{len(piv.per_metric)} metrics outside tolerance)")
        print(f"  {narrate(result.ref_metrics, dut_m)}")
        for m in piv.per_metric:
            flag = "  " if m.within_tol else "* "
            print(f"  {flag}{m.name:14s}  ref={m.ref:10.3f}  "
                  f"dut={m.dut_agg:10.3f}  Δ={m.delta:+10.3f}  ({m.direction})")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        result = run_evaluation(
            ref_dir=args.ref,
            dut_dir=args.dut,
            scene=args.scene,
            config_path=args.config,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    out_path = write_pdf(result, args.out, keep_md=args.keep_md)

    v = result.verdict
    _print_per_image(result)
    print()
    print("=" * 72)
    print(f"Batch verdict: {v.verdict.upper()}  "
          f"({v.n_outside_tol}/{len(v.per_metric)} metrics outside tolerance "
          f"on the {v.aggregation} aggregate across {v.n_dut} DUT image(s))")
    pass_n = sum(1 for p in result.per_image_verdicts if p.verdict == "pass")
    margin_n = sum(1 for p in result.per_image_verdicts if p.verdict == "marginal")
    fail_n = sum(1 for p in result.per_image_verdicts if p.verdict == "fail")
    print(f"Per-image roll-call: {pass_n} pass / {margin_n} marginal / {fail_n} fail")
    print(f"Report: {out_path}")

    return {"pass": EXIT_PASS, "marginal": EXIT_MARGINAL, "fail": EXIT_FAIL}[v.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
