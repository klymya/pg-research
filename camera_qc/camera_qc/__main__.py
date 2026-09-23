"""CLI entry point: `python -m camera_qc` or `camera-qc`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .evaluate import run_evaluation
from .narrative import narrate
from .preprocess import MODE as PREPROCESS_MODE
from .report import write_pdf


DEFAULT_PREPROCESSED_DIR = Path("./preprocessed")

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
    p.add_argument("--simplified", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Simplified report (default): measure sharpness only "
                        "and emit a pass/fail headcount, a one-row-per-image "
                        "table and the ref-vs-DUT image pairs. Sharpness is "
                        "the only metric that can fail a unit, so the verdict "
                        "is unchanged. Use --no-simplified for the full "
                        "metric pack, narratives and RGB histograms.")
    p.add_argument("--preprocess-ref", action="store_true",
                   help="Apply the capture pipeline (ALIF white balance -> "
                        "CLAHE) to the reference image before measuring. Use "
                        "when the reference comes from a raw source and the "
                        "DUT captures are already preprocessed.")
    p.add_argument("--preprocess-dut", action="store_true",
                   help="Apply the same pipeline to the DUT images. Both "
                        "sides must be processed the same way for the metric "
                        "deltas to reflect the sensor alone.")
    p.add_argument("--save-preprocessed", type=Path, nargs="?",
                   const=DEFAULT_PREPROCESSED_DIR, default=None,
                   metavar="DIR",
                   help="Write the preprocessed images to DIR/ref and DIR/dut "
                        "for inspection. DIR is optional and defaults to "
                        f"{DEFAULT_PREPROCESSED_DIR}. Source folders are never "
                        "modified.")
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


def _metric_flag(m) -> str:
    """`*` = blocking metric outside tolerance (fails the unit);
    `!` = advisory metric outside tolerance (warning only)."""
    if m.fails:
        return "* "
    if m.warns:
        return "! "
    return "  "


def _print_per_image(result) -> None:
    """Per-DUT-image block: filename, verdict, narrative, per-metric table."""
    n = len(result.dut_metrics)
    for i, (path, dut_m, piv) in enumerate(zip(
        result.inputs.dut_paths, result.dut_metrics, result.per_image_verdicts
    )):
        print()
        print(f"--- DUT {i+1}/{n}: {path.name} ---")
        advisory = (
            f", {piv.n_advisory_outside_tol}/{piv.n_advisory} advisory"
            if piv.n_advisory_outside_tol else ""
        )
        if piv.override_reason:
            # Printing "FAIL (0/1 blocking metrics outside tolerance)" reads
            # as a contradiction; name the reason instead.
            print(f"  Verdict: FAIL  (non-optical: {piv.override_reason}; "
                  f"optics measured {piv.n_outside_tol}/{piv.n_blocking} "
                  f"blocking metrics outside tolerance{advisory})")
        else:
            print(f"  Verdict: {piv.verdict.upper()}  "
                  f"({piv.n_outside_tol}/{piv.n_blocking} blocking metrics "
                  f"outside tolerance{advisory})")
        # narrate() reads the full metric pack; in simplified mode only
        # sharpness was measured, so the per-metric lines below say it all.
        if not result.simplified:
            print(f"  {narrate(result.ref_metrics, dut_m)}")
        for m in piv.per_metric:
            print(f"  {_metric_flag(m)}{m.name:14s}  ref={m.ref:10.3f}  "
                  f"dut={m.dut_agg:10.3f}  Δ={m.delta:+10.3f}  "
                  f"({m.direction}, {m.severity})")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        result = run_evaluation(
            ref_dir=args.ref,
            dut_dir=args.dut,
            scene=args.scene,
            config_path=args.config,
            preprocess_ref=args.preprocess_ref,
            preprocess_dut=args.preprocess_dut,
            save_preprocessed=args.save_preprocessed,
            simplified=args.simplified,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR

    if result.inputs.preprocess_asymmetric:
        processed = "reference" if result.inputs.preprocess_ref else "DUT"
        raw = "DUT" if result.inputs.preprocess_ref else "reference"
        print(f"WARNING: {processed} preprocessed ({PREPROCESS_MODE}), "
              f"{raw} left raw — metric deltas reflect the processing "
              f"difference, not only the sensor difference.", file=sys.stderr)

    out_path = write_pdf(result, args.out, keep_md=args.keep_md)

    v = result.verdict
    _print_per_image(result)
    print()
    print("=" * 72)
    print(f"Batch verdict: {v.verdict.upper()}  "
          f"({v.n_outside_tol}/{v.n_blocking} blocking metrics outside tolerance "
          f"on the {v.aggregation} aggregate across {v.n_dut} DUT image(s))")
    if v.n_advisory_outside_tol:
        offenders = ", ".join(m.name for m in v.per_metric if m.warns)
        print(f"Advisory: {v.n_advisory_outside_tol}/{v.n_advisory} colour/noise "
              f"metrics outside tolerance ({offenders}) — reported only, does "
              f"not affect the verdict.")
    pass_n = sum(1 for p in result.per_image_verdicts if p.verdict == "pass")
    margin_n = sum(1 for p in result.per_image_verdicts if p.verdict == "marginal")
    fail_n = sum(1 for p in result.per_image_verdicts if p.verdict == "fail")
    print(f"Per-image roll-call: {pass_n} pass / {margin_n} marginal / {fail_n} fail")
    forced = [(p.name, v.override_reason)
              for p, v in zip(result.inputs.dut_paths, result.per_image_verdicts)
              if v.override_reason]
    if forced:
        print(f"  of which {len(forced)} failed on non-optical grounds "
              f"(optics within tolerance):")
        for name, reason in forced:
            print(f"    {name} — {reason}")
    print(f"Report: {out_path}")
    if args.save_preprocessed is not None and (args.preprocess_ref or args.preprocess_dut):
        print(f"Preprocessed images: {args.save_preprocessed.resolve()}")

    return {"pass": EXIT_PASS, "marginal": EXIT_MARGINAL, "fail": EXIT_FAIL}[v.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
