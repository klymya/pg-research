"""Camera QC: compare a DUT camera against a reference, emit a PDF verdict.

Public API
----------

>>> from camera_qc import evaluate
>>> result = evaluate(ref_dir="./ref", dut_dir="./dut", scene="chessboard")
>>> result.verdict.verdict        # "pass" | "marginal" | "fail"
>>> result.write_report("./report.pdf")
"""

from __future__ import annotations

from pathlib import Path

from .evaluate import EvaluationResult, run_evaluation
from .report import write_pdf


class _ResultWrapper:
    """Thin wrapper that adds `write_report()` to EvaluationResult so the
    Python API in the spec works as documented.
    """

    def __init__(self, inner: EvaluationResult):
        self._inner = inner

    # Pass-through attribute access — verdict, ref_metrics, dut_metrics, etc.
    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def per_dut(self) -> list[dict]:
        return self._inner.dut_metrics

    @property
    def summary(self) -> dict:
        return {m.name: m.dut_agg for m in self._inner.verdict.per_metric}

    def write_report(self, out_path: str | Path, keep_md: bool = False) -> Path:
        return write_pdf(self._inner, Path(out_path), keep_md=keep_md)


def evaluate(
    ref_dir: str | Path,
    dut_dir: str | Path,
    scene: str,
    config_path: str | Path | None = None,
) -> _ResultWrapper:
    return _ResultWrapper(run_evaluation(ref_dir, dut_dir, scene, config_path))


__all__ = ["evaluate", "write_pdf", "run_evaluation", "EvaluationResult"]
