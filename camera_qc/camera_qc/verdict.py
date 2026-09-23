"""Tolerance + aggregation + pass/marginal/fail decision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import yaml


Verdict = Literal["pass", "marginal", "fail"]
Direction = Literal["higher_better", "lower_better", "closer_to_ref"]
Severity = Literal["blocking", "advisory"]

DEFAULT_SEVERITY: Severity = "blocking"


@dataclass
class MetricResult:
    name: str
    ref: float
    dut_agg: float       # aggregated DUT value across N images (median/mean/worst)
    delta: float         # dut_agg - ref
    pct_of_ref: float    # dut_agg / ref * 100 (NaN if ref==0)
    direction: Direction
    tolerance_pct: float
    within_tol: bool
    severity: Severity = DEFAULT_SEVERITY

    @property
    def is_blocking(self) -> bool:
        return self.severity == "blocking"

    @property
    def fails(self) -> bool:
        """Outside tolerance *and* allowed to affect the verdict."""
        return self.is_blocking and not self.within_tol

    @property
    def warns(self) -> bool:
        """Outside tolerance but advisory — highlighted, never fatal."""
        return (not self.is_blocking) and (not self.within_tol)


@dataclass
class VerdictResult:
    verdict: Verdict
    scene: str
    aggregation: str
    n_dut: int
    per_metric: list[MetricResult]
    n_outside_tol: int           # blocking metrics outside tolerance — drives the verdict
    n_blocking: int              # blocking metrics evaluated
    n_advisory_outside_tol: int  # advisory metrics outside tolerance — warnings only
    n_advisory: int              # advisory metrics evaluated


@dataclass
class PerImageVerdict:
    """Verdict + per-metric detail for a single DUT image vs the reference.

    `override_reason` is set when the verdict was forced regardless of the
    measurements — a unit rejected for something the optics cannot show, such
    as an electronics or communication fault. The metric results are still
    the real measurements.
    """
    verdict: Verdict
    n_outside_tol: int           # blocking metrics outside tolerance — drives the verdict
    per_metric: list[MetricResult]   # dut_agg == this single image's value
    n_blocking: int
    n_advisory_outside_tol: int
    n_advisory: int
    override_reason: str | None = None


def load_config(path: Path | str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def aggregate(values: list[float], method: str, direction: Direction) -> float:
    arr = np.asarray(values, dtype=float)
    if method == "median":
        return float(np.median(arr))
    if method == "mean":
        return float(arr.mean())
    if method == "worst_case":
        # Pick the DUT image that's farthest from the reference *in the bad
        # direction* — used so a single bad capture can fail the batch.
        if direction == "higher_better":
            return float(arr.min())
        if direction == "lower_better":
            return float(arr.max())
        return float(arr[np.argmax(np.abs(arr - np.median(arr)))])
    raise ValueError(f"Unknown aggregation method: {method!r}")


def within_tolerance(
    ref: float, dut: float, direction: Direction, tol_pct: float
) -> bool:
    if ref == 0:
        # Avoid divide-by-zero; if both are zero, pass.
        return dut == 0
    if direction == "higher_better":
        # Bad if DUT drops more than tol% below ref. DUT above ref is always OK.
        floor = ref * (1 - tol_pct / 100.0)
        return dut >= floor
    if direction == "lower_better":
        # Bad if DUT rises more than tol% above ref. DUT below ref is always OK.
        ceiling = ref * (1 + tol_pct / 100.0)
        return dut <= ceiling
    # closer_to_ref: bad if |delta| exceeds tol% of |ref|
    return abs(dut - ref) <= abs(ref) * (tol_pct / 100.0)


def _scene_config(scene: str, config: Mapping) -> dict:
    scene_cfg = config["scenes"].get(scene)
    if scene_cfg is None:
        raise ValueError(
            f"Unknown scene preset {scene!r}. "
            f"Available: {list(config['scenes'].keys())}"
        )
    return scene_cfg


def _verdict_thresholds(config: Mapping) -> tuple[int, int]:
    """(fail_n, marginal_n) over blocking metrics.

    The `*_n_metrics_outside_tol` spellings are the pre-severity key names,
    accepted so that older copied configs keep working.
    """
    cfg = config["verdict"]
    fail_n = cfg.get(
        "fail_if_n_blocking_outside_tol", cfg.get("fail_if_n_metrics_outside_tol")
    )
    marginal_n = cfg.get(
        "marginal_if_n_blocking_outside_tol",
        cfg.get("marginal_if_n_metrics_outside_tol"),
    )
    if fail_n is None or marginal_n is None:
        raise ValueError(
            "config['verdict'] must define fail_if_n_blocking_outside_tol and "
            "marginal_if_n_blocking_outside_tol."
        )
    return int(fail_n), int(marginal_n)


def _classify(n_blocking_outside: int, config: Mapping) -> Verdict:
    """Verdict from the blocking metrics alone. Advisory metrics outside
    tolerance are reported and highlighted but never move the verdict.
    """
    fail_n, marginal_n = _verdict_thresholds(config)
    if n_blocking_outside >= fail_n:
        return "fail"
    if n_blocking_outside >= marginal_n:
        return "marginal"
    return "pass"


def _tally(per_metric: list[MetricResult]) -> dict[str, int]:
    return {
        "n_outside_tol": sum(1 for m in per_metric if m.fails),
        "n_blocking": sum(1 for m in per_metric if m.is_blocking),
        "n_advisory_outside_tol": sum(1 for m in per_metric if m.warns),
        "n_advisory": sum(1 for m in per_metric if not m.is_blocking),
    }


def _per_metric_results(
    ref_metrics: Mapping[str, float],
    dut_values_by_metric: Mapping[str, float],
    scene: str,
    config: Mapping,
) -> list[MetricResult]:
    scene_cfg = _scene_config(scene, config)
    tolerances: dict[str, float] = scene_cfg["tolerances_pct"]
    directions: dict[str, str] = config["metric_direction"]
    severities: dict[str, str] = config.get("metric_severity", {})

    results: list[MetricResult] = []
    for name, ref_val in ref_metrics.items():
        if name not in tolerances:
            continue
        direction = directions.get(name, "closer_to_ref")
        severity = severities.get(name, DEFAULT_SEVERITY)
        if severity not in ("blocking", "advisory"):
            raise ValueError(
                f"metric_severity[{name!r}] must be 'blocking' or 'advisory', "
                f"got {severity!r}."
            )
        dut_val = float(dut_values_by_metric[name])
        tol_pct = float(tolerances[name])
        within = within_tolerance(ref_val, dut_val, direction, tol_pct)
        results.append(MetricResult(
            name=name,
            ref=ref_val,
            dut_agg=dut_val,
            delta=dut_val - ref_val,
            pct_of_ref=(dut_val / ref_val * 100.0) if ref_val != 0 else float("nan"),
            direction=direction,  # type: ignore[arg-type]
            tolerance_pct=tol_pct,
            within_tol=within,
            severity=severity,  # type: ignore[arg-type]
        ))
    return results


def decide_single_image(
    ref_metrics: Mapping[str, float],
    dut_metrics: Mapping[str, float],
    scene: str,
    config: Mapping,
) -> PerImageVerdict:
    """Verdict for one DUT image against the reference."""
    per_metric = _per_metric_results(ref_metrics, dut_metrics, scene, config)
    counts = _tally(per_metric)
    return PerImageVerdict(
        verdict=_classify(counts["n_outside_tol"], config),
        per_metric=per_metric,
        **counts,
    )


def decide(
    ref_metrics: Mapping[str, float],
    dut_metrics_per_image: list[Mapping[str, float]],
    scene: str,
    config: Mapping,
) -> VerdictResult:
    """Aggregate verdict across N DUT images (median/mean/worst-case)."""
    scene_cfg = _scene_config(scene, config)
    tolerances: dict[str, float] = scene_cfg["tolerances_pct"]
    directions: dict[str, str] = config["metric_direction"]
    agg_method: str = config["verdict"].get("aggregation", "median")

    dut_agg: dict[str, float] = {}
    for name in ref_metrics:
        if name not in tolerances:
            continue
        direction = directions.get(name, "closer_to_ref")
        values = [m[name] for m in dut_metrics_per_image]
        dut_agg[name] = aggregate(values, agg_method, direction)  # type: ignore[arg-type]

    per_metric = _per_metric_results(ref_metrics, dut_agg, scene, config)
    counts = _tally(per_metric)

    return VerdictResult(
        verdict=_classify(counts["n_outside_tol"], config),
        scene=scene,
        aggregation=agg_method,
        n_dut=len(dut_metrics_per_image),
        per_metric=per_metric,
        **counts,
    )
