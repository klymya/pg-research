"""Tolerance + aggregation + pass/marginal/fail decision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import yaml


Verdict = Literal["pass", "marginal", "fail"]
Direction = Literal["higher_better", "lower_better", "closer_to_ref"]


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


@dataclass
class VerdictResult:
    verdict: Verdict
    scene: str
    aggregation: str
    n_dut: int
    per_metric: list[MetricResult]
    n_outside_tol: int


@dataclass
class PerImageVerdict:
    """Verdict + per-metric detail for a single DUT image vs the reference."""
    verdict: Verdict
    n_outside_tol: int
    per_metric: list[MetricResult]   # dut_agg == this single image's value


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


def _classify(n_outside: int, config: Mapping) -> Verdict:
    fail_n = int(config["verdict"]["fail_if_n_metrics_outside_tol"])
    marginal_n = int(config["verdict"]["marginal_if_n_metrics_outside_tol"])
    if n_outside >= fail_n:
        return "fail"
    if n_outside >= marginal_n:
        return "marginal"
    return "pass"


def _per_metric_results(
    ref_metrics: Mapping[str, float],
    dut_values_by_metric: Mapping[str, float],
    scene: str,
    config: Mapping,
) -> list[MetricResult]:
    scene_cfg = _scene_config(scene, config)
    tolerances: dict[str, float] = scene_cfg["tolerances_pct"]
    directions: dict[str, str] = config["metric_direction"]

    results: list[MetricResult] = []
    for name, ref_val in ref_metrics.items():
        if name not in tolerances:
            continue
        direction = directions.get(name, "closer_to_ref")
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
    n_outside = sum(1 for m in per_metric if not m.within_tol)
    return PerImageVerdict(
        verdict=_classify(n_outside, config),
        n_outside_tol=n_outside,
        per_metric=per_metric,
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
    n_outside = sum(1 for m in per_metric if not m.within_tol)

    return VerdictResult(
        verdict=_classify(n_outside, config),
        scene=scene,
        aggregation=agg_method,
        n_dut=len(dut_metrics_per_image),
        per_metric=per_metric,
        n_outside_tol=n_outside,
    )
