"""Top-level orchestration: load → metrics → verdict.

The CLI in __main__.py and the public Python API in __init__.py both call
into `run_evaluation` here.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .loader import LoadedInputs, load_inputs
from .metrics import headline_metrics, sharpness_metrics
from .preprocess import MODE as PREPROCESS_MODE
from .verdict import (
    PerImageVerdict, VerdictResult, decide, decide_single_image, load_config,
)


log = logging.getLogger(__name__)
PACKAGE_CONFIG = Path(__file__).parent / "config.yaml"


def load_forced_failures(dut_dir: Path) -> dict[str, str]:
    """Non-optical failures declared in the batch's `manifest.csv`.

    A unit can be rejected for something no image metric can see — an
    electronics or communication fault. The optical measurement is still taken
    and reported, but the verdict is forced to FAIL so such a unit is never
    counted as accepted merely because its optics are fine.

    Absent or malformed manifest: no overrides, behaviour unchanged.
    """
    manifest = Path(dut_dir).parent / "manifest.csv"
    if not manifest.is_file():
        return {}
    try:
        with manifest.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error) as e:
        log.warning("Could not read %s: %s", manifest, e)
        return {}
    return {
        r["dut"]: (r.get("forced_reason") or "declared non-optical failure")
        for r in rows
        if r.get("dut") and (r.get("forced_verdict") or "").strip() == "fail"
    }


@dataclass
class EvaluationResult:
    inputs: LoadedInputs
    ref_metrics: dict
    dut_metrics: list[dict]                  # one dict per DUT image, same order as inputs.dut_paths
    per_image_verdicts: list[PerImageVerdict]  # one per DUT image, same order
    verdict: VerdictResult                   # aggregate across all DUT images
    config: dict
    scene: str
    simplified: bool = False                 # True when only sharpness was measured


def run_evaluation(
    ref_dir: Path | str,
    dut_dir: Path | str,
    scene: str,
    config_path: Path | str | None = None,
    preprocess_ref: bool = False,
    preprocess_dut: bool = False,
    save_preprocessed: Path | str | None = None,
    simplified: bool = True,
) -> EvaluationResult:
    """Load, measure and judge one ref/dut pair of folders.

    `simplified` (the default) measures sharpness alone. Sharpness is the only
    blocking metric, so the verdict is unchanged; the colour/exposure/noise
    metrics are advisory and are simply not reported. Pass `simplified=False`
    for the full metric pack and the long-form report.
    """
    config = load_config(config_path or PACKAGE_CONFIG)

    if scene not in config["scenes"]:
        raise ValueError(
            f"Unknown --scene {scene!r}. "
            f"Available: {list(config['scenes'].keys())}"
        )

    inputs = load_inputs(
        ref_dir,
        dut_dir,
        preprocess_ref=preprocess_ref,
        preprocess_dut=preprocess_dut,
        save_preprocessed=save_preprocessed,
    )

    if inputs.preprocess_asymmetric:
        log.warning(
            "Only the %s side was preprocessed (%s) — metric deltas reflect "
            "the processing difference, not only the sensor difference.",
            "reference" if inputs.preprocess_ref else "DUT",
            PREPROCESS_MODE,
        )

    min_n = int(config["scenes"][scene].get("min_dut_images", 1))
    if len(inputs.dut_images) < min_n:
        raise ValueError(
            f"Scene preset {scene!r} requires at least {min_n} DUT image(s); "
            f"got {len(inputs.dut_images)}."
        )

    measure = sharpness_metrics if simplified else headline_metrics
    log.info("Computing reference metrics (%s)",
             "sharpness only" if simplified else "full metric pack")
    ref_metrics = measure(inputs.ref_image)
    dut_metrics = [
        measure(img)
        for img in tqdm(inputs.dut_images, desc="Computing DUT metrics", unit="img")
    ]

    log.info("Computing per-image verdicts (scene=%s, n_dut=%d)",
             scene, len(dut_metrics))
    per_image_verdicts = [
        decide_single_image(ref_metrics, m, scene, config) for m in dut_metrics
    ]
    forced = load_forced_failures(Path(dut_dir))
    if forced:
        log.info("%d non-optical failure(s) declared in the manifest", len(forced))
        for i, path in enumerate(inputs.dut_paths):
            reason = forced.get(path.name)
            if reason is None:
                continue
            piv = per_image_verdicts[i]
            log.info("Forcing %s to FAIL — %s (optics read %s)",
                     path.name, reason, piv.verdict.upper())
            per_image_verdicts[i] = dataclasses.replace(
                piv, verdict="fail", override_reason=reason)

    # Present sharpest first. Sorting here rather than in the renderer keeps
    # the terminal roll-call, the simplified report and the full report in one
    # order, and makes the DUT numbering (#1, #2, ...) mean the same thing in
    # all three. Ties break on filename so the order is reproducible.
    order = sorted(
        range(len(dut_metrics)),
        key=lambda i: (-dut_metrics[i]["sharpness"], inputs.dut_paths[i].name),
    )
    inputs.dut_paths = [inputs.dut_paths[i] for i in order]
    inputs.dut_images = [inputs.dut_images[i] for i in order]
    dut_metrics = [dut_metrics[i] for i in order]
    per_image_verdicts = [per_image_verdicts[i] for i in order]
    log.info("Ordered %d DUT image(s) by sharpness, highest first "
             "(%.1f down to %.1f)", len(dut_metrics),
             dut_metrics[0]["sharpness"], dut_metrics[-1]["sharpness"])

    log.info("Aggregating batch verdict")
    verdict = decide(ref_metrics, dut_metrics, scene, config)

    return EvaluationResult(
        inputs=inputs,
        ref_metrics=ref_metrics,
        dut_metrics=dut_metrics,
        per_image_verdicts=per_image_verdicts,
        verdict=verdict,
        config=config,
        scene=scene,
        simplified=simplified,
    )
