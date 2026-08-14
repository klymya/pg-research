"""Top-level orchestration: load → metrics → verdict.

The CLI in __main__.py and the public Python API in __init__.py both call
into `run_evaluation` here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .loader import LoadedInputs, load_inputs
from .metrics import headline_metrics
from .preprocess import MODE as PREPROCESS_MODE
from .verdict import (
    PerImageVerdict, VerdictResult, decide, decide_single_image, load_config,
)


log = logging.getLogger(__name__)
PACKAGE_CONFIG = Path(__file__).parent / "config.yaml"


@dataclass
class EvaluationResult:
    inputs: LoadedInputs
    ref_metrics: dict
    dut_metrics: list[dict]                  # one dict per DUT image, same order as inputs.dut_paths
    per_image_verdicts: list[PerImageVerdict]  # one per DUT image, same order
    verdict: VerdictResult                   # aggregate across all DUT images
    config: dict
    scene: str


def run_evaluation(
    ref_dir: Path | str,
    dut_dir: Path | str,
    scene: str,
    config_path: Path | str | None = None,
    preprocess_ref: bool = False,
    preprocess_dut: bool = False,
    save_preprocessed: Path | str | None = None,
) -> EvaluationResult:
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

    log.info("Computing reference metrics")
    ref_metrics = headline_metrics(inputs.ref_image)
    dut_metrics = [
        headline_metrics(img)
        for img in tqdm(inputs.dut_images, desc="Computing DUT metrics", unit="img")
    ]

    log.info("Computing per-image verdicts (scene=%s, n_dut=%d)",
             scene, len(dut_metrics))
    per_image_verdicts = [
        decide_single_image(ref_metrics, m, scene, config) for m in dut_metrics
    ]
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
    )
