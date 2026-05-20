#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

"""
Batch runner for the existing extended evaluation pipeline.

Purpose
- Discover YOLO training result folders matching a glob.
- Reuse `extended_evaluation_metrics/run_eval_and_viz.sh` for each experiment.
- Materialize per-experiment env files derived from the existing
  `.env.H270_16-44-09.batch1_10` settings pattern.

Required overrides for this workflow
- Holdout attributes from `extended_evaluation_metrics/holdout_data`
- Dataset is selected per experiment name:
  - `--use-baseline-test-set` -> `batch_testAccuracy_margin_10_wb_alif_clahe`
  - contains `clahe` -> `batch1_3_margin_15_wb_alif_clahe`
  - otherwise -> `batch1_3_margin_15_wb_alif`
- Backend is selected by CLI:
  - default: `tflite_backbone_only`
  - `--evaluate-fp32`: `ultralytics_pt`

This script intentionally does not reimplement evaluation logic. It only:
- validates experiment artifacts expected by the configured backend
- writes concrete env files for each discovered experiment
- invokes the current orchestration shell script
- captures logs and prints a final summary
"""


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_dir: Path
    env_file: Path
    log_file: Path
    project_path: Path
    backend: str
    dataset_name: str
    weights_path: Path


@dataclass(frozen=True)
class RunResult:
    experiment_dir: Path
    env_file: Path
    log_file: Path
    returncode: int


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def extended_eval_root() -> Path:
    return repo_root() / "people_detection_pg2" / "extended_evaluation_metrics"


def parse_args() -> argparse.Namespace:
    default_pattern = "people_detection_pg2/results/dataset_v2_cam4_revB_batch1_*"
    parser = argparse.ArgumentParser(
        description="Run the existing extended evaluation pipeline for dataset_v2 batch1 experiments."
    )
    parser.add_argument(
        "--pattern",
        default=default_pattern,
        help=f"Glob used to discover experiment directories (default: {default_pattern})",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Optional experiment directory name or path to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of discovered experiments after sorting.",
    )
    parser.add_argument(
        "--template-env",
        default="people_detection_pg2/extended_evaluation_metrics/.env.H270_16-44-09.batch1_10",
        help="Template env file used as the source of evaluation settings.",
    )
    parser.add_argument(
        "--run-script",
        default="people_detection_pg2/extended_evaluation_metrics/run_eval_and_viz.sh",
        help="Existing orchestration script to invoke per experiment.",
    )
    parser.add_argument(
        "--python-env",
        default=os.environ.get("PYENV_VERSION", ""),
        help="Optional PYENV_VERSION value to export when launching evaluations.",
    )
    parser.add_argument(
        "--project",
        default="val_dataset_v2_batch1",
        help="PROJECT value for generated env files.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix to append to project name with underscore (e.g., 'holdout' -> 'val_dataset_v2_batch1_holdout').",
    )
    parser.add_argument(
        "--generated-env-dir",
        default="people_detection_pg2/extended_evaluation_metrics/generated_envs/dataset_v2_batch1",
        help="Directory where generated env files are written.",
    )
    parser.add_argument(
        "--log-dir",
        default="people_detection_pg2/extended_evaluation_metrics/logs/dataset_v2_batch1",
        help="Directory where per-experiment stdout/stderr logs are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovery and generated config paths without running evaluations.",
    )
    parser.add_argument(
        "--evaluate-fp32",
        action="store_true",
        help="Use ultralytics_pt with weights/best.pt instead of the default tflite_backbone_only backend.",
    )
    parser.add_argument(
        "--evaluate-tflite-mixed",
        action="store_true",
        help="Use tflite_mixed: int8 backbone (best_saved_model/best_full_integer_quant.tflite) + fp32 head (head_saved_model/head_float32.tflite).",
    )
    parser.add_argument(
        "--use-baseline-test-set",
        action="store_true",
        help="Override dataset selection and always evaluate on batch_testAccuracy_margin_10_wb_alif_clahe.",
    )
    parser.add_argument(
        "--holdout-dir",
        default="holdout_data",
        help="Holdout data directory name within extended_evaluation_metrics/ (default: holdout_data).",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failing experiment.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def load_template_env(template_path: Path) -> dict[str, str]:
    template: dict[str, str] = {}
    variable_pattern = re.compile(r"\$\{([^}]+)\}")
    for raw_line in template_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        value = variable_pattern.sub(lambda match: template.get(match.group(1), os.environ.get(match.group(1), "")), value)
        template[key] = value
    return template


def discover_experiments(pattern: str, explicit: Iterable[str]) -> list[Path]:
    base = repo_root()
    paths: set[Path] = set()

    for match in base.glob(pattern):
        if match.is_dir():
            paths.add(match.resolve())

    for item in explicit:
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        if candidate.is_dir():
            paths.add(candidate)
            continue

        by_name = (base / "people_detection_pg2" / "results" / item).resolve()
        if by_name.is_dir():
            paths.add(by_name)
            continue

        raise FileNotFoundError(f"Experiment directory not found: {item}")

    return sorted(paths, key=lambda path: path.name)


def required_artifacts(experiment_dir: Path, backend: str) -> list[Path]:
    weights_dir = experiment_dir / "weights"
    required = [weights_dir / "best.pt"]
    if backend == "ultralytics_pt":
        return required
    if backend == "tflite_backbone_only":
        return required + [weights_dir / "best_non_separated_saved_model" / "best_full_integer_quant.tflite"]
    if backend == "tflite_mixed":
        return required + [
            weights_dir / "best_saved_model" / "best_full_integer_quant.tflite",
            weights_dir / "head_saved_model" / "head_float32.tflite",
        ]
    raise ValueError(f"Unsupported EVAL_BACKEND in template env: {backend}")


def select_dataset_name(experiment_dir: Path, use_baseline_test_set: bool) -> str:
    if use_baseline_test_set:
        return "batch_testAccuracy_margin_10_wb_alif_clahe"

    name = experiment_dir.name.lower()
    if "wb_alif" not in name:
        raise ValueError(
            f"Ambiguous dataset selection for {experiment_dir.name}: expected experiment name to contain 'wb_alif'."
        )
    if "clahe" in name:
        return "batch1_3_margin_15_wb_alif_clahe"
    return "batch1_3_margin_15_wb_alif"


def select_backend(evaluate_fp32: bool, evaluate_tflite_mixed: bool) -> str:
    if evaluate_fp32 and evaluate_tflite_mixed:
        raise ValueError("--evaluate-fp32 and --evaluate-tflite-mixed are mutually exclusive")
    if evaluate_fp32:
        return "ultralytics_pt"
    if evaluate_tflite_mixed:
        return "tflite_mixed"
    return "tflite_backbone_only"


def select_weights_path(experiment_dir: Path, backend: str) -> Path:
    weights_dir = experiment_dir / "weights"
    if backend == "ultralytics_pt":
        return weights_dir / "best.pt"
    if backend == "tflite_backbone_only":
        return weights_dir / "best_non_separated_saved_model" / "best_full_integer_quant.tflite"
    if backend == "tflite_mixed":
        return weights_dir / "best_saved_model" / "best_full_integer_quant.tflite"
    raise ValueError(f"Unsupported backend selection: {backend}")


def validate_common_inputs(run_script: Path, template_env: Path, holdout_dir: str) -> None:
    eval_root = extended_eval_root()
    required_paths = [
        run_script,
        template_env,
        eval_root / holdout_dir / "frames_attributes.csv",
        eval_root / holdout_dir / "detections_attributes.csv",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_str = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required inputs are missing:\n{missing_str}")


def build_env_text(
    *,
    template: dict[str, str],
    experiment_dir: Path,
    env_dir: Path,
    log_dir: Path,
    project_name: str,
    project: str,
    dataset_name: str,
    backend: str,
    holdout_dir: str,
) -> str:
    eval_root = extended_eval_root()
    dataset_root = Path("/home/klymy/data/cam4_revB/yolo_dataset") / dataset_name

    report_base = eval_root
    holdout_root = report_base / holdout_dir
    project_root = report_base / project / project_name
    base_experiment_dir = experiment_dir

    input_width = template.get("INPUT_WIDTH", "320")
    input_height = template.get("INPUT_HEIGHT", "320")
    subset_name = template.get("SUBSET_NAME", "holdout")
    use_inference_params = template.get("USE_INFERENCE_PARAMS", "1")
    conf_th = template.get("CONF_TH", "0.35")
    iou_th = template.get("IOU_TH", "0.50")
    enable_sweep = template.get("ENABLE_SWEEP", "1")
    sweep_conf_ths = template.get("SWEEP_CONF_THS", "0.20,0.25,0.30")
    sweep_iou_ths = template.get("SWEEP_IOU_THS", "0.35,0.40,0.45")
    selection_metric = template.get("SELECTION_METRIC", "f1")
    device = template.get("DEVICE", "cpu")
    batch = template.get("BATCH", "64")
    top_n = template.get("TOP_N", "6")
    f1_bins = template.get("F1_BINS", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    run_conclusions_agent = template.get("RUN_CONCLUSIONS_AGENT", "0")
    anthropic_api_key = template.get("ANTHROPIC_API_KEY", "")

    weights_dir = base_experiment_dir / "weights"
    pt_weights = weights_dir / "best.pt"
    if backend == "ultralytics_pt":
        tflite_backbone = ""
        tflite_head = ""
        correction_model = ""
    elif backend == "tflite_backbone_only":
        tflite_backbone = weights_dir / "best_non_separated_saved_model" / "best_full_integer_quant.tflite"
        tflite_head = ""
        correction_model = ""
    elif backend == "tflite_mixed":
        tflite_backbone = weights_dir / "best_saved_model" / "best_full_integer_quant.tflite"
        tflite_head = weights_dir / "head_saved_model" / "head_float32.tflite"
        correction_model = ""
    else:
        raise ValueError(f"Unsupported backend selection: {backend}")

    lines = [
        "# Auto-generated by people_detection_pg2/run_batch_evaluation_v2.py",
        "# Template source: people_detection_pg2/extended_evaluation_metrics/.env.H270_16-44-09.batch1_10",
        "# Required overrides for this workflow:",
        f"# - holdout attributes from extended_evaluation_metrics/{holdout_dir}",
        "# - dataset selected per experiment name (clahe => ..._clahe, otherwise ..._wb_alif)",
        "# - backend selected by CLI (--evaluate-fp32 => ultralytics_pt, else tflite_backbone_only)",
        "",
        "# ===== COMMON =====",
        f'REPORT_BASE="{report_base}"',
        f'FRAMES_ATTRS="{holdout_root / "frames_attributes.csv"}"',
        f'DETECTIONS_ATTRS="{holdout_root / "detections_attributes.csv"}"',
        "",
        "# ===== EVALUATION =====",
        f"INPUT_WIDTH={input_width}",
        f"INPUT_HEIGHT={input_height}",
        f"DATASET_NAME={dataset_name}",
        f"EXP_NAME={experiment_dir.name}",
        f'BASE_DATASET_PTH="{dataset_root}"',
        f"SUBSET_NAME={subset_name}",
        f'DATASET_PTH="{dataset_root / "project_holdout.yaml"}"',
        f'BASE_EXPERIMENT_DIR="{base_experiment_dir}"',
        f"EVAL_BACKEND={backend}",
        f'WEIGHTS_PTH="{pt_weights}"',
        f'USE_INFERENCE_PARAMS={use_inference_params}',
        f"CONF_TH={conf_th}",
        f"IOU_TH={iou_th}",
        f"ENABLE_SWEEP={enable_sweep}",
        f"SWEEP_CONF_THS={sweep_conf_ths}",
        f"SWEEP_IOU_THS={sweep_iou_ths}",
        f"SELECTION_METRIC={selection_metric}",
        f"DEVICE={device}",
        f"BATCH={batch}",
        "",
        f"PROJECT={project}",
        f"PROJECT_NAME={project_name}",
        f'PROJECT_PTH="{project_root}"',
        f'PREDICTIONS_PTH="{project_root / "predictions.json"}"',
        f'GLOBAL_METRICS_PTH="{project_root / "global_metrics.json"}"',
        f'SWEEP_RESULTS_PTH="{project_root / "sweep_results.csv"}"',
        f'BEST_PARAMS_PTH="{project_root / "best_params.json"}"',
        f'METRICS_PER_IMG="{project_root / "metrics_per_img.csv"}"',
        f'METRICS_PER_GT="{project_root / "metrics_per_gt.csv"}"',
        f'METRICS_PER_PRED="{project_root / "metrics_per_pred.csv"}"',
        "",
        "# ===== VISUALIZATION =====",
        f'YOLO_HOLDOUT="{dataset_root}"',
        f'VIZ_DIR="{project_root / "viz"}"',
        f'VIZ_OUT_DIR="{project_root / "viz"}"',
        "",
        "# ===== REPORT =====",
        f"TOP_N={top_n}",
        f"F1_BINS={f1_bins}",
        f"RUN_CONCLUSIONS_AGENT={run_conclusions_agent}",
        "",
        "# ===== ANTHROPIC API =====",
        f"ANTHROPIC_API_KEY={anthropic_api_key}",
        "",
        "# ===== BACKEND ARTIFACTS =====",
    ]

    if tflite_backbone:
        lines.append(f'TFLITE_BACKBONE_PTH="{tflite_backbone}"')
    if tflite_head:
        lines.append(f'TFLITE_HEAD_PTH="{tflite_head}"')
    if backend == "tflite_backbone_only":
        lines.append(f'TFLITE_OUTPUT_CORRECTION_MODEL_PTH="{correction_model}"')

    lines.extend(
        [
            "",
            "# ===== RUNNER METADATA =====",
            f'GENERATED_ENV_DIR="{env_dir}"',
            f'RUN_LOG_DIR="{log_dir}"',
            "",
        ]
    )

    return "\n".join(lines)


def prepare_experiment_configs(
    *,
    experiments: list[Path],
    template: dict[str, str],
    generated_env_dir: Path,
    log_dir: Path,
    project: str,
    backend: str,
    use_baseline_test_set: bool,
    holdout_dir: str,
) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []

    for experiment_dir in experiments:
        dataset_name = select_dataset_name(experiment_dir, use_baseline_test_set=use_baseline_test_set)
        dataset_yaml = Path("/home/klymy/data/cam4_revB/yolo_dataset") / dataset_name / "project_holdout.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(
                f"Dataset config not found for {experiment_dir.name}: {dataset_yaml}"
            )

        weights_path = select_weights_path(experiment_dir, backend)
        missing = [path for path in required_artifacts(experiment_dir, backend) if not path.exists()]
        if missing:
            missing_str = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Experiment {experiment_dir.name} is missing required artifacts: {missing_str}")

        project_name = f"{experiment_dir.name}__{dataset_name}__{backend}"
        env_file = generated_env_dir / f"{experiment_dir.name}.env"
        log_file = log_dir / f"{experiment_dir.name}.log"
        project_path = extended_eval_root() / project / project_name
        env_text = build_env_text(
            template=template,
            experiment_dir=experiment_dir,
            env_dir=generated_env_dir,
            log_dir=log_dir,
            project_name=project_name,
            project=project,
            dataset_name=dataset_name,
            backend=backend,
            holdout_dir=holdout_dir,
        )
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(env_text + "\n", encoding="utf-8")

        configs.append(
            ExperimentConfig(
                experiment_dir=experiment_dir,
                env_file=env_file,
                log_file=log_file,
                project_path=project_path,
                backend=backend,
                dataset_name=dataset_name,
                weights_path=weights_path,
            )
        )

    return configs


def run_experiment(config: ExperimentConfig, run_script: Path, pyenv_version: str) -> RunResult:
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if pyenv_version:
        env["PYENV_VERSION"] = pyenv_version

    command = [str(run_script), str(config.env_file)]
    with config.log_file.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {' '.join(command)}\n")
        if pyenv_version:
            log_handle.write(f"PYENV_VERSION={pyenv_version}\n")
        log_handle.write("\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(repo_root()),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    return RunResult(
        experiment_dir=config.experiment_dir,
        env_file=config.env_file,
        log_file=config.log_file,
        returncode=completed.returncode,
    )


def print_plan(configs: list[ExperimentConfig], dry_run: bool) -> None:
    mode = "DRY RUN" if dry_run else "PLAN"
    print(f"{mode}: {len(configs)} experiment(s)")
    for index, config in enumerate(configs, start=1):
        print(f"{index}. {config.experiment_dir.name}")
        print(f"   experiment: {config.experiment_dir}")
        print(f"   env:        {config.env_file}")
        print(f"   output:     {config.project_path}")
        print(f"   log:        {config.log_file}")
        print(f"   dataset:    {config.dataset_name}")
        print(f"   backend:    {config.backend}")
        print(f"   weights:    {config.weights_path}")


def print_summary(results: list[RunResult]) -> None:
    print("\nSummary")
    for result in results:
        status = "OK" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"- {result.experiment_dir.name}: {status}")
        print(f"  env: {result.env_file}")
        print(f"  log: {result.log_file}")


def main() -> int:
    args = parse_args()

    # Apply suffix to project name if provided
    project = args.project
    if args.suffix:
        project = f"{project}_{args.suffix}"

    template_env = resolve_repo_path(args.template_env)
    run_script = resolve_repo_path(args.run_script)
    generated_env_dir = resolve_repo_path(args.generated_env_dir)
    log_dir = resolve_repo_path(args.log_dir)

    validate_common_inputs(run_script=run_script, template_env=template_env, holdout_dir=args.holdout_dir)
    template = load_template_env(template_env)
    backend = select_backend(args.evaluate_fp32, args.evaluate_tflite_mixed)
    experiments = discover_experiments(args.pattern, args.experiment)
    if args.limit > 0:
        experiments = experiments[: args.limit]
    if not experiments:
        print("No matching experiments found.", file=sys.stderr)
        return 2

    configs = prepare_experiment_configs(
        experiments=experiments,
        template=template,
        generated_env_dir=generated_env_dir,
        log_dir=log_dir,
        project=project,
        backend=backend,
        use_baseline_test_set=args.use_baseline_test_set,
        holdout_dir=args.holdout_dir,
    )

    print(f"Baseline test set mode: {'enabled' if args.use_baseline_test_set else 'disabled'}")
    print(f"Selected evaluation backend: {backend}")
    print_plan(configs, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    results: list[RunResult] = []
    for config in configs:
        print(f"\nRunning {config.experiment_dir.name}")
        print(f"Dataset: {config.dataset_name}")
        print(f"Backend: {config.backend}")
        print(f"Weights: {config.weights_path}")
        result = run_experiment(config, run_script=run_script, pyenv_version=args.python_env)
        results.append(result)
        status = "OK" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"Completed {config.experiment_dir.name}: {status}")
        print(f"Log: {result.log_file}")
        if result.returncode != 0 and args.stop_on_error:
            print_summary(results)
            return result.returncode

    print_summary(results)
    failures = [result for result in results if result.returncode != 0]
    return failures[0].returncode if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
