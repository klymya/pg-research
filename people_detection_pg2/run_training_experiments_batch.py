"""Run YOLO training pipeline experiments in parallel across multiple GPUs.

This scheduler reads experiment definitions from JSON, assigns one long-lived
worker process per GPU, and executes experiments sequentially inside each
worker. Each experiment is launched through ``run_training_pipeline.py`` with
``--device`` set automatically to the worker's GPU.

Example:
    python people_detection_pg2/run_training_experiments_batch.py \
        --experiments-json experiments.json \
        --gpus 0,1,2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any


DEFAULT_EXPERIMENTS_JSON = "experiments.json"
DEFAULT_GPUS = "0,1,2"
DEFAULT_MPLBACKEND = "Agg"
WORKER_POLL_TIMEOUT_SECONDS = 1.0
RESULTS_POLL_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Normalized experiment definition loaded from JSON."""

    exp_name: str
    dataset: str
    weights: str
    cfg: str = ""
    yolo_kwargs: str = ""
    max_ncalib_imgs: int = 100
    converted_output_folder_name: str = "best_saved_model"
    batch: int = 32
    epochs: int = 1000
    patience: int = 50


@dataclass(frozen=True, slots=True)
class WorkerTask:
    """Task consumed by a GPU worker."""

    index: int
    total: int
    experiment: ExperimentConfig
    log_path: str


@dataclass(frozen=True, slots=True)
class WorkerEvent:
    """Event emitted by workers back to the parent process."""

    event_type: str
    index: int
    total: int
    exp_name: str
    gpu_id: int
    log_path: str
    command: list[str]
    started_at: float
    finished_at: float | None = None
    returncode: int | None = None
    error_message: str | None = None


class ExperimentBatchError(Exception):
    """Raised when the batch runner configuration is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run run_training_pipeline.py experiments in parallel with one "
            "persistent worker per GPU."
        )
    )
    parser.add_argument(
        "--experiments-json",
        type=Path,
        default=Path(DEFAULT_EXPERIMENTS_JSON),
        help="Path to JSON array with experiment definitions.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=DEFAULT_GPUS,
        help="Comma-separated list of GPU indexes, for example 0,1,2.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs" / "training_batch_runs",
        help="Directory where per-experiment logs are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned commands without executing them.",
    )
    return parser.parse_args()


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    if not sanitized:
        raise ExperimentBatchError(f"Experiment name {value!r} cannot be sanitized into a filename.")
    return sanitized


def parse_gpu_list(raw_value: str) -> list[int]:
    gpu_values: list[int] = []
    seen: set[int] = set()
    for token in raw_value.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        if not cleaned.isdigit():
            raise ExperimentBatchError(
                f"Invalid GPU index {cleaned!r}; expected a comma-separated list of integers."
            )
        gpu_id = int(cleaned)
        if gpu_id in seen:
            raise ExperimentBatchError(f"Duplicate GPU index {gpu_id} is not allowed.")
        seen.add(gpu_id)
        gpu_values.append(gpu_id)
    if not gpu_values:
        raise ExperimentBatchError("At least one GPU index must be provided via --gpus.")
    return gpu_values


def load_json_file(json_path: Path) -> Any:
    if not json_path.is_file():
        raise ExperimentBatchError(f"Experiments JSON file not found: {json_path}")
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentBatchError(f"Failed to parse JSON from {json_path}: {exc}") from exc


def validate_positive_int(value: Any, field_name: str, exp_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExperimentBatchError(
            f"Experiment {exp_name!r} has invalid {field_name!r}: expected a positive integer."
        )
    return value


def validate_string(value: Any, field_name: str, exp_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ExperimentBatchError(
            f"Experiment {exp_name!r} has invalid {field_name!r}: expected a string."
        )
    if not allow_empty and not value.strip():
        raise ExperimentBatchError(f"Experiment {exp_name!r} has empty required field {field_name!r}.")
    return value


def normalize_experiment(entry: Any, index: int) -> ExperimentConfig:
    if not isinstance(entry, dict):
        raise ExperimentBatchError(f"Experiment at index {index} must be a JSON object.")

    exp_name = validate_string(entry.get("exp_name"), "exp_name", f"index {index}")
    dataset = validate_string(entry.get("dataset"), "dataset", exp_name)
    weights = validate_string(entry.get("weights"), "weights", exp_name)
    cfg = validate_string(entry.get("cfg", ""), "cfg", exp_name, allow_empty=True)
    yolo_kwargs = validate_string(entry.get("yolo_kwargs", ""), "yolo_kwargs", exp_name, allow_empty=True)
    max_ncalib_imgs = validate_positive_int(entry.get("max_ncalib_imgs", 100), "max_ncalib_imgs", exp_name)

    converted_output_folder_name = entry.get(
        "converted-output-folder-name",
        entry.get("converted_output_folder_name", "best_saved_model"),
    )
    converted_output_folder_name = validate_string(
        converted_output_folder_name,
        "converted-output-folder-name",
        exp_name,
    )

    batch = validate_positive_int(entry.get("batch", 32), "batch", exp_name)
    epochs = validate_positive_int(entry.get("epochs", 1000), "epochs", exp_name)
    patience = validate_positive_int(entry.get("patience", 50), "patience", exp_name)

    unexpected_keys = sorted(
        set(entry) - {
            "exp_name",
            "dataset",
            "weights",
            "cfg",
            "yolo_kwargs",
            "max_ncalib_imgs",
            "converted-output-folder-name",
            "converted_output_folder_name",
            "batch",
            "epochs",
            "patience",
        }
    )
    if unexpected_keys:
        raise ExperimentBatchError(
            f"Experiment {exp_name!r} contains unsupported keys: {', '.join(unexpected_keys)}"
        )

    return ExperimentConfig(
        exp_name=exp_name,
        dataset=dataset,
        weights=weights,
        cfg=cfg,
        yolo_kwargs=yolo_kwargs,
        max_ncalib_imgs=max_ncalib_imgs,
        converted_output_folder_name=converted_output_folder_name,
        batch=batch,
        epochs=epochs,
        patience=patience,
    )


def load_experiments(json_path: Path) -> list[ExperimentConfig]:
    payload = load_json_file(json_path)
    if not isinstance(payload, list):
        raise ExperimentBatchError(f"Experiments JSON must contain a top-level array: {json_path}")
    if not payload:
        raise ExperimentBatchError(f"Experiments JSON is empty: {json_path}")

    experiments = [normalize_experiment(entry, index) for index, entry in enumerate(payload)]
    exp_names = [experiment.exp_name for experiment in experiments]
    if len(exp_names) != len(set(exp_names)):
        raise ExperimentBatchError("Experiment exp_name values must be unique.")
    return experiments


def resolve_log_paths(experiments: list[ExperimentConfig], log_dir: Path) -> dict[str, Path]:
    log_paths: dict[str, Path] = {}
    used_names: set[str] = set()
    for experiment in experiments:
        log_name = f"{sanitize_filename(experiment.exp_name)}.log"
        if log_name in used_names:
            raise ExperimentBatchError(
                "Sanitized experiment names collide; rename experiments so each log filename is unique."
            )
        used_names.add(log_name)
        log_paths[experiment.exp_name] = log_dir / log_name
    return log_paths


def build_command(pipeline_path: Path, experiment: ExperimentConfig, gpu_id: int) -> list[str]:
    command = [
        sys.executable,
        str(pipeline_path),
        "--dataset",
        experiment.dataset,
        "--exp_name",
        experiment.exp_name,
        "--weights",
        experiment.weights,
        "--device",
        str(gpu_id),
        "--batch",
        str(experiment.batch),
        "--yolo_kwargs",
        experiment.yolo_kwargs,
        "--max_ncalib_imgs",
        str(experiment.max_ncalib_imgs),
        "--converted-output-folder-name",
        experiment.converted_output_folder_name,
        "--epochs",
        str(experiment.epochs),
        "--patience",
        str(experiment.patience),
    ]
    if experiment.cfg:
        command.extend(["--cfg", experiment.cfg])
    return command


def write_log_header(log_handle: Any, command: list[str], gpu_id: int, dry_run: bool) -> None:
    log_handle.write(f"gpu_id={gpu_id}\n")
    log_handle.write(f"dry_run={str(dry_run).lower()}\n")
    log_handle.write(f"matplotlib_backend={DEFAULT_MPLBACKEND}\n")
    log_handle.write(f"command={' '.join(command)}\n")
    log_handle.write(f"started_at={time.strftime('%Y-%m-%d %H:%M:%S %z')}\n")
    log_handle.write("\n")
    log_handle.flush()


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MPLBACKEND"] = DEFAULT_MPLBACKEND
    return env


def worker_loop(
    gpu_id: int,
    task_queue: mp.Queue[Any],
    event_queue: mp.Queue[Any],
    pipeline_path: str,
    project_root: str,
    dry_run: bool,
) -> None:
    current_process: subprocess.Popen[Any] | None = None

    def handle_shutdown(signum: int, _frame: Any) -> None:
        nonlocal current_process
        if current_process is not None:
            terminate_process_group(current_process)
        raise KeyboardInterrupt(f"Worker on GPU {gpu_id} received signal {signum}.")

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    while True:
        try:
            task = task_queue.get(timeout=WORKER_POLL_TIMEOUT_SECONDS)
        except Empty:
            continue

        if task is None:
            return
        if not isinstance(task, WorkerTask):
            raise ExperimentBatchError(f"Worker on GPU {gpu_id} received unexpected task payload: {task!r}")

        command = build_command(Path(pipeline_path), task.experiment, gpu_id)
        started_at = time.time()
        event_queue.put(
            WorkerEvent(
                event_type="started",
                index=task.index,
                total=task.total,
                exp_name=task.experiment.exp_name,
                gpu_id=gpu_id,
                log_path=task.log_path,
                command=command,
                started_at=started_at,
            )
        )

        log_path = Path(task.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
                write_log_header(log_handle, command, gpu_id, dry_run)

                if dry_run:
                    log_handle.write("Dry run: command not executed.\n")
                    event_queue.put(
                        WorkerEvent(
                            event_type="dry-run",
                            index=task.index,
                            total=task.total,
                            exp_name=task.experiment.exp_name,
                            gpu_id=gpu_id,
                            log_path=str(log_path),
                            command=command,
                            started_at=started_at,
                            finished_at=time.time(),
                            returncode=0,
                        )
                    )
                    continue

                current_process = subprocess.Popen(
                    command,
                    cwd=project_root,
                    env=build_subprocess_env(),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                returncode = current_process.wait()
                finished_at = time.time()
                log_handle.write(f"\nfinished_at={time.strftime('%Y-%m-%d %H:%M:%S %z')}\n")
                log_handle.write(f"returncode={returncode}\n")
                log_handle.flush()

                event_queue.put(
                    WorkerEvent(
                        event_type="succeeded" if returncode == 0 else "failed",
                        index=task.index,
                        total=task.total,
                        exp_name=task.experiment.exp_name,
                        gpu_id=gpu_id,
                        log_path=str(log_path),
                        command=command,
                        started_at=started_at,
                        finished_at=finished_at,
                        returncode=returncode,
                    )
                )
        except KeyboardInterrupt:
            if current_process is not None:
                terminate_process_group(current_process)
            event_queue.put(
                WorkerEvent(
                    event_type="interrupted",
                    index=task.index,
                    total=task.total,
                    exp_name=task.experiment.exp_name,
                    gpu_id=gpu_id,
                    log_path=str(log_path),
                    command=command,
                    started_at=started_at,
                    finished_at=time.time(),
                    error_message="Worker interrupted during execution.",
                )
            )
            return
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            event_queue.put(
                WorkerEvent(
                    event_type="failed",
                    index=task.index,
                    total=task.total,
                    exp_name=task.experiment.exp_name,
                    gpu_id=gpu_id,
                    log_path=str(log_path),
                    command=command,
                    started_at=started_at,
                    finished_at=time.time(),
                    error_message=str(exc),
                )
            )
        finally:
            current_process = None


def format_duration(started_at: float, finished_at: float | None) -> str:
    if finished_at is None:
        return "n/a"
    duration_seconds = max(0.0, finished_at - started_at)
    minutes, seconds = divmod(int(duration_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_started_event(event: WorkerEvent) -> None:
    print(
        f"[start {event.index + 1}/{event.total}] gpu={event.gpu_id} "
        f"exp={event.exp_name} log={event.log_path}"
    )


def print_finished_event(event: WorkerEvent) -> None:
    status = event.event_type
    duration = format_duration(event.started_at, event.finished_at)
    suffix = f" returncode={event.returncode}" if event.returncode is not None else ""
    if event.error_message:
        suffix += f" error={event.error_message}"
    print(
        f"[{status} {event.index + 1}/{event.total}] gpu={event.gpu_id} "
        f"exp={event.exp_name} duration={duration}{suffix} log={event.log_path}"
    )


def stop_workers(workers: list[mp.Process], task_queue: mp.Queue[Any]) -> None:
    for _ in workers:
        task_queue.put(None)
    for worker in workers:
        worker.join(timeout=5)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=5)


def run_batch(
    experiments: list[ExperimentConfig],
    gpu_ids: list[int],
    log_dir: Path,
    pipeline_path: Path,
    project_root: Path,
    dry_run: bool,
) -> dict[str, list[WorkerEvent]]:
    context = mp.get_context("spawn")
    task_queue: mp.Queue[Any] = context.Queue()
    event_queue: mp.Queue[Any] = context.Queue()

    log_paths = resolve_log_paths(experiments, log_dir)
    total = len(experiments)
    for index, experiment in enumerate(experiments):
        task_queue.put(
            WorkerTask(
                index=index,
                total=total,
                experiment=experiment,
                log_path=str(log_paths[experiment.exp_name]),
            )
        )

    workers: list[mp.Process] = []
    for gpu_id in gpu_ids:
        process = context.Process(
            target=worker_loop,
            args=(gpu_id, task_queue, event_queue, str(pipeline_path), str(project_root), dry_run),
            name=f"gpu-worker-{gpu_id}",
        )
        process.start()
        workers.append(process)

    final_events: dict[str, list[WorkerEvent]] = {"succeeded": [], "failed": [], "dry-run": [], "interrupted": []}
    completed = 0

    try:
        while completed < total:
            try:
                event = event_queue.get(timeout=RESULTS_POLL_TIMEOUT_SECONDS)
            except Empty:
                dead_workers = [worker for worker in workers if worker.exitcode not in (None, 0)]
                if dead_workers:
                    dead_descriptions = ", ".join(
                        f"{worker.name}(exitcode={worker.exitcode})" for worker in dead_workers
                    )
                    raise ExperimentBatchError(
                        "One or more GPU workers exited unexpectedly before the batch completed: "
                        f"{dead_descriptions}"
                    )
                continue

            if not isinstance(event, WorkerEvent):
                raise ExperimentBatchError(f"Received unexpected worker event: {event!r}")

            if event.event_type == "started":
                print_started_event(event)
                continue

            if event.event_type not in final_events:
                raise ExperimentBatchError(f"Received unsupported worker event type: {event.event_type}")

            final_events[event.event_type].append(event)
            completed += 1
            print_finished_event(event)
    except KeyboardInterrupt as exc:
        stop_workers(workers, task_queue)
        raise ExperimentBatchError("Batch execution interrupted by user.") from exc
    finally:
        stop_workers(workers, task_queue)

    return final_events


def print_summary(final_events: dict[str, list[WorkerEvent]]) -> None:
    succeeded = len(final_events["succeeded"])
    failed = len(final_events["failed"])
    dry_run = len(final_events["dry-run"])
    interrupted = len(final_events["interrupted"])
    skipped = dry_run

    print("\nSummary")
    print(f"  succeeded: {succeeded}")
    print(f"  failed: {failed}")
    print(f"  skipped: {skipped}")
    print(f"  interrupted: {interrupted}")

    if failed:
        print("\nFailed experiments")
        for event in final_events["failed"]:
            message = event.error_message or f"returncode={event.returncode}"
            print(f"  - {event.exp_name} gpu={event.gpu_id} {message} log={event.log_path}")


def validate_runtime_paths(pipeline_path: Path, project_root: Path) -> None:
    if not pipeline_path.is_file():
        raise ExperimentBatchError(f"Training pipeline entrypoint not found: {pipeline_path}")
    if not project_root.is_dir():
        raise ExperimentBatchError(f"Project root not found: {project_root}")


def validate_experiment_runtime_inputs(experiments: list[ExperimentConfig], project_root: Path) -> None:
    for experiment in experiments:
        dataset_path = Path(experiment.dataset)
        if not dataset_path.is_dir():
            raise ExperimentBatchError(
                f"Experiment {experiment.exp_name!r} dataset directory not found: {dataset_path}"
            )

        weights_path = Path(experiment.weights)
        if not weights_path.is_file():
            raise ExperimentBatchError(
                f"Experiment {experiment.exp_name!r} weights file not found: {weights_path}"
            )

        if experiment.cfg:
            cfg_path = Path(experiment.cfg)
            resolved_cfg_path = cfg_path if cfg_path.is_absolute() else project_root / cfg_path
            if not resolved_cfg_path.is_file():
                raise ExperimentBatchError(
                    f"Experiment {experiment.exp_name!r} cfg file not found: {resolved_cfg_path}"
                )


def main() -> int:
    args = parse_args()
    pipeline_path = Path(__file__).resolve().parent / "run_training_pipeline.py"
    project_root = pipeline_path.parent
    validate_runtime_paths(pipeline_path, project_root)

    experiments = load_experiments(args.experiments_json)
    gpu_ids = parse_gpu_list(args.gpus)
    validate_experiment_runtime_inputs(experiments, project_root)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    final_events = run_batch(
        experiments=experiments,
        gpu_ids=gpu_ids,
        log_dir=args.log_dir,
        pipeline_path=pipeline_path,
        project_root=project_root,
        dry_run=args.dry_run,
    )
    print_summary(final_events)

    if final_events["failed"] or final_events["interrupted"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentBatchError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
