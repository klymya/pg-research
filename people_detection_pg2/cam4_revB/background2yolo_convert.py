#!/usr/bin/env python3
"""Build a YOLO dataset from background-only (no-person) image folders.

Every image gets an empty label file and goes to ``train``; optionally
``--val-count`` images are drawn at random (seeded) into ``val``. Sample ids follow
cvat2yolo_convert (``<folder>____<stem>``), so backgrounds from different
folders never collide and match the ids already used by
regenerate_yolo_datasets_v2.

Output is plain converted images (no WB/CLAHE); run preprocess_dataset.py on
the result, e.g.:

    python background2yolo_convert.py --image-root /home/klymy/data/cam4_revB/4.2_background_v2 \
        --exclude-root /home/klymy/data/cam4_revB/4.2_background --dataset-name background_v2
    python preprocess_dataset.py --base-pth /home/klymy/data/cam4_revB/yolo_dataset_v2 \
        --base-dataset background_v2 --mode wb_alif_clahe --no-sharpness-augmentation
"""
from __future__ import annotations

import argparse
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from cvat2yolo_convert import DEFAULT_ALLOWED_EXTENSIONS, build_sample_id, relative_key_from_path
from pipeline_dataset_utils import refresh_project_files, write_json


DEFAULT_ROOT_PTH = Path("/home/klymy/data/cam4_revB")
DEFAULT_YOLO_OUTPUT_DIRNAME = "yolo_dataset_v2"
DEFAULT_OUTPUT_IMAGE_EXT = "jpg"
DEFAULT_MAX_WORKERS = 8
DEFAULT_VAL_COUNT = 0
DEFAULT_SEED = 42
DATASET_SUBFOLDERS = ("images/train", "images/val", "labels/train", "labels/val")


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(item) for item in value.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected WIDTHxHEIGHT, got {value!r}") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"Size must be positive, got {value!r}")
    return width, height


def discover_images(image_root: Path) -> list[Path]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    return sorted(
        path
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in DEFAULT_ALLOWED_EXTENSIONS
    )


def collect_excluded_stems(exclude_roots: list[Path]) -> set[str]:
    return {path.stem for root in exclude_roots for path in discover_images(root)}


def _write_sample(args) -> dict[str, Any]:
    src_path, sample_id, split, dataset_path, output_image_ext, resize = args
    image_path = dataset_path / "images" / split / f"{sample_id}.{output_image_ext}"
    label_path = dataset_path / "labels" / split / f"{sample_id}.txt"

    with Image.open(src_path) as image:
        source_size = image.size
        image = image.convert("RGB")
        if resize is not None and image.size != resize:
            image = image.resize(resize, Image.BILINEAR)
        save_kwargs: dict[str, Any] = {}
        if output_image_ext.lower() in {"jpg", "jpeg"}:
            save_kwargs["quality"] = 95
        image.save(image_path, **save_kwargs)
        output_size = image.size
    label_path.write_text("")

    return {
        "source_path": str(src_path),
        "sample_id": sample_id,
        "split": split,
        "source_size": list(source_size),
        "output_size": list(output_size),
    }


def run_conversion(
    *,
    root_path: Path,
    image_roots: list[Path],
    output_dataset_path: Path,
    exclude_roots: list[Path] | None = None,
    resize: tuple[int, int] | None = None,
    output_image_ext: str = DEFAULT_OUTPUT_IMAGE_EXT,
    max_workers: int = DEFAULT_MAX_WORKERS,
    overwrite: bool = False,
    val_count: int = DEFAULT_VAL_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if val_count < 0:
        raise ValueError(f"val_count must be >= 0, got {val_count}")

    excluded_stems = collect_excluded_stems(exclude_roots or [])
    root_resolved = root_path.resolve(strict=False)

    sources: list[tuple[Path, str]] = []
    skipped: list[str] = []
    seen_ids: dict[str, str] = {}
    for image_root in image_roots:
        for src_path in discover_images(image_root):
            if src_path.stem in excluded_stems:
                skipped.append(str(src_path))
                continue
            relative_key = relative_key_from_path(root_resolved, src_path.resolve(strict=False))
            sample_id = build_sample_id(relative_key)
            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate sample id {sample_id!r}: {seen_ids[sample_id]} and {src_path}"
                )
            seen_ids[sample_id] = str(src_path)
            sources.append((src_path, sample_id))

    if not sources:
        raise ValueError(f"No background images to convert under {[str(p) for p in image_roots]}")
    if val_count >= len(sources):
        raise ValueError(f"val_count={val_count} leaves no train images out of {len(sources)}")

    # Sample from the sorted id list so the split depends only on seed and file set.
    val_ids = set(random.Random(seed).sample(sorted(sid for _, sid in sources), val_count))
    tasks = [
        (src_path, sample_id, "val" if sample_id in val_ids else "train", output_dataset_path, output_image_ext, resize)
        for src_path, sample_id in sources
    ]

    if output_dataset_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {output_dataset_path}")
        shutil.rmtree(output_dataset_path)
    for subfolder in DATASET_SUBFOLDERS:
        (output_dataset_path / subfolder).mkdir(parents=True, exist_ok=True)

    if max_workers == 1:
        samples = [_write_sample(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            samples = list(executor.map(_write_sample, tasks))

    written_project_files = refresh_project_files(output_dataset_path)
    source_sizes: dict[str, int] = {}
    for sample in samples:
        key = "x".join(str(v) for v in sample["source_size"])
        source_sizes[key] = source_sizes.get(key, 0) + 1

    summary = {
        "config": {
            "root_path": str(root_path),
            "image_roots": [str(p) for p in image_roots],
            "exclude_roots": [str(p) for p in exclude_roots or []],
            "output_dataset_path": str(output_dataset_path),
            "resize": list(resize) if resize else None,
            "output_image_ext": output_image_ext,
            "val_count": val_count,
            "seed": seed,
        },
        "counts": {
            "background_images_written": len(samples),
            "train": sum(1 for sample in samples if sample["split"] == "train"),
            "val": sum(1 for sample in samples if sample["split"] == "val"),
            "skipped_excluded": len(skipped),
        },
        "source_sizes": source_sizes,
        "skipped_excluded": skipped,
        "val_sample_ids": sorted(val_ids),
        "samples": samples,
        "project_files_written": written_project_files,
    }
    write_json(output_dataset_path / "background_conversion_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert background-only image folders into a YOLO dataset with empty label files (train only)."
    )
    parser.add_argument("--root-pth", default=str(DEFAULT_ROOT_PTH), help=f"Default: {DEFAULT_ROOT_PTH}")
    parser.add_argument(
        "--image-root",
        action="append",
        required=True,
        help="Background image folder (under --root-pth). Repeatable.",
    )
    parser.add_argument(
        "--exclude-root",
        action="append",
        default=[],
        help="Skip images whose file stem already exists in this folder (e.g. an older background batch). Repeatable.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Output dataset name under <root_pth>/<yolo_output_dirname>.",
    )
    parser.add_argument(
        "--yolo-output-dirname",
        default=DEFAULT_YOLO_OUTPUT_DIRNAME,
        help=f"Output directory name under root. Default: {DEFAULT_YOLO_OUTPUT_DIRNAME}",
    )
    parser.add_argument(
        "--resize",
        type=parse_size,
        help="Optional WIDTHxHEIGHT to resize every image to, e.g. 640x480. Default: keep source size.",
    )
    parser.add_argument(
        "--output-image-ext",
        default=DEFAULT_OUTPUT_IMAGE_EXT,
        help=f"Output image extension without dot. Default: {DEFAULT_OUTPUT_IMAGE_EXT}",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum worker threads. Default: {DEFAULT_MAX_WORKERS}",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=DEFAULT_VAL_COUNT,
        help=f"Number of images drawn at random into val (rest go to train). Default: {DEFAULT_VAL_COUNT}",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for the val draw. Default: {DEFAULT_SEED}")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output dataset if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = Path(args.root_pth)
    output_dataset_path = root_path / args.yolo_output_dirname / args.dataset_name

    summary = run_conversion(
        root_path=root_path,
        image_roots=[Path(p) for p in args.image_root],
        output_dataset_path=output_dataset_path,
        exclude_roots=[Path(p) for p in args.exclude_root],
        resize=args.resize,
        output_image_ext=args.output_image_ext.lstrip("."),
        max_workers=args.max_workers,
        overwrite=args.overwrite,
        val_count=args.val_count,
        seed=args.seed,
    )

    print(
        f"Wrote {summary['counts']['background_images_written']} background images "
        f"(train {summary['counts']['train']}, val {summary['counts']['val']}) "
        f"(skipped {summary['counts']['skipped_excluded']} excluded) into {output_dataset_path}"
    )
    print(f"Source sizes: {summary['source_sizes']}")
    print(f"Summary: {output_dataset_path / 'background_conversion_summary.json'}")


if __name__ == "__main__":
    main()
