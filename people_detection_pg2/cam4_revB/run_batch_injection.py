#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import antileakage_split
import cvat2yolo_convert
import filter_small_bboxes
import merge_yolo_datasets
import preprocess_dataset
from pipeline_dataset_utils import write_json


DEFAULT_ROOT_PTH = Path("/home/klymy/data/cam4_revB")


def build_run_manifest(run_name: str, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "config": {
            "root_path": str(Path(args.root_pth).resolve(strict=False)),
            "new_batch_roots": [str(Path(path).resolve(strict=False)) for path in args.new_batch_root],
            "protected_subsets_jsons": [str(Path(path).resolve(strict=False)) for path in args.protected_subsets_json],
            "annotation_xmls": [str(Path(path).resolve(strict=False)) for path in args.annotation_xml],
            "image_roots": [str(Path(path).resolve(strict=False)) for path in (args.image_root or args.new_batch_root)],
            "converted_dataset_name": args.converted_dataset_name,
            "filtered_dataset_name": args.filtered_dataset_name,
            "preprocess_mode": args.preprocess_mode,
            "preprocessed_dataset_name": args.preprocessed_dataset_name,
            "merge_base_dataset": args.merge_base_dataset,
            "merge_result_dataset": args.merge_result_dataset,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "margin": args.margin,
            "min_inside_ratio": args.min_inside_ratio,
            "max_workers": args.max_workers,
            "fail_on_issues": args.fail_on_issues,
            "overwrite": args.overwrite,
        },
        "review": {
            "confirmed": args.review_confirmed,
            "note": args.review_note,
        },
        "steps": {},
        "final_outputs": {},
        "status": "running",
    }


def update_step(manifest: dict[str, Any], step_name: str, status: str, payload: dict[str, Any]) -> None:
    manifest["steps"][step_name] = {"status": status, **payload}


def enforce_review_gate(args: argparse.Namespace) -> None:
    if not args.review_confirmed:
        raise ValueError("Post-review pipeline requires --review-confirmed")
    if not args.review_note.strip():
        raise ValueError("Post-review pipeline requires a non-empty --review-note")


def run_antileakage_stage(
    new_batch_roots: list[str],
    protected_subset_jsons: list[str],
    run_dir: Path,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    new_paths = antileakage_split.discover_paths(new_batch_roots)
    protected_groups, source_to_groups, malformed_protected = antileakage_split.collect_protected_holdout_groups(
        protected_subset_jsons
    )
    split_data, malformed_new = antileakage_split.split_new_batch(
        new_paths=new_paths,
        protected_holdout_groups=protected_groups,
        val_ratio=val_ratio,
        seed=seed,
    )
    malformed_paths = malformed_protected + malformed_new
    validation = antileakage_split.validate_split(
        split_data,
        protected_groups,
        malformed_paths=malformed_paths,
        target_val_ratio=val_ratio,
    )

    overlap_report = {
        "new_batch_roots": new_batch_roots,
        "protected_subset_jsons": protected_subset_jsons,
        "protected_holdout_group_count": len(protected_groups),
        "protected_holdout_groups_sample": sorted(protected_groups)[:50],
        "protected_groups_by_source": source_to_groups,
        "matched_new_holdout_groups": sorted(
            {antileakage_split.extract_raw_scene_group_id(path) for path in split_data["holdout"]}
        ),
        "matched_new_holdout_frame_count": len(split_data["holdout"]),
        "malformed_protected_paths": malformed_protected,
        "malformed_new_paths": malformed_new,
    }
    split_summary = {
        "config": {
            "new_batch_roots": new_batch_roots,
            "protected_subset_jsons": protected_subset_jsons,
            "val_ratio": val_ratio,
            "seed": seed,
        },
        "validation": validation,
    }

    subsets_path = run_dir / "split_subsets.json"
    overlap_path = run_dir / "split_overlap.json"
    summary_path = run_dir / "split_summary.json"
    antileakage_split.write_json(str(subsets_path), split_data)
    antileakage_split.write_json(str(overlap_path), overlap_report)
    antileakage_split.write_json(str(summary_path), split_summary)

    if not validation["is_valid"]:
        raise ValueError(f"Anti-leakage split validation failed: {validation}")

    return {
        "subsets_path": str(subsets_path),
        "overlap_path": str(overlap_path),
        "summary_path": str(summary_path),
        "validation": validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full post-review batch injection pipeline for cam4_revB."
    )
    parser.add_argument("--root-pth", default=str(DEFAULT_ROOT_PTH), help=f"Default: {DEFAULT_ROOT_PTH}")
    parser.add_argument("--run-name", required=True, help="Logical run name used for manifest and step artifacts.")
    parser.add_argument("--new-batch-root", action="append", required=True, help="Raw batch folder. Repeatable.")
    parser.add_argument(
        "--protected-subsets-json",
        action="append",
        required=True,
        help="Historical protected subsets JSON. Repeatable.",
    )
    parser.add_argument("--annotation-xml", action="append", required=True, help="Corrected CVAT XML. Repeatable.")
    parser.add_argument("--image-root", action="append", help="Image root for conversion. Defaults to --new-batch-root.")
    parser.add_argument("--review-confirmed", action="store_true", help="Required gate after manual Step-0 review.")
    parser.add_argument("--review-note", required=True, help="Free-text note confirming review status.")
    parser.add_argument("--converted-dataset-name", required=True, help="Dataset name for the conversion stage output.")
    parser.add_argument(
        "--filtered-dataset-name",
        help="Optional filtered dataset name. Defaults to <converted>_margin_<int(margin*100)>.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["none", "wb_alif", "wb_alif_clahe"],
        default="none",
        help="Optional dataset preprocessing mode.",
    )
    parser.add_argument(
        "--preprocessed-dataset-name",
        help="Optional preprocessing output dataset name. Defaults to mode-based suffix if preprocessing is enabled.",
    )
    parser.add_argument("--merge-base-dataset", help="Optional base dataset name for merge stage.")
    parser.add_argument("--merge-result-dataset", help="Optional merged dataset result name.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio for anti-leakage split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for anti-leakage split.")
    parser.add_argument("--margin", type=float, default=0.10, help="Border filtering margin.")
    parser.add_argument("--min-inside-ratio", type=float, default=0.01, help="Border filtering threshold.")
    parser.add_argument("--max-workers", type=int, default=8, help="Worker threads for step scripts.")
    parser.add_argument("--fail-on-issues", action="store_true", help="Enable fail-fast validation in step scripts.")
    parser.add_argument("--overwrite", action="store_true", help="Allow step outputs to overwrite existing datasets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_review_gate(args)

    root_path = Path(args.root_pth).resolve(strict=False)
    yolo_base_path = root_path / "yolo_dataset_v2"
    run_dir = root_path / "batch_injection_runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "batch_injection_run_manifest.json"
    manifest = build_run_manifest(args.run_name, args, run_dir)
    write_json(manifest_path, manifest)

    try:
        split_stage = run_antileakage_stage(
            new_batch_roots=args.new_batch_root,
            protected_subset_jsons=args.protected_subsets_json,
            run_dir=run_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        update_step(manifest, "anti_leakage_split", "completed", split_stage)
        write_json(manifest_path, manifest)

        conversion_summary = cvat2yolo_convert.run_conversion(
            root_path=root_path,
            annotation_xmls=[Path(path) for path in args.annotation_xml],
            image_roots=[Path(path) for path in (args.image_root or args.new_batch_root)],
            subsets_json_path=Path(split_stage["subsets_path"]),
            output_dataset_path=yolo_base_path / args.converted_dataset_name,
            overwrite=args.overwrite,
            output_image_ext="jpg",
            max_workers=args.max_workers,
            fail_on_issues=args.fail_on_issues,
        )
        update_step(
            manifest,
            "cvat_to_yolo_conversion",
            "completed",
            {
                "output_dataset_path": str(yolo_base_path / args.converted_dataset_name),
                "summary_path": str(yolo_base_path / args.converted_dataset_name / "conversion_summary.json"),
                "counts": conversion_summary["counts"],
            },
        )
        write_json(manifest_path, manifest)

        filtered_dataset_name = args.filtered_dataset_name or filter_small_bboxes.build_output_dataset_name(
            args.converted_dataset_name,
            args.margin,
        )
        filter_summary = filter_small_bboxes.filter_dataset(
            src_dataset_path=yolo_base_path / args.converted_dataset_name,
            dst_dataset_path=yolo_base_path / filtered_dataset_name,
            margin=args.margin,
            min_inside_ratio=args.min_inside_ratio,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
        )
        update_step(
            manifest,
            "border_filtering",
            "completed",
            {
                "output_dataset_path": str(yolo_base_path / filtered_dataset_name),
                "summary_path": str(yolo_base_path / filtered_dataset_name / "filter_report.json"),
                "summary": filter_summary["summary"]["total"],
            },
        )
        write_json(manifest_path, manifest)

        current_dataset_name = filtered_dataset_name
        if args.preprocess_mode != "none":
            preprocessed_dataset_name = args.preprocessed_dataset_name or preprocess_dataset.build_output_dataset_name(
                filtered_dataset_name,
                args.preprocess_mode,
            )
            preprocess_summary = preprocess_dataset.preprocess_dataset(
                base_dataset_path=yolo_base_path / filtered_dataset_name,
                output_dataset_path=yolo_base_path / preprocessed_dataset_name,
                mode=args.preprocess_mode,
                overwrite=args.overwrite,
                max_workers=args.max_workers,
                fail_on_issues=args.fail_on_issues,
            )
            current_dataset_name = preprocessed_dataset_name
            update_step(
                manifest,
                "preprocessing",
                "completed",
                {
                    "mode": args.preprocess_mode,
                    "output_dataset_path": str(yolo_base_path / preprocessed_dataset_name),
                    "summary_path": str(yolo_base_path / preprocessed_dataset_name / "preprocess_summary.json"),
                    "counts": preprocess_summary["counts"],
                },
            )
            write_json(manifest_path, manifest)
        else:
            update_step(manifest, "preprocessing", "skipped", {"reason": "preprocess_mode=none"})
            write_json(manifest_path, manifest)

        if args.merge_base_dataset:
            if not args.merge_result_dataset:
                raise ValueError("--merge-result-dataset is required when --merge-base-dataset is provided")
            merge_summary = merge_yolo_datasets.merge_yolo_datasets(
                base_dataset_path=yolo_base_path / args.merge_base_dataset,
                update_dataset_path=yolo_base_path / current_dataset_name,
                result_dataset_path=yolo_base_path / args.merge_result_dataset,
                overwrite=args.overwrite,
            )
            current_dataset_name = args.merge_result_dataset
            update_step(
                manifest,
                "dataset_merge",
                "completed",
                {
                    "output_dataset_path": str(yolo_base_path / args.merge_result_dataset),
                    "summary_path": str(yolo_base_path / args.merge_result_dataset / "merge_summary.json"),
                    "summary": merge_summary["counts"],
                },
            )
            write_json(manifest_path, manifest)
        else:
            update_step(manifest, "dataset_merge", "skipped", {"reason": "merge_base_dataset not provided"})
            write_json(manifest_path, manifest)

        manifest["final_outputs"] = {
            "subsets_json": split_stage["subsets_path"],
            "overlap_json": split_stage["overlap_path"],
            "split_summary_json": split_stage["summary_path"],
            "converted_dataset_path": str(yolo_base_path / args.converted_dataset_name),
            "filtered_dataset_path": str(yolo_base_path / filtered_dataset_name),
            "final_dataset_path": str(yolo_base_path / current_dataset_name),
        }
        manifest["status"] = "completed"
        write_json(manifest_path, manifest)

        print(f"Batch injection pipeline completed. Final dataset: {yolo_base_path / current_dataset_name}")
        print(f"Manifest: {manifest_path}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
