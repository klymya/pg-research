#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT_PATH = Path("/home/klymy/data/cam4_revB")
ORIGINAL_SUBSETS_JSON = Path("/home/klymy/data/cam4_revB/train_val_holdout_subsets_batches_1_3.json")
SUBSETS_V2_DIR = Path("/home/klymy/data/cam4_revB/subsets_v2")
YOLO_DATASET_V2_DIR = Path("/home/klymy/data/cam4_revB/yolo_dataset_v2")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def build_sample_id_from_source_path(path: str) -> str:
    relative_path = Path(path).resolve(strict=False).relative_to(ROOT_PATH)
    return relative_path.with_suffix("").as_posix().replace("/", "____")


def main() -> None:
    original_data = load_json(ORIGINAL_SUBSETS_JSON)
    original_holdout = set(original_data.get("holdout", []))
    original_holdout_sample_ids = {build_sample_id_from_source_path(path) for path in original_holdout}

    print("subsets_v2 json holdout equality:")

    for json_path in sorted(SUBSETS_V2_DIR.glob("*.json")):
        data = load_json(json_path)
        generated_holdout = data.get("holdout")
        is_match = isinstance(generated_holdout, list) and set(generated_holdout) == original_holdout
        print(f"{json_path.name}: {'true' if is_match else 'false'}")

    print()
    print("yolo_dataset_v2 labels/holdout equality:")

    for dataset_path in sorted(path for path in YOLO_DATASET_V2_DIR.iterdir() if path.is_dir()):
        holdout_labels_dir = dataset_path / "labels" / "holdout"
        if not holdout_labels_dir.exists():
            continue
        generated_holdout_sample_ids = {path.stem for path in holdout_labels_dir.glob("*.txt")}
        is_match = generated_holdout_sample_ids == original_holdout_sample_ids
        print(f"{dataset_path.name}: {'true' if is_match else 'false'}")


if __name__ == "__main__":
    main()
