#!/usr/bin/env python3
"""
Validate CVAT XML annotations for one-to-one box/points pairing by ID.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation_poc.utils import parse_cvat_annotations


@dataclass(frozen=True)
class ValidationIssue:
    image: str
    annotation_id: str
    boxes_count: int
    points_count: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "id": self.annotation_id,
            "boxes_count": self.boxes_count,
            "points_count": self.points_count,
            "message": self.message,
        }


def build_output_path(xml_path: Path) -> Path:
    return xml_path.with_name(f"{xml_path.stem}_validation_report.json")


def validate_annotations(xml_path: Path) -> list[ValidationIssue]:
    parsed_annotations = parse_cvat_annotations(xml_path)
    issues: list[ValidationIssue] = []

    for image_name, image_data in parsed_annotations.items():
        ids = sorted(set(image_data["boxes"]) | set(image_data["points"]))

        for annotation_id in ids:
            boxes_count = len(image_data["boxes"].get(annotation_id, []))
            points_count = len(image_data["points"].get(annotation_id, []))

            if boxes_count == 1 and points_count == 1:
                continue

            if boxes_count == 0:
                message = "missing box for points annotation"
            elif points_count == 0:
                message = "missing points for box annotation"
            else:
                message = "expected exactly one box and one points annotation for the same ID"

            issues.append(
                ValidationIssue(
                    image=image_name,
                    annotation_id=annotation_id,
                    boxes_count=boxes_count,
                    points_count=points_count,
                    message=message,
                )
            )

    return issues


def write_report(xml_path: Path, output_path: Path, issues: list[ValidationIssue]) -> None:
    report = {
        "xml_file": str(xml_path),
        "is_valid": not issues,
        "incorrect_annotations": [issue.to_dict() for issue in issues],
        "incorrect_images": sorted({issue.image for issue in issues}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that each CVAT box has exactly one matching points annotation by ID.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/annotations.xml
  %(prog)s /path/to/annotations.xml --output /path/to/report.json
        """,
    )
    parser.add_argument(
        "xml_path",
        type=Path,
        help="Path to the CVAT XML annotations file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON validation report",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> int:
    args = parse_args()
    xml_path = args.xml_path.expanduser().resolve()
    output_path = (args.output or build_output_path(xml_path)).expanduser().resolve()

    if not xml_path.exists():
        print("not ok")
        print(f"Error: XML file does not exist: {xml_path}", file=sys.stderr)
        return 2

    if not xml_path.is_file():
        print("not ok")
        print(f"Error: Path is not a file: {xml_path}", file=sys.stderr)
        return 2

    try:
        issues = validate_annotations(xml_path)
    except (ET.ParseError, OSError) as exc:
        print("not ok")
        print(f"Error: failed to parse XML file: {exc}", file=sys.stderr)
        return 2

    write_report(xml_path, output_path, issues)

    if issues:
        print("not ok")
        return 1

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
