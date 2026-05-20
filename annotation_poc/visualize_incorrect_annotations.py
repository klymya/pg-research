#!/usr/bin/env python3
"""
Render incorrect CVAT annotations from a validation report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation_poc.utils import parse_cvat_annotations

GREEN = (0, 180, 0)
RED = (220, 30, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize incorrect CVAT annotations using a validation report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s report.json annotations.xml /path/to/images /path/to/output
        """,
    )
    parser.add_argument("report_json", type=Path, help="Path to the validation report JSON file")
    parser.add_argument("xml_path", type=Path, help="Path to the CVAT XML annotations file")
    parser.add_argument("images_dir", type=Path, help="Root folder containing the images")
    parser.add_argument("output_dir", type=Path, help="Folder where rendered images will be saved")
    parser.add_argument(
        "--point-radius",
        type=int,
        default=5,
        help="Radius for drawn points (default: 5)",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="Line width for bounding boxes (default: 3)",
    )
    return parser.parse_args()


def load_report(report_path: Path) -> dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def draw_box(draw: ImageDraw.ImageDraw, box_element: Any, color: tuple[int, int, int], line_width: int) -> None:
    bbox = [
        float(box_element.get("xtl")),
        float(box_element.get("ytl")),
        float(box_element.get("xbr")),
        float(box_element.get("ybr")),
    ]
    draw.rectangle(bbox, outline=color, width=line_width)


def draw_points(draw: ImageDraw.ImageDraw, points_element: Any, color: tuple[int, int, int], radius: int) -> None:
    raw_points = points_element.get("points", "")
    for raw_point in raw_points.split(";"):
        if not raw_point:
            continue
        x_str, y_str = raw_point.split(",")
        x_coord = float(x_str)
        y_coord = float(y_str)
        draw.ellipse(
            [x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius],
            fill=color,
            outline=color,
        )


def render_image(
    image_path: Path,
    output_path: Path,
    image_data: dict[str, Any],
    invalid_ids: set[str],
    point_radius: int,
    line_width: int,
) -> None:
    with Image.open(image_path) as image:
        rendered = image.convert("RGB")

    draw = ImageDraw.Draw(rendered)
    all_ids = sorted(set(image_data["boxes"]) | set(image_data["points"]))

    for annotation_id in all_ids:
        color = RED if annotation_id in invalid_ids else GREEN

        for box in image_data["boxes"].get(annotation_id, []):
            draw_box(draw, box, color, line_width)

        for points in image_data["points"].get(annotation_id, []):
            draw_points(draw, points, color, point_radius)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


def main() -> int:
    args = parse_args()
    report_path = args.report_json.expanduser().resolve()
    xml_path = args.xml_path.expanduser().resolve()
    images_dir = args.images_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not report_path.is_file():
        print(f"Error: report JSON file does not exist: {report_path}", file=sys.stderr)
        return 2

    if not xml_path.is_file():
        print(f"Error: XML file does not exist: {xml_path}", file=sys.stderr)
        return 2

    if not images_dir.is_dir():
        print(f"Error: images directory does not exist: {images_dir}", file=sys.stderr)
        return 2

    report = load_report(report_path)
    incorrect_images = report.get("incorrect_images", [])
    incorrect_annotations = report.get("incorrect_annotations", [])

    if not incorrect_images:
        print("No incorrect images found in the report.")
        return 0

    annotations = parse_cvat_annotations(xml_path)
    invalid_ids_by_image: dict[str, set[str]] = {}
    for issue in incorrect_annotations:
        image_name = issue["image"]
        invalid_ids_by_image.setdefault(image_name, set()).add(str(issue["id"]))

    missing_images: list[str] = []
    rendered_count = 0

    for image_name in incorrect_images:
        image_data = annotations.get(image_name)
        if image_data is None:
            print(f"Warning: image not found in XML: {image_name}", file=sys.stderr)
            continue

        # Strip any directory prefix from image_name to get just the filename
        image_path = images_dir / Path(image_name).name
        if not image_path.is_file():
            missing_images.append(image_name)
            continue

        output_path = output_dir / Path(image_name).name.replace(".tif", ".jpg")
        render_image(
            image_path=image_path,
            output_path=output_path,
            image_data=image_data,
            invalid_ids=invalid_ids_by_image.get(image_name, set()),
            point_radius=args.point_radius,
            line_width=args.line_width,
        )
        rendered_count += 1

    if missing_images:
        print(f"Rendered {rendered_count} image(s). Missing {len(missing_images)} image file(s).", file=sys.stderr)
        return 1

    print(f"Rendered {rendered_count} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
