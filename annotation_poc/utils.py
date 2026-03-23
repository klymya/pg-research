import os
import random
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


def normalize_annotation_id(raw_id: str | None) -> str | None:
    """Convert CVAT numeric IDs to a stable string representation."""
    if raw_id is None:
        return None

    value = raw_id.strip()
    if not value:
        return None

    try:
        decimal_value = Decimal(value)
    except InvalidOperation:
        return value

    if decimal_value == decimal_value.to_integral():
        return str(int(decimal_value))

    normalized = format(decimal_value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def get_attribute_value(element: ET.Element, attribute_name: str) -> str | None:
    """Return the text of a CVAT attribute by name."""
    attribute = element.find(f"attribute[@name='{attribute_name}']")
    if attribute is None or attribute.text is None:
        return None
    return attribute.text.strip()


def parse_cvat_annotations(xml_file: str | Path) -> dict[str, dict[str, Any]]:
    """Parse CVAT XML into a per-image structure containing raw boxes and points."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations: dict[str, dict[str, Any]] = {}

    for image in root.findall("image"):
        image_name = image.get("name")
        if image_name is None:
            continue

        width = int(image.get("width", 0))
        height = int(image.get("height", 0))
        boxes: dict[str, list[ET.Element]] = defaultdict(list)
        points: dict[str, list[ET.Element]] = defaultdict(list)

        for box in image.findall("box"):
            obj_id = normalize_annotation_id(get_attribute_value(box, "ID"))
            if obj_id is not None:
                boxes[obj_id].append(box)

        for point in image.findall("points"):
            obj_id = normalize_annotation_id(get_attribute_value(point, "ID"))
            if obj_id is not None:
                points[obj_id].append(point)

        annotations[image_name] = {
            "image_id": image.get("id"),
            "width": width,
            "height": height,
            "boxes": dict(boxes),
            "points": dict(points),
        }

    return annotations


# Function to parse the XML file and merge bounding boxes and points with the same ID
def parse_and_merge_annotations(xml_file):
    parsed_annotations = parse_cvat_annotations(xml_file)

    annotations = {}

    for image_name, image_data in parsed_annotations.items():
        annotations[image_name] = {
            "width": image_data["width"],
            "height": image_data["height"],
            "objects": [],
        }

        boxes = image_data["boxes"]
        points = image_data["points"]

        for obj_id in boxes:
            if obj_id in points:
                for box in boxes[obj_id]:
                    for point in points[obj_id]:
                        annotations[image_name]["objects"].append({
                            "label": box.get("label"),
                            "bounding_box": [
                                float(box.get("xtl")),
                                float(box.get("ytl")),
                                float(box.get("xbr")),
                                float(box.get("ybr")),
                            ],
                            "points": [
                                (float(p.split(",")[0]), float(p.split(",")[1])) for p in point.get("points").split(";")
                            ],
                            "color": (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                        })

    return annotations

# Function to draw annotations on the image
def draw_annotations(image_path, annotations):
    radius = 4
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for obj in annotations['objects']:
        bounding_box = obj['bounding_box']
        points = obj['points']
        color = obj['color']

        draw.rectangle(bounding_box, outline=color, width=2)
        for p in points:
            draw.ellipse([p[0]-radius, p[1]-radius, p[0]+radius, p[1]+radius], fill=color)

    return image


def draw_test_sample(annotations: dict, root_img_pth: str, n=9, n_rows=3, n_cols=3):
    count = 0

    plt.figure(figsize=(n_cols * 5 + 2, n_rows * 4))
    for image_name, annotation in annotations.items():
        image_path = os.path.join(root_img_pth, image_name)  # Adjust this if images are in a different directory
        annotated_image = draw_annotations(image_path, annotation)

        plt.subplot(n_rows, n_cols, count+1)
        plt.imshow(annotated_image)
        plt.title(image_name, fontsize=7)
        plt.axis('off')
        count += 1
        if count == n:
            break
