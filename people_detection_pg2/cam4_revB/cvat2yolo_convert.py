#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_ROOT_PTH = Path("/home/klymy/data/cam4_revB")
DEFAULT_OUTPUT_IMAGE_EXT = "jpg"
DEFAULT_ALLOWED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpeg", ".jpg")
DEFAULT_MAX_WORKERS = 8


class ConversionValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SourceImage:
    abs_path: str
    relative_key: str
    sample_id: str


@dataclass(frozen=True)
class AnnotationObject:
    xml_path: str
    image_name: str
    object_id: str
    width: int
    height: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    x: float
    y: float


@dataclass(frozen=True)
class InvalidObjectRecord:
    xml_path: str
    image_name: str
    object_id: str | None
    reason: str
    details: str


@dataclass(frozen=True)
class ImageIssueRecord:
    split: str | None
    abs_path: str | None
    relative_key: str
    reason: str
    details: str


@dataclass(frozen=True)
class ConvertedObject:
    cls: int
    x_center: float
    y_center: float
    width: float
    height: float
    x: float
    y: float


@dataclass(frozen=True)
class ConvertedSample:
    split: str
    source_image: SourceImage
    objects: list[ConvertedObject]
    dropped_object_count: int


def normalize_annotation_id(raw_id: str | None) -> str | None:
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
    attribute = element.find(f"attribute[@name='{attribute_name}']")
    if attribute is None or attribute.text is None:
        return None
    return attribute.text.strip()


def write_project_file(
    file_path: Path,
    dataset_path: Path,
    label_mapping: dict[int, str],
    val_folder: str = "images/val",
) -> None:
    label_lines = "\n".join(f"  {class_id}: {label_name}" for class_id, label_name in label_mapping.items())
    file_path.write_text(
        f"path: {dataset_path}\n"
        "train: images/train\n"
        f"val: {val_folder}\n"
        "kpt_shape:\n"
        "- 1\n"
        "- 2\n"
        "names:\n"
        f"{label_lines}\n"
    )


def create_project_files(dataset_path: Path, label_mapping: dict[int, str], include_holdout: bool) -> None:
    write_project_file(dataset_path / "project.yaml", dataset_path, label_mapping, val_folder="images/val")
    if include_holdout:
        write_project_file(
            dataset_path / "project_holdout.yaml",
            dataset_path,
            label_mapping,
            val_folder="images/holdout",
        )


def canonicalize_path(path: str | Path, base_path: Path | None = None) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base_path is not None:
        candidate = base_path / candidate
    return str(candidate.resolve(strict=False))


def relative_key_from_path(root_path: Path, image_path: Path) -> str:
    try:
        return image_path.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ConversionValidationError(
            f"Image path {image_path} is not under root path {root_path}"
        ) from exc


def build_sample_id(relative_key: str) -> str:
    relative_path = Path(relative_key)
    return relative_path.with_suffix("").as_posix().replace("/", "____")


def parse_point_text(points_text: str) -> tuple[float, float]:
    points = [item.strip() for item in points_text.split(";") if item.strip()]
    if len(points) != 1:
        raise ConversionValidationError(f"Expected exactly one keypoint, got {len(points)}")

    x_str, y_str = [item.strip() for item in points[0].split(",", 1)]
    return float(x_str), float(y_str)


def _is_finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def parse_annotation_xml(xml_path: Path) -> tuple[dict[str, list[AnnotationObject]], list[InvalidObjectRecord]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects_by_image: dict[str, list[AnnotationObject]] = defaultdict(list)
    invalid_records: list[InvalidObjectRecord] = []

    for image in root.findall("image"):
        image_name = image.get("name")
        if image_name is None:
            invalid_records.append(
                InvalidObjectRecord(
                    xml_path=str(xml_path),
                    image_name="",
                    object_id=None,
                    reason="missing_image_name",
                    details="Image element is missing the name attribute",
                )
            )
            continue

        objects_by_image[image_name]

        try:
            width = int(image.get("width", "0"))
            height = int(image.get("height", "0"))
        except ValueError:
            width = 0
            height = 0

        boxes_by_id: dict[str, list[ET.Element]] = defaultdict(list)
        points_by_id: dict[str, list[ET.Element]] = defaultdict(list)

        for box in image.findall("box"):
            object_id = normalize_annotation_id(get_attribute_value(box, "ID"))
            if object_id is None:
                invalid_records.append(
                    InvalidObjectRecord(
                        xml_path=str(xml_path),
                        image_name=image_name,
                        object_id=None,
                        reason="missing_box_id",
                        details="Box annotation is missing attribute ID",
                    )
                )
                continue
            boxes_by_id[object_id].append(box)

        for points in image.findall("points"):
            object_id = normalize_annotation_id(get_attribute_value(points, "ID"))
            if object_id is None:
                invalid_records.append(
                    InvalidObjectRecord(
                        xml_path=str(xml_path),
                        image_name=image_name,
                        object_id=None,
                        reason="missing_points_id",
                        details="Points annotation is missing attribute ID",
                    )
                )
                continue
            points_by_id[object_id].append(points)

        for object_id in sorted(set(boxes_by_id) | set(points_by_id)):
            boxes = boxes_by_id.get(object_id, [])
            points = points_by_id.get(object_id, [])
            if len(boxes) != 1 or len(points) != 1:
                invalid_records.append(
                    InvalidObjectRecord(
                        xml_path=str(xml_path),
                        image_name=image_name,
                        object_id=object_id,
                        reason="ambiguous_object_id",
                        details=f"Expected 1 box and 1 points entry, got {len(boxes)} box(es) and {len(points)} points entry(ies)",
                    )
                )
                continue

            try:
                xtl = float(boxes[0].get("xtl"))
                ytl = float(boxes[0].get("ytl"))
                xbr = float(boxes[0].get("xbr"))
                ybr = float(boxes[0].get("ybr"))
                x, y = parse_point_text(points[0].get("points", ""))
            except (TypeError, ValueError, ConversionValidationError) as exc:
                invalid_records.append(
                    InvalidObjectRecord(
                        xml_path=str(xml_path),
                        image_name=image_name,
                        object_id=object_id,
                        reason="parse_error",
                        details=str(exc),
                    )
                )
                continue

            objects_by_image[image_name].append(
                AnnotationObject(
                    xml_path=str(xml_path),
                    image_name=image_name,
                    object_id=object_id,
                    width=width,
                    height=height,
                    xtl=xtl,
                    ytl=ytl,
                    xbr=xbr,
                    ybr=ybr,
                    x=x,
                    y=y,
                )
            )

    return dict(objects_by_image), invalid_records


def load_annotations(xml_paths: list[Path]) -> tuple[dict[str, list[AnnotationObject]], list[InvalidObjectRecord]]:
    merged: dict[str, list[AnnotationObject]] = {}
    invalid_records: list[InvalidObjectRecord] = []

    for xml_path in xml_paths:
        objects_by_image, xml_invalid = parse_annotation_xml(xml_path)
        invalid_records.extend(xml_invalid)
        for image_name, objects in objects_by_image.items():
            if image_name in merged:
                raise ConversionValidationError(
                    f"Duplicate annotation image name {image_name!r} found across XML files"
                )
            merged[image_name] = objects

    return merged, invalid_records


def discover_images(root_path: Path, image_roots: list[Path], allowed_extensions: tuple[str, ...]) -> dict[str, SourceImage]:
    discovered: dict[str, SourceImage] = {}
    normalized_extensions = {extension.lower() for extension in allowed_extensions}
    normalized_root_path = root_path.resolve(strict=False)

    for image_root in image_roots:
        image_root = image_root.resolve(strict=False)
        if not image_root.exists():
            raise FileNotFoundError(f"Image root does not exist: {image_root}")

        for path in sorted(image_root.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in normalized_extensions:
                continue

            path = path.resolve(strict=False)
            abs_path = str(path)
            relative_key = relative_key_from_path(normalized_root_path, path)
            sample_id = build_sample_id(relative_key)
            if abs_path in discovered:
                raise ConversionValidationError(f"Duplicate image path discovered: {abs_path}")

            discovered[abs_path] = SourceImage(
                abs_path=abs_path,
                relative_key=relative_key,
                sample_id=sample_id,
            )

    return discovered


def load_subsets_json(subsets_json_path: Path, root_path: Path) -> dict[str, list[str]]:
    data = json.loads(subsets_json_path.read_text())
    expected_keys = {"train", "val", "holdout"}
    if set(data) != expected_keys:
        raise ConversionValidationError(
            f"Expected subsets JSON keys {sorted(expected_keys)}, got {sorted(data)}"
        )
    return {
        split: [canonicalize_path(path, base_path=root_path) for path in paths]
        for split, paths in data.items()
    }


def validate_subset_uniqueness(subsets: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    for split, paths in subsets.items():
        for path in paths:
            previous_split = seen.get(path)
            if previous_split is not None and previous_split != split:
                duplicates.append((path, previous_split, split))
            else:
                seen[path] = split
    if duplicates:
        sample = duplicates[:5]
        raise ConversionValidationError(f"Duplicate paths across subsets detected: {sample}")


def normalize_object(annotation_object: AnnotationObject) -> ConvertedObject | InvalidObjectRecord:
    if annotation_object.width <= 0 or annotation_object.height <= 0:
        return InvalidObjectRecord(
            xml_path=annotation_object.xml_path,
            image_name=annotation_object.image_name,
            object_id=annotation_object.object_id,
            reason="invalid_image_size",
            details=f"width={annotation_object.width}, height={annotation_object.height}",
        )

    raw_values = (
        annotation_object.xtl,
        annotation_object.ytl,
        annotation_object.xbr,
        annotation_object.ybr,
        annotation_object.x,
        annotation_object.y,
    )
    if not _is_finite(*raw_values):
        return InvalidObjectRecord(
            xml_path=annotation_object.xml_path,
            image_name=annotation_object.image_name,
            object_id=annotation_object.object_id,
            reason="non_finite_coordinates",
            details=str(raw_values),
        )

    if annotation_object.xbr <= annotation_object.xtl or annotation_object.ybr <= annotation_object.ytl:
        return InvalidObjectRecord(
            xml_path=annotation_object.xml_path,
            image_name=annotation_object.image_name,
            object_id=annotation_object.object_id,
            reason="invalid_bbox_geometry",
            details=(
                f"xtl={annotation_object.xtl}, ytl={annotation_object.ytl}, "
                f"xbr={annotation_object.xbr}, ybr={annotation_object.ybr}"
            ),
        )

    width = float(annotation_object.width)
    height = float(annotation_object.height)
    xtl = annotation_object.xtl / width
    ytl = annotation_object.ytl / height
    xbr = annotation_object.xbr / width
    ybr = annotation_object.ybr / height
    x = annotation_object.x / width
    y = annotation_object.y / height

    normalized_values = (xtl, ytl, xbr, ybr, x, y)
    if not _is_finite(*normalized_values):
        return InvalidObjectRecord(
            xml_path=annotation_object.xml_path,
            image_name=annotation_object.image_name,
            object_id=annotation_object.object_id,
            reason="non_finite_normalized_coordinates",
            details=str(normalized_values),
        )

    if not all(0.0 <= value <= 1.0 for value in normalized_values):
        return InvalidObjectRecord(
            xml_path=annotation_object.xml_path,
            image_name=annotation_object.image_name,
            object_id=annotation_object.object_id,
            reason="normalized_coordinates_out_of_bounds",
            details=str(normalized_values),
        )

    box_width = xbr - xtl
    box_height = ybr - ytl
    x_center = xtl + box_width / 2
    y_center = ytl + box_height / 2

    if box_width <= 0.0 or box_height <= 0.0:
        return InvalidObjectRecord(
            xml_path=annotation_object.xml_path,
            image_name=annotation_object.image_name,
            object_id=annotation_object.object_id,
            reason="invalid_normalized_bbox_geometry",
            details=(
                f"x_center={x_center}, y_center={y_center}, "
                f"width={box_width}, height={box_height}"
            ),
        )

    return ConvertedObject(
        cls=0,
        x_center=x_center,
        y_center=y_center,
        width=box_width,
        height=box_height,
        x=x,
        y=y,
    )


def prepare_converted_samples(
    root_path: Path,
    subsets: dict[str, list[str]],
    discovered_images: dict[str, SourceImage],
    annotations_by_image: dict[str, list[AnnotationObject]],
) -> tuple[list[ConvertedSample], list[ImageIssueRecord], list[InvalidObjectRecord], list[ImageIssueRecord]]:
    converted_samples: list[ConvertedSample] = []
    image_issues: list[ImageIssueRecord] = []
    invalid_objects: list[InvalidObjectRecord] = []

    split_entries = {
        path: split
        for split, paths in subsets.items()
        for path in paths
    }

    for source_image in discovered_images.values():
        if source_image.abs_path not in split_entries:
            image_issues.append(
                ImageIssueRecord(
                    split=None,
                    abs_path=source_image.abs_path,
                    relative_key=source_image.relative_key,
                    reason="source_image_not_in_split",
                    details="Source image was discovered on disk but is not present in subsets JSON",
                )
            )

    for split, paths in subsets.items():
        for path in paths:
            source_image = discovered_images.get(path)
            if source_image is None:
                image_issues.append(
                    ImageIssueRecord(
                        split=split,
                        abs_path=path,
                        relative_key="",
                        reason="missing_source_image",
                        details="Path is present in subsets JSON but was not found on disk",
                    )
                )
                continue

            annotation_objects = annotations_by_image.get(source_image.relative_key)
            if annotation_objects is None:
                image_issues.append(
                    ImageIssueRecord(
                        split=split,
                        abs_path=source_image.abs_path,
                        relative_key=source_image.relative_key,
                        reason="missing_annotation",
                        details="Source image exists in split JSON but has no annotation entry",
                    )
                )
                continue

            converted_objects: list[ConvertedObject] = []
            for annotation_object in annotation_objects:
                normalized = normalize_object(annotation_object)
                if isinstance(normalized, InvalidObjectRecord):
                    invalid_objects.append(normalized)
                else:
                    converted_objects.append(normalized)

            converted_samples.append(
                ConvertedSample(
                    split=split,
                    source_image=source_image,
                    objects=converted_objects,
                    dropped_object_count=len(annotation_objects) - len(converted_objects),
                )
            )

    annotation_images_without_source: list[ImageIssueRecord] = []
    for image_name in sorted(annotations_by_image):
        abs_path = canonicalize_path(root_path / image_name)
        if abs_path in discovered_images:
            continue
        annotation_images_without_source.append(
            ImageIssueRecord(
                split=None,
                abs_path=None,
                relative_key=image_name,
                reason="annotation_image_missing_source",
                details="Annotation exists but the source image was not discovered on disk",
            )
        )

    return converted_samples, image_issues, invalid_objects, annotation_images_without_source


def ensure_output_dataset_path(output_dataset_path: Path, overwrite: bool) -> None:
    if output_dataset_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_dataset_path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dataset_path)

    (output_dataset_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dataset_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dataset_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dataset_path / "labels" / "val").mkdir(parents=True, exist_ok=True)


def _write_one_sample(args: tuple[ConvertedSample, Path, str]) -> dict[str, Any]:
    sample, output_dataset_path, output_image_ext = args
    split_dir_images = output_dataset_path / "images" / sample.split
    split_dir_labels = output_dataset_path / "labels" / sample.split
    split_dir_images.mkdir(parents=True, exist_ok=True)
    split_dir_labels.mkdir(parents=True, exist_ok=True)

    label_path = split_dir_labels / f"{sample.source_image.sample_id}.txt"
    with label_path.open("wt") as handle:
        for converted_object in sample.objects:
            handle.write(
                f"{converted_object.cls} {converted_object.x_center} {converted_object.y_center} "
                f"{converted_object.width} {converted_object.height} {converted_object.x} {converted_object.y}\n"
            )

    image_path = split_dir_images / f"{sample.source_image.sample_id}.{output_image_ext}"
    with Image.open(sample.source_image.abs_path) as image:
        save_kwargs: dict[str, Any] = {}
        if output_image_ext.lower() in {"jpg", "jpeg"}:
            save_kwargs["quality"] = 95
        image.convert("RGB").save(image_path, **save_kwargs)

    return {
        "split": sample.split,
        "sample_id": sample.source_image.sample_id,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "object_count": len(sample.objects),
        "empty_label": len(sample.objects) == 0,
    }


def write_samples(
    samples: list[ConvertedSample],
    output_dataset_path: Path,
    output_image_ext: str,
    max_workers: int,
) -> list[dict[str, Any]]:
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    tasks = [(sample, output_dataset_path, output_image_ext) for sample in samples]
    if max_workers == 1:
        return [_write_one_sample(task) for task in tasks]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_write_one_sample, tasks))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(
    root_path: Path,
    annotation_xmls: list[Path],
    image_roots: list[Path],
    subsets_json_path: Path,
    output_dataset_path: Path,
    output_image_ext: str,
    discovered_images: dict[str, SourceImage],
    annotations_by_image: dict[str, list[AnnotationObject]],
    subsets: dict[str, list[str]],
    converted_samples: list[ConvertedSample],
    written_outputs: list[dict[str, Any]],
    image_issues: list[ImageIssueRecord],
    annotation_image_issues: list[ImageIssueRecord],
    invalid_objects: list[InvalidObjectRecord],
    parser_invalid_records: list[InvalidObjectRecord],
) -> dict[str, Any]:
    per_split_summary: dict[str, dict[str, Any]] = {}
    outputs_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in written_outputs:
        outputs_by_split[item["split"]].append(item)

    image_issues_by_split: dict[str, list[ImageIssueRecord]] = defaultdict(list)
    for item in image_issues:
        image_issues_by_split[item.split or "unknown"].append(item)

    samples_by_split: dict[str, list[ConvertedSample]] = defaultdict(list)
    for sample in converted_samples:
        samples_by_split[sample.split].append(sample)

    for split in ("train", "val", "holdout"):
        outputs = outputs_by_split.get(split, [])
        samples = samples_by_split.get(split, [])
        per_split_summary[split] = {
            "requested_images": len(subsets.get(split, [])),
            "converted_images": len(outputs),
            "objects_written": sum(item["object_count"] for item in outputs),
            "empty_label_files": sum(1 for item in outputs if item["empty_label"]),
            "missing_or_skipped_images": len(image_issues_by_split.get(split, [])),
            "samples_with_dropped_objects": sum(1 for sample in samples if sample.dropped_object_count > 0),
        }

    invalid_reason_counts = Counter(record.reason for record in invalid_objects + parser_invalid_records)
    image_issue_reason_counts = Counter(record.reason for record in image_issues + annotation_image_issues)

    return {
        "config": {
            "root_path": str(root_path),
            "annotation_xmls": [str(path) for path in annotation_xmls],
            "image_roots": [str(path) for path in image_roots],
            "subsets_json": str(subsets_json_path),
            "output_dataset_path": str(output_dataset_path),
            "label_mapping": {0: "person"},
            "output_image_ext": output_image_ext,
        },
        "counts": {
            "discovered_source_images": len(discovered_images),
            "split_entries_total": sum(len(paths) for paths in subsets.values()),
            "annotation_images_total": len(annotations_by_image),
            "converted_images_total": len(written_outputs),
            "objects_written_total": sum(item["object_count"] for item in written_outputs),
            "invalid_object_total": len(invalid_objects) + len(parser_invalid_records),
            "image_issue_total": len(image_issues) + len(annotation_image_issues),
        },
        "per_split": per_split_summary,
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "image_issue_reason_counts": dict(sorted(image_issue_reason_counts.items())),
        "samples": {
            "image_issues": [asdict(item) for item in image_issues[:100]],
            "annotation_image_issues": [asdict(item) for item in annotation_image_issues[:100]],
            "invalid_objects": [asdict(item) for item in (parser_invalid_records + invalid_objects)[:200]],
        },
    }


def run_conversion(
    root_path: Path,
    annotation_xmls: list[Path],
    image_roots: list[Path],
    subsets_json_path: Path,
    output_dataset_path: Path,
    overwrite: bool,
    output_image_ext: str,
    max_workers: int,
    fail_on_issues: bool = False,
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS,
) -> dict[str, Any]:
    root_path = root_path.resolve(strict=False)
    annotation_xmls = [path.resolve(strict=False) for path in annotation_xmls]
    image_roots = [path.resolve(strict=False) for path in image_roots]
    subsets_json_path = subsets_json_path.resolve(strict=False)
    output_dataset_path = output_dataset_path.resolve(strict=False)

    subsets = load_subsets_json(subsets_json_path, root_path=root_path)
    validate_subset_uniqueness(subsets)
    discovered_images = discover_images(root_path, image_roots, allowed_extensions)
    annotations_by_image, parser_invalid_records = load_annotations(annotation_xmls)
    converted_samples, image_issues, invalid_objects, annotation_image_issues = prepare_converted_samples(
        root_path=root_path,
        subsets=subsets,
        discovered_images=discovered_images,
        annotations_by_image=annotations_by_image,
    )

    issue_count = len(image_issues) + len(annotation_image_issues) + len(invalid_objects) + len(parser_invalid_records)
    if fail_on_issues and issue_count > 0:
        issue_counter = Counter(
            [record.reason for record in image_issues]
            + [record.reason for record in annotation_image_issues]
            + [record.reason for record in invalid_objects]
            + [record.reason for record in parser_invalid_records]
        )
        raise ConversionValidationError(
            "Conversion issues detected with --fail-on-issues: "
            + ", ".join(f"{reason}={count}" for reason, count in sorted(issue_counter.items()))
        )

    ensure_output_dataset_path(output_dataset_path, overwrite=overwrite)
    include_holdout = len(subsets.get("holdout", [])) > 0
    if include_holdout:
        (output_dataset_path / "images" / "holdout").mkdir(parents=True, exist_ok=True)
        (output_dataset_path / "labels" / "holdout").mkdir(parents=True, exist_ok=True)

    written_outputs = write_samples(
        samples=converted_samples,
        output_dataset_path=output_dataset_path,
        output_image_ext=output_image_ext,
        max_workers=max_workers,
    )
    create_project_files(output_dataset_path, {0: "person"}, include_holdout=include_holdout)

    summary = build_summary(
        root_path=root_path,
        annotation_xmls=annotation_xmls,
        image_roots=image_roots,
        subsets_json_path=subsets_json_path,
        output_dataset_path=output_dataset_path,
        output_image_ext=output_image_ext,
        discovered_images=discovered_images,
        annotations_by_image=annotations_by_image,
        subsets=subsets,
        converted_samples=converted_samples,
        written_outputs=written_outputs,
        image_issues=image_issues,
        annotation_image_issues=annotation_image_issues,
        invalid_objects=invalid_objects,
        parser_invalid_records=parser_invalid_records,
    )

    write_json(output_dataset_path / "conversion_summary.json", summary)
    write_csv(
        output_dataset_path / "image_issues.csv",
        [asdict(record) for record in image_issues + annotation_image_issues],
        fieldnames=["split", "abs_path", "relative_key", "reason", "details"],
    )
    write_csv(
        output_dataset_path / "invalid_annotations.csv",
        [asdict(record) for record in parser_invalid_records + invalid_objects],
        fieldnames=["xml_path", "image_name", "object_id", "reason", "details"],
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert corrected CVAT XML annotations into a YOLO pose dataset with explicit validation."
    )
    parser.add_argument(
        "--root-pth",
        default=str(DEFAULT_ROOT_PTH),
        help=f"Common root path for raw batches, corrected annotations, and yolo_dataset. Default: {DEFAULT_ROOT_PTH}",
    )
    parser.add_argument(
        "--annotation-xml",
        action="append",
        required=True,
        help="Path to a corrected CVAT XML file. Repeatable.",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        required=True,
        help="Path to a raw image folder. Repeatable.",
    )
    parser.add_argument(
        "--subsets-json",
        required=True,
        help="Path to train/val/holdout subsets JSON.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Output dataset name under <root_pth>/yolo_dataset.",
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
        help=f"Maximum worker threads for label/image writing. Default: {DEFAULT_MAX_WORKERS}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output dataset if it already exists.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Fail before writing output if any image issues or invalid annotations are detected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = Path(args.root_pth)
    annotation_xmls = [Path(path) for path in args.annotation_xml]
    image_roots = [Path(path) for path in args.image_root]
    subsets_json_path = Path(args.subsets_json)
    output_dataset_path = root_path / "yolo_dataset" / args.dataset_name

    summary = run_conversion(
        root_path=root_path,
        annotation_xmls=annotation_xmls,
        image_roots=image_roots,
        subsets_json_path=subsets_json_path,
        output_dataset_path=output_dataset_path,
        overwrite=args.overwrite,
        output_image_ext=args.output_image_ext.lstrip("."),
        max_workers=args.max_workers,
        fail_on_issues=args.fail_on_issues,
    )

    print(
        "Converted "
        f"{summary['counts']['converted_images_total']} images with "
        f"{summary['counts']['objects_written_total']} objects into {output_dataset_path}"
    )
    print(f"Summary: {output_dataset_path / 'conversion_summary.json'}")


if __name__ == "__main__":
    main()
