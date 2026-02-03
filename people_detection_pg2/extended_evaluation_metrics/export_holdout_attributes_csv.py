#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class YoloDet:
    cls: int
    x_c: float
    y_c: float
    w: float
    h: float
    x: float | None
    y: float | None


@dataclass(frozen=True)
class XmlGtDet:
    """
    GT detection parsed from CVAT XML (filtered to IDs that have both box and points).
    Coordinates are normalized to [0..1].
    """

    x_c: float
    y_c: float
    w: float
    h: float
    posture: str
    occluded_by_object: str
    occluded_by_person: str
    obj_id: str


def _float_or_none(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return float(s)


def parse_yolo_label_file(path: str) -> list[YoloDet]:
    dets: list[YoloDet] = []
    with open(path, "rt") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"{path}:{ln}: expected at least 5 columns, got {len(parts)}")
            cls = int(float(parts[0]))
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            x = _float_or_none(parts[5]) if len(parts) > 5 else None
            y = _float_or_none(parts[6]) if len(parts) > 6 else None
            dets.append(YoloDet(cls=cls, x_c=x_c, y_c=y_c, w=w, h=h, x=x, y=y))
    return dets


_HEIGHT_RE = re.compile(r"(?:^|_)H(?:_cam_)?(?P<h>\d+)(?:_|$)")


def parse_camera_height_from_yolo_id(yolo_id: str) -> int | None:
    """
    Examples:
      4.2_batch1____4.2_H260_me_2            -> 260
      4.2_batch1____4.2_H_cam_215_mev...     -> 215
      4.2_batch2____4.2_H_cam_272_spaces...  -> 272
    """
    m = _HEIGHT_RE.search(yolo_id)
    if not m:
        return None
    return int(m.group("h"))


def infer_image_name_from_yolo_id(yolo_id: str, ext: str = ".tif") -> str:
    # Reverse of img_pth2id(): replace "____" with "/" and append extension.
    return yolo_id.replace("____", "/") + ext


def position_center_or_edge(x_c: float, y_c: float, center_threshold: float) -> str:
    if (
        center_threshold <= x_c <= (1.0 - center_threshold)
        and center_threshold <= y_c <= (1.0 - center_threshold)
    ):
        return "center"
    return "edge"


def leg_distance_to_center(x: float | None, y: float | None) -> float | None:
    if x is None or y is None:
        return None
    dx = x - 0.5
    dy = y - 0.5
    return (dx * dx + dy * dy) ** 0.5


def position_center_or_edge_by_leg_distance(distance: float | None, center_distance_threshold: float) -> str:
    if distance is None:
        return ""
    return "center" if distance <= center_distance_threshold else "edge"


def density_class(num_gt_detections: int, sparse_max: int, medium_max: int) -> str:
    if num_gt_detections <= sparse_max:
        return "sparse"
    if num_gt_detections <= medium_max:
        return "medium"
    return "dense"


def _iter_xml_paths(xml_dirs_or_files: list[str]) -> list[str]:
    paths: list[str] = []
    for p in xml_dirs_or_files:
        if "*" in p or "?" in p or "[" in p:
            paths.extend(glob.glob(p))
            continue
        if os.path.isdir(p):
            paths.extend(glob.glob(os.path.join(p, "*.xml")))
            continue
        paths.append(p)
    return sorted(set(paths))


def _yolo_id_from_cvat_image_name(name: str) -> str:
    return str(Path(name).with_suffix("")).replace("/", "____")


def _normalize_bool_like(s: str) -> str:
    v = (s or "").strip().lower()
    if v in ("true", "1", "yes", "y"):
        return "true"
    if v in ("false", "0", "no", "n"):
        return "false"
    return v


def _posture_from_attrs(attrs: dict[str, str]) -> str:
    posture = (attrs.get("Posture") or "").strip()
    sitting = (attrs.get("Sitting") or "").strip()

    if posture == "Standing":
        return "Standing"
    if posture == "Lying":
        return "Lying"
    if posture == "Sitting":
        if sitting == "Regularly":
            return "Sitting Regularly"
        if sitting == "Leaning Forward":
            return "Leaning Forward"
        if sitting == "Leaning Backward":
            return "Leaning Backward"
        return "Sitting"
    return posture


def _parse_cvat_gt_dets_for_image(image_el: ET.Element) -> list[XmlGtDet]:
    w = image_el.get("width")
    h = image_el.get("height")
    if w is None or h is None:
        return []
    width = int(w)
    height = int(h)

    boxes_by_id: dict[str, list[ET.Element]] = {}
    points_by_id: dict[str, list[ET.Element]] = {}

    for box in image_el.findall("box"):
        if box.get("label") != "Person":
            continue
        id_el = box.find("attribute[@name='ID']")
        if id_el is None or id_el.text is None:
            continue
        obj_id = id_el.text.strip()
        boxes_by_id.setdefault(obj_id, []).append(box)

    for pts in image_el.findall("points"):
        id_el = pts.find("attribute[@name='ID']")
        if id_el is None or id_el.text is None:
            continue
        obj_id = id_el.text.strip()
        points_by_id.setdefault(obj_id, []).append(pts)

    dets: list[XmlGtDet] = []
    for obj_id in boxes_by_id.keys():  # preserve insertion order
        if obj_id not in points_by_id:
            continue
        for box in boxes_by_id[obj_id]:
            xtl = float(box.get("xtl")) / width
            ytl = float(box.get("ytl")) / height
            xbr = float(box.get("xbr")) / width
            ybr = float(box.get("ybr")) / height
            bw = xbr - xtl
            bh = ybr - ytl
            x_c = xtl + bw / 2
            y_c = ytl + bh / 2

            attrs = {a.get("name"): (a.text or "").strip() for a in box.findall("attribute") if a.get("name")}
            posture = _posture_from_attrs(attrs)
            occl_obj = _normalize_bool_like(attrs.get("Occluded by object", ""))
            occl_per = _normalize_bool_like(attrs.get("Occluded by person", ""))

            dets.append(
                XmlGtDet(
                    x_c=x_c,
                    y_c=y_c,
                    w=bw,
                    h=bh,
                    posture=posture,
                    occluded_by_object=occl_obj,
                    occluded_by_person=occl_per,
                    obj_id=obj_id,
                )
            )
    return dets


def parse_cvat_xmls(xml_paths: list[str]) -> dict[str, dict]:
    by_yolo_id: dict[str, dict] = {}
    for xp in xml_paths:
        root = ET.parse(xp).getroot()
        for image in root.findall("image"):
            name = image.get("name")
            if name is None:
                continue
            yolo_id = _yolo_id_from_cvat_image_name(name)
            if yolo_id in by_yolo_id:
                continue
            by_yolo_id[yolo_id] = {
                "xml_path": xp,
                "xml_image_name": name,
                "dets": _parse_cvat_gt_dets_for_image(image),
            }
    return by_yolo_id


def load_mismatch_mapping(report_path: str) -> dict[str, dict[str, list[int]]]:
    if not report_path:
        return {}
    p = Path(report_path)
    if not p.is_file():
        raise FileNotFoundError(f"Mismatch report not found: {report_path}")

    out: dict[str, dict[str, list[int]]] = {}
    with open(p, "rt") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{report_path}:{ln}: invalid JSON: {e}") from e
            if not isinstance(rec, dict):
                continue
            yolo_id = rec.get("yolo_id")
            mapping = rec.get("mapping")
            if not yolo_id or not isinstance(mapping, dict):
                continue
            y = mapping.get("yolo")
            x = mapping.get("xml")
            if not isinstance(y, list) or not isinstance(x, list) or len(y) != len(x):
                continue
            out[yolo_id] = {"yolo": [int(i) for i in y], "xml": [int(i) for i in x]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export frame-level and GT-detection-level attribute CSVs for YOLO holdout set labels.\n"
            "Attributes are based on: YOLOv8 Evaluation Report — Extended Design (Next Iteration)."
        )
    )
    ap.add_argument(
        "--yolo-glob",
        default="/home/klymy/data/cam4_revB/yolo_dataset/batch1_3_margin_15_awb_clahe/labels/holdout/*.txt",
        help="Glob for YOLO holdout label files (default: %(default)s)",
    )
    ap.add_argument(
        "--xml",
        nargs="+",
        default=[
            "/home/klymy/data/cam4_revB/batch1_corrected/*.xml",
            "/home/klymy/data/cam4_revB/batch2_corrected/*.xml",
            "/home/klymy/data/cam4_revB/batch3_corrected/*.xml",
        ],
        help="CVAT XML directories/files/globs (default: batch1-3 corrected globs)",
    )
    ap.add_argument(
        "--mismatch-report",
        default="people_detection_pg2/extended_evaluation_metrics/mismatch_report.json",
        help="Mismatch report JSONL with index mapping (default: %(default)s)",
    )
    ap.add_argument(
        "--out-frames-csv",
        default="frames_attributes.csv",
        help="Output CSV with frame-level attributes (default: %(default)s)",
    )
    ap.add_argument(
        "--out-dets-csv",
        default="detections_attributes.csv",
        help="Output CSV with detection-level attributes (default: %(default)s)",
    )
    ap.add_argument(
        "--center-threshold",
        type=float,
        default=0.25,
        help="Center/edge threshold for position (default: %(default)s)",
    )
    ap.add_argument(
        "--center-distance-threshold",
        type=float,
        default=0.25,
        help="Center/edge threshold applied to Euclidean distance from leg point (x,y) to image center (default: %(default)s)",
    )
    ap.add_argument(
        "--sparse-max",
        type=int,
        default=2,
        help="Max #GT detections for scene_density=sparse (default: %(default)s)",
    )
    ap.add_argument(
        "--medium-max",
        type=int,
        default=5,
        help="Max #GT detections for scene_density=medium (default: %(default)s)",
    )
    ap.add_argument(
        "--image-ext",
        default=".tif",
        help="Extension to use when inferring image_name from yolo_id (default: %(default)s)",
    )

    args = ap.parse_args()

    yolo_paths = sorted(glob.glob(args.yolo_glob))
    if not yolo_paths:
        raise SystemExit("No YOLO label files found (check --yolo-glob).")

    xml_paths = _iter_xml_paths(args.xml)
    if not xml_paths:
        raise SystemExit("No XML files found (check --xml).")
    gt_by_yolo_id = parse_cvat_xmls(xml_paths)
    mismatch_map = load_mismatch_mapping(args.mismatch_report) if args.mismatch_report else {}

    frames_rows: list[dict] = []
    det_rows: list[dict] = []

    for p in yolo_paths:
        yolo_id = Path(p).with_suffix("").name
        dets = parse_yolo_label_file(p)

        gt_rec = gt_by_yolo_id.get(yolo_id)
        xml_path = "" if gt_rec is None else gt_rec["xml_path"]
        xml_image_name = "" if gt_rec is None else gt_rec["xml_image_name"]
        gt_dets: list[XmlGtDet] = [] if gt_rec is None else gt_rec["dets"]

        camera_height = parse_camera_height_from_yolo_id(yolo_id)
        num_gt = len(gt_dets) if gt_rec is not None else len(dets)
        scene_density = density_class(num_gt, sparse_max=args.sparse_max, medium_max=args.medium_max)

        frames_rows.append(
            {
                # Frame-level attributes from the spec
                "yolo_id": yolo_id,
                "yolo_path": p,
                "xml_path": xml_path,
                "xml_image_name": xml_image_name or infer_image_name_from_yolo_id(yolo_id, ext=args.image_ext),
                "camera_height": "" if camera_height is None else camera_height,
                "num_gt_detections": num_gt,
                "scene_density": scene_density,
            }
        )

        y2x: dict[int, int] = {}
        mm = mismatch_map.get(yolo_id)
        if mm is not None:
            for yi, xi in zip(mm.get("yolo", []), mm.get("xml", [])):
                y2x[int(yi)] = int(xi)
        else:
            for i in range(min(len(dets), len(gt_dets))):
                y2x[i] = i

        for det_idx, d in enumerate(dets):
            xml_det_idx = y2x.get(det_idx)
            gt = gt_dets[xml_det_idx] if (xml_det_idx is not None and 0 <= xml_det_idx < len(gt_dets)) else None

            dist = leg_distance_to_center(d.x, d.y)
            if dist is None:
                # Fallback for missing leg points: use bbox center distance to image center.
                dist = leg_distance_to_center(d.x_c, d.y_c)
            det_rows.append(
                {
                    # Keys to join back to frames
                    "yolo_id": yolo_id,
                    "yolo_path": p,
                    "xml_path": xml_path,
                    "xml_image_name": xml_image_name,
                    "det_index": det_idx,
                    "xml_det_index": "" if xml_det_idx is None else xml_det_idx,
                    # Raw YOLO label columns
                    "cls": d.cls,
                    "x_c": d.x_c,
                    "y_c": d.y_c,
                    "w": d.w,
                    "h": d.h,
                    "x": "" if d.x is None else d.x,
                    "y": "" if d.y is None else d.y,
                    # GT detection attributes from the spec (some derived / some placeholders)
                    "leg_distance_to_center": "" if dist is None else dist,
                    "position": position_center_or_edge_by_leg_distance(
                        dist,
                        center_distance_threshold=args.center_distance_threshold,
                    )
                    or position_center_or_edge(d.x_c, d.y_c, center_threshold=args.center_threshold),
                    "posture": "" if gt is None else gt.posture,
                    "occluded_by_object": "" if gt is None else gt.occluded_by_object,
                    "occluded_by_person": "" if gt is None else gt.occluded_by_person,
                }
            )

    Path(args.out_frames_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_dets_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_frames_csv, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(frames_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frames_rows)

    # Write an empty file with header if there are 0 detections total (unlikely but possible).
    det_fieldnames = list(det_rows[0].keys()) if det_rows else [
        "yolo_id",
        "yolo_path",
        "xml_path",
        "xml_image_name",
        "det_index",
        "xml_det_index",
        "cls",
        "x_c",
        "y_c",
        "w",
        "h",
        "x",
        "y",
        "leg_distance_to_center",
        "position",
        "posture",
        "occluded_by_object",
        "occluded_by_person",
    ]
    with open(args.out_dets_csv, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=det_fieldnames)
        writer.writeheader()
        writer.writerows(det_rows)

    print(f"Frames CSV: {args.out_frames_csv} ({len(frames_rows)} rows)")
    print(f"Detections CSV: {args.out_dets_csv} ({len(det_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
