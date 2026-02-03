#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class YoloDet:
    cls: int
    x_c: float
    y_c: float
    w: float
    h: float
    x: float | None = None
    y: float | None = None


@dataclass(frozen=True)
class XmlDet:
    label: str
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    x: float | None
    y: float | None
    width: int
    height: int
    obj_id: str

    def to_yolo(self, cls: int = 0) -> YoloDet:
        xtl = self.xtl / self.width
        ytl = self.ytl / self.height
        xbr = self.xbr / self.width
        ybr = self.ybr / self.height
        w = xbr - xtl
        h = ybr - ytl
        x_c = xtl + w / 2
        y_c = ytl + h / 2
        x = None if self.x is None else self.x / self.width
        y = None if self.y is None else self.y / self.height
        return YoloDet(cls=cls, x_c=x_c, y_c=y_c, w=w, h=h, x=x, y=y)


def _float_or_none(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return float(s)


def parse_yolo_label_file(pth: str) -> list[YoloDet]:
    dets: list[YoloDet] = []
    with open(pth, "rt") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"{pth}:{ln}: expected at least 5 columns, got {len(parts)}")
            cls = int(float(parts[0]))
            x_c = float(parts[1])
            y_c = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            x = _float_or_none(parts[5]) if len(parts) > 5 else None
            y = _float_or_none(parts[6]) if len(parts) > 6 else None
            dets.append(YoloDet(cls=cls, x_c=x_c, y_c=y_c, w=w, h=h, x=x, y=y))
    return dets


def yolo_id_from_cvat_image_name(name: str, prefix: str = "") -> str:
    p = Path(name)
    stem = str(p.with_suffix(""))
    stem = stem.replace("/", "____")
    return f"{prefix}{stem}"


def _cvat_obj_id(el: ET.Element) -> str | None:
    attr = el.find("attribute[@name='ID']")
    if attr is None or attr.text is None:
        return None
    return attr.text.strip()


def parse_cvat_xml(
    xml_path: str,
    *,
    yolo_id_prefix: str = "",
    required_box_label: str | None = None,
) -> dict[str, dict]:
    """
    Parses CVAT 1.1-style XML:
      <annotations>
        <image id=".." name="..tif" width=".." height="..">
          <box label="Person" xtl=".." ytl=".." xbr=".." ybr="..">
            <attribute name="ID">1</attribute>
          </box>
          <points label="Dot" points="x,y;..." ...>
            <attribute name="ID">1</attribute>
          </points>
        </image>
      </annotations>

    Returns a mapping {yolo_id -> {"xml_path", "image_name", "width", "height", "dets": [XmlDet...]}}
    where ordering of dets follows the same "box ID insertion order" logic as in annotation_poc/utils.py.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    out: dict[str, dict] = {}
    for image in root.findall("image"):
        name = image.get("name")
        w = image.get("width")
        h = image.get("height")
        if name is None or w is None or h is None:
            continue

        width = int(w)
        height = int(h)
        yolo_id = yolo_id_from_cvat_image_name(name, prefix=yolo_id_prefix)

        boxes_by_id: dict[str, list[ET.Element]] = {}
        points_by_id: dict[str, list[ET.Element]] = {}

        for box in image.findall("box"):
            if required_box_label is not None and box.get("label") != required_box_label:
                continue
            obj_id = _cvat_obj_id(box)
            if obj_id is None:
                continue
            boxes_by_id.setdefault(obj_id, []).append(box)

        for pts in image.findall("points"):
            obj_id = _cvat_obj_id(pts)
            if obj_id is None:
                continue
            points_by_id.setdefault(obj_id, []).append(pts)

        dets: list[XmlDet] = []
        for obj_id in boxes_by_id.keys():  # insertion order preserved (Python 3.7+)
            if obj_id not in points_by_id:
                continue
            pts_el = points_by_id[obj_id][0]
            points = (pts_el.get("points") or "").strip()
            first_xy = points.split(";")[0].strip() if points else ""
            x = y = None
            if first_xy and "," in first_xy:
                x_s, y_s = first_xy.split(",", 1)
                x = _float_or_none(x_s)
                y = _float_or_none(y_s)

            for box_el in boxes_by_id[obj_id]:
                dets.append(
                    XmlDet(
                        label=(box_el.get("label") or ""),
                        xtl=float(box_el.get("xtl")),
                        ytl=float(box_el.get("ytl")),
                        xbr=float(box_el.get("xbr")),
                        ybr=float(box_el.get("ybr")),
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        obj_id=obj_id,
                    )
                )

        out[yolo_id] = {
            "xml_path": xml_path,
            "image_name": name,
            "width": width,
            "height": height,
            "dets": dets,
        }

    return out


def _abs_diff(a: float | None, b: float | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return abs(a - b)


def det_pair_errors(
    y: YoloDet,
    x: YoloDet,
    *,
    atol_bbox: float,
    atol_xy: float,
    require_cls: bool,
    compare_xy: bool,
) -> dict:
    cls_ok = (y.cls == x.cls) if require_cls else True
    bbox_errs = {
        "x_c": _abs_diff(y.x_c, x.x_c),
        "y_c": _abs_diff(y.y_c, x.y_c),
        "w": _abs_diff(y.w, x.w),
        "h": _abs_diff(y.h, x.h),
    }
    bbox_ok = all(v <= atol_bbox for v in bbox_errs.values())

    if compare_xy:
        # Treat x/y as optional: if either side lacks x/y, we don't fail comparison on it.
        xy_errs: dict[str, float | None] = {"x": None, "y": None}
        xy_ok = True
        if y.x is not None and x.x is not None:
            xy_errs["x"] = abs(y.x - x.x)
            xy_ok = xy_ok and (xy_errs["x"] <= atol_xy)
        if y.y is not None and x.y is not None:
            xy_errs["y"] = abs(y.y - x.y)
            xy_ok = xy_ok and (xy_errs["y"] <= atol_xy)
        max_xy = max([v for v in xy_errs.values() if isinstance(v, float) and math.isfinite(v)], default=0.0)
    else:
        xy_errs = {}
        xy_ok = True
        max_xy = None

    ok = cls_ok and bbox_ok and xy_ok
    return {
        "ok": ok,
        "cls_ok": cls_ok,
        "bbox_ok": bbox_ok,
        "xy_ok": xy_ok,
        "bbox_abs_err": bbox_errs,
        "xy_abs_err": xy_errs,
        "max_bbox_abs_err": max(bbox_errs.values(), default=0.0),
        "max_xy_abs_err": max_xy,
    }


def _pair_cost(
    y: YoloDet,
    x: YoloDet,
    *,
    require_cls: bool,
    compare_xy: bool,
    cls_mismatch_penalty: float,
    xy_weight: float,
) -> float:
    cls_penalty = 0.0
    if require_cls and y.cls != x.cls:
        cls_penalty = cls_mismatch_penalty

    bbox_cost = max(
        _abs_diff(y.x_c, x.x_c),
        _abs_diff(y.y_c, x.y_c),
        _abs_diff(y.w, x.w),
        _abs_diff(y.h, x.h),
    )
    if not compare_xy:
        return cls_penalty + bbox_cost

    # Treat x/y as optional: if a coordinate is missing in either det, ignore it in cost.
    dx = abs(y.x - x.x) if (y.x is not None and x.x is not None) else 0.0
    dy = abs(y.y - x.y) if (y.y is not None and x.y is not None) else 0.0
    xy_cost = max(dx, dy)
    return cls_penalty + max(bbox_cost, xy_weight * xy_cost)


def best_match_mapping_yolo_to_xml(
    yolo: list[YoloDet],
    xml_as_yolo: list[YoloDet],
    *,
    require_cls: bool,
    compare_xy: bool,
    allow_unmatched_yolo: bool,
    unmatched_penalty: float,
    max_exact: int = 15,
    cls_mismatch_penalty: float = 10.0,
    xy_weight: float = 1.0,
) -> tuple[list[int | None], list[int]]:
    """
    Returns:
      - mapping: list of length len(yolo), where mapping[i] is xml index matched to yolo i (or None)
      - unmatched_xml_indices: xml indices not used by any match
    """
    n = len(yolo)
    m = len(xml_as_yolo)
    if n == 0:
        return [], list(range(m))
    if m == 0:
        return [None] * n, []
    if not allow_unmatched_yolo and n > m:
        raise ValueError(f"Cannot fully match yolo->xml: yolo={n} > xml={m} and allow_unmatched_yolo=False")

    cost = lambda i, j: _pair_cost(
        yolo[i],
        xml_as_yolo[j],
        require_cls=require_cls,
        compare_xy=compare_xy,
        cls_mismatch_penalty=cls_mismatch_penalty,
        xy_weight=xy_weight,
    )

    # Exact: try all permutations if very small, else DP over xml mask.
    if max(n, m) <= max_exact and m <= 20:
        dp: dict[int, float] = {0: 0.0}
        back: list[dict[int, tuple[int, int]]] = []  # per i: mask -> (prev_mask, choice_j)

        for i in range(n):
            new_dp: dict[int, float] = {}
            new_back: dict[int, tuple[int, int]] = {}

            for mask, cur_cost in dp.items():
                if allow_unmatched_yolo:
                    c_un = cur_cost + unmatched_penalty
                    prev_cost = new_dp.get(mask)
                    if prev_cost is None or c_un < prev_cost:
                        new_dp[mask] = c_un
                        new_back[mask] = (mask, -1)

                for j in range(m):
                    bit = 1 << j
                    if mask & bit:
                        continue
                    nm = mask | bit
                    c = cur_cost + cost(i, j)
                    prev_cost = new_dp.get(nm)
                    if prev_cost is None or c < prev_cost:
                        new_dp[nm] = c
                        new_back[nm] = (mask, j)

            dp = new_dp
            back.append(new_back)

        def _popcount(x: int) -> int:
            return x.bit_count()

        best_mask = min(dp.keys(), key=lambda mask: (dp[mask], -_popcount(mask)))
        mapping: list[int | None] = [None] * n
        mask = best_mask
        for i in range(n - 1, -1, -1):
            prev_mask, choice_j = back[i][mask]
            if choice_j != -1:
                mapping[i] = choice_j
            mask = prev_mask

        used = {j for j in mapping if j is not None}
        unmatched_xml = [j for j in range(m) if j not in used]
        return mapping, unmatched_xml

    # Greedy fallback.
    remaining = set(range(m))
    mapping: list[int | None] = [None] * n
    for i in range(n):
        if not remaining:
            if allow_unmatched_yolo:
                break
            raise ValueError(f"Cannot fully match yolo->xml: ran out of xml dets at i={i}")
        best_j = min(remaining, key=lambda j: cost(i, j))
        best_c = cost(i, best_j)
        if not allow_unmatched_yolo or best_c <= unmatched_penalty:
            mapping[i] = best_j
            remaining.remove(best_j)
        else:
            mapping[i] = None
    return mapping, sorted(remaining)


def analyze_order_mismatch(
    yolo: list[YoloDet],
    xml: list[XmlDet],
    *,
    cls_from_xml: int,
    atol_bbox: float,
    atol_xy: float,
    require_cls: bool,
    compare_xy: bool,
    unmatched_penalty: float,
) -> dict:
    xml_as_yolo = [d.to_yolo(cls=cls_from_xml) for d in xml]

    direct_mismatch_indices: list[int] = []
    for i in range(min(len(yolo), len(xml_as_yolo))):
        if not det_pair_errors(
            yolo[i],
            xml_as_yolo[i],
            atol_bbox=atol_bbox,
            atol_xy=atol_xy,
            require_cls=require_cls,
            compare_xy=compare_xy,
        )["ok"]:
            direct_mismatch_indices.append(i)

    mapping, unmatched_xml = best_match_mapping_yolo_to_xml(
        yolo,
        xml_as_yolo,
        require_cls=require_cls,
        compare_xy=compare_xy,
        allow_unmatched_yolo=(len(yolo) != len(xml_as_yolo)),
        unmatched_penalty=unmatched_penalty,
    )
    unmatched_yolo = [i for i, j in enumerate(mapping) if j is None]

    matched_pairs = []
    reordered_pairs = []
    all_matched_ok = True
    for i, j in enumerate(mapping):
        if j is None:
            all_matched_ok = False
            continue
        pe = det_pair_errors(
            yolo[i],
            xml_as_yolo[j],
            atol_bbox=atol_bbox,
            atol_xy=atol_xy,
            require_cls=require_cls,
            compare_xy=compare_xy,
        )
        if not pe["ok"]:
            all_matched_ok = False
        matched_pairs.append(
            {
                "yolo_index": i,
                "xml_index": j,
                "ok": pe["ok"],
                "max_bbox_abs_err": pe["max_bbox_abs_err"],
                "max_xy_abs_err": pe["max_xy_abs_err"],
            }
        )
        if j != i:
            reordered_pairs.append(
                {
                    "yolo_index": i,
                    "xml_index": j,
                    "max_bbox_abs_err": pe["max_bbox_abs_err"],
                    "max_xy_abs_err": pe["max_xy_abs_err"],
                }
            )

    if len(yolo) != len(xml_as_yolo):
        mismatch_type = "count"
    elif all_matched_ok and any((j is not None and j != i) for i, j in enumerate(mapping)):
        mismatch_type = "reorder"
    elif all_matched_ok:
        mismatch_type = "unknown"
    else:
        mismatch_type = "content"

    return {
        "mismatch_type": mismatch_type,
        "direct_mismatch_indices": direct_mismatch_indices,
        "suggested_xml_index_for_yolo_index": mapping,
        "unmatched_yolo_indices": unmatched_yolo,
        "unmatched_xml_indices": unmatched_xml,
        "matched_pairs_summary": matched_pairs,
        "reordered_pairs_summary": reordered_pairs,
    }


def missing_xml_mismatch_analysis(n_yolo: int) -> dict:
    return {
        "mismatch_type": "missing_xml",
        "direct_mismatch_indices": [],
        "suggested_xml_index_for_yolo_index": [None] * n_yolo,
        "unmatched_yolo_indices": list(range(n_yolo)),
        "unmatched_xml_indices": [],
        "matched_pairs_summary": [],
        "reordered_pairs_summary": [],
    }


def compare_dets_in_order(
    yolo: list[YoloDet],
    xml: list[XmlDet],
    *,
    cls_from_xml: int = 0,
    atol_bbox: float = 1e-4,
    atol_xy: float = 1e-4,
    require_cls: bool = True,
    compare_xy: bool = True,
) -> dict:
    n_y = len(yolo)
    n_x = len(xml)
    n = min(n_y, n_x)

    mismatch_indices: list[int] = []
    max_bbox_err = 0.0
    max_xy_err = 0.0

    for i in range(n):
        y_det = yolo[i]
        x_det = xml[i].to_yolo(cls=cls_from_xml)

        cls_ok = (y_det.cls == x_det.cls) if require_cls else True
        bbox_errs = [
            _abs_diff(y_det.x_c, x_det.x_c),
            _abs_diff(y_det.y_c, x_det.y_c),
            _abs_diff(y_det.w, x_det.w),
            _abs_diff(y_det.h, x_det.h),
        ]
        bbox_ok = all(e <= atol_bbox for e in bbox_errs)
        max_bbox_err = max(max_bbox_err, max(bbox_errs, default=0.0))

        if compare_xy:
            xy_ok = True
            xy_err_vals: list[float] = []
            if y_det.x is not None and x_det.x is not None:
                e = abs(y_det.x - x_det.x)
                xy_err_vals.append(e)
                xy_ok = xy_ok and (e <= atol_xy)
            if y_det.y is not None and x_det.y is not None:
                e = abs(y_det.y - x_det.y)
                xy_err_vals.append(e)
                xy_ok = xy_ok and (e <= atol_xy)
            max_xy_err = max(max_xy_err, max(xy_err_vals, default=0.0))
        else:
            xy_ok = True

        if not (cls_ok and bbox_ok and xy_ok):
            mismatch_indices.append(i)

    order_match = (n_y == n_x) and (len(mismatch_indices) == 0)
    return {
        "n_yolo": n_y,
        "n_xml": n_x,
        "compared": n,
        "order_match": order_match,
        "mismatch_indices": mismatch_indices,
        "first_mismatch": (mismatch_indices[0] if mismatch_indices else None),
        "max_bbox_abs_err": max_bbox_err,
        "max_xy_abs_err": max_xy_err if compare_xy else None,
    }


def _iter_xml_paths(xml_dirs_or_files: Iterable[str]) -> list[str]:
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Join CVAT XML annotations with YOLO holdout labels and verify that detection order matches.",
    )
    ap.add_argument(
        "--yolo-glob",
        default="/home/klymy/data/cam4_revB/yolo_dataset/batch1_3_margin_15_awb_clahe/labels/holdout/*.txt",
        help="Glob for YOLO label files (default: %(default)s)",
    )
    ap.add_argument(
        "--xml",
        nargs="+",
        default=[
            "/home/klymy/data/cam4_revB/batch1_corrected/*.xml",
            "/home/klymy/data/cam4_revB/batch2_corrected/*.xml",
            "/home/klymy/data/cam4_revB/batch3_corrected/*.xml",
        ],
        help="XML directories/files/globs (default: batch1-3 corrected globs)",
    )
    ap.add_argument(
        "--required-box-label",
        default="Person",
        help="Only consider <box label=...> with this label (default: %(default)s). Use empty string to disable.",
    )
    ap.add_argument(
        "--yolo-id-prefix",
        default="",
        help="Optional prefix for computed YOLO ids from CVAT image names.",
    )
    ap.add_argument(
        "--atol-bbox",
        type=float,
        default=1e-3,
        help="Absolute tolerance for x_c,y_c,w,h (default: %(default)s)",
    )
    ap.add_argument(
        "--atol-xy",
        type=float,
        default=1e-3,
        help="Absolute tolerance for x,y point columns (default: %(default)s)",
    )
    ap.add_argument(
        "--no-compare-xy",
        action="store_true",
        help="Ignore the 6th/7th columns (x,y) when comparing.",
    )
    ap.add_argument(
        "--no-require-cls",
        action="store_true",
        help="Do not require class id to match between XML-derived and YOLO.",
    )
    ap.add_argument(
        "--unmatched-penalty",
        type=float,
        default=2.0,
        help="Penalty used by best-match analysis to leave a YOLO det unmatched (default: %(default)s).",
    )
    ap.add_argument(
        "--max-mismatch-details",
        type=int,
        default=10,
        help="Max mismatched IDs to print detailed mismatch analysis for (default: %(default)s).",
    )
    ap.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV output path.",
    )
    ap.add_argument(
        "--out-jsonl",
        default="",
        help="Optional JSONL output path (one JSON object per line).",
    )
    ap.add_argument(
        "--out-json",
        default="",
        help="Alias for --out-jsonl (kept for backward compatibility).",
    )
    ap.add_argument(
        "--report-all",
        action="store_true",
        help="Write all records to JSONL, not only mismatches.",
    )
    ap.add_argument(
        "--report-format",
        choices=["minimal", "full"],
        default="minimal",
        help="JSONL schema. minimal={yolo_id,yolo_path,xml_path,xml_image_name,mapping}; full=debug-rich (default: %(default)s).",
    )

    args = ap.parse_args()

    required_label = args.required_box_label if args.required_box_label else None

    xml_paths = _iter_xml_paths(args.xml)
    if not xml_paths:
        raise SystemExit("No XML files found (check --xml).")

    by_yolo_id: dict[str, dict] = {}
    dupes = 0
    for xp in xml_paths:
        parsed = parse_cvat_xml(xp, yolo_id_prefix=args.yolo_id_prefix, required_box_label=required_label)
        for k, v in parsed.items():
            if k in by_yolo_id:
                dupes += 1
                continue
            by_yolo_id[k] = v

    yolo_paths = sorted(glob.glob(args.yolo_glob))
    if not yolo_paths:
        raise SystemExit("No YOLO label files found (check --yolo-glob).")

    rows: list[dict] = []
    n_missing_xml = 0
    n_ok = 0
    n_bad = 0
    bad_ids: list[str] = []
    mismatch_details_printed = 0

    out_jsonl = args.out_jsonl or args.out_json
    json_f = None
    if out_jsonl:
        Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        json_f = open(out_jsonl, "wt")

    def _mapping_obj_from_yolo_to_xml_list(y2x: list[int | None]) -> dict:
        y_idx: list[int] = []
        x_idx: list[int] = []
        for yi, xi in enumerate(y2x):
            if xi is None:
                continue
            y_idx.append(yi)
            x_idx.append(int(xi))
        return {"yolo": y_idx, "xml": x_idx}

    for yp in yolo_paths:
        yolo_id = Path(yp).with_suffix("").name
        xml_rec = by_yolo_id.get(yolo_id)
        yolo_dets = parse_yolo_label_file(yp)

        if xml_rec is None:
            n_missing_xml += 1
            mismatch_analysis = missing_xml_mismatch_analysis(len(yolo_dets))
            row = {
                "yolo_id": yolo_id,
                "yolo_path": yp,
                "xml_path": "",
                "xml_image_name": "",
                "n_yolo": len(yolo_dets),
                "n_xml": "",
                "order_match": False,
                "missing_xml": True,
                "first_mismatch": "",
                "mismatch_count": "",
                "max_bbox_abs_err": "",
                "max_xy_abs_err": "",
                "mismatch_type": "missing_xml",
                "direct_mismatch_indices": "",
                "suggested_xml_index_for_yolo_index": "",
                "unmatched_yolo_indices": "",
                "unmatched_xml_indices": "",
                "reordered_pairs_summary": "",
            }
            rows.append(row)
            bad_ids.append(yolo_id)
            if json_f is not None:
                if args.report_format == "minimal":
                    rec = {
                        "yolo_id": yolo_id,
                        "yolo_path": yp,
                        "xml_path": "",
                        "xml_image_name": "",
                        "mapping": _mapping_obj_from_yolo_to_xml_list(
                            mismatch_analysis["suggested_xml_index_for_yolo_index"]
                        ),
                    }
                else:
                    rec = {
                        "yolo_id": yolo_id,
                        "yolo_path": yp,
                        "xml_path": "",
                        "xml_image_name": "",
                        "compare": {"n_yolo": len(yolo_dets), "n_xml": None, "order_match": False},
                        "mismatch_analysis": mismatch_analysis,
                    }
                json_f.write(json.dumps(rec) + "\n")
            continue

        xml_dets: list[XmlDet] = xml_rec["dets"]
        cmp_res = compare_dets_in_order(
            yolo_dets,
            xml_dets,
            atol_bbox=args.atol_bbox,
            atol_xy=args.atol_xy,
            require_cls=not args.no_require_cls,
            compare_xy=not args.no_compare_xy,
        )
        mismatch_analysis = None
        if not cmp_res["order_match"]:
            mismatch_analysis = analyze_order_mismatch(
                yolo_dets,
                xml_dets,
                cls_from_xml=0,
                atol_bbox=args.atol_bbox,
                atol_xy=args.atol_xy,
                require_cls=not args.no_require_cls,
                compare_xy=not args.no_compare_xy,
                unmatched_penalty=args.unmatched_penalty,
            )

        if cmp_res["order_match"]:
            n_ok += 1
        else:
            n_bad += 1
            bad_ids.append(yolo_id)
            if mismatch_analysis is not None and mismatch_details_printed < args.max_mismatch_details:
                mismatch_details_printed += 1
                print(f"\nMismatch details for: {yolo_id}")
                print(f"- xml_image_name: {xml_rec['image_name']}")
                print(f"- mismatch_type: {mismatch_analysis['mismatch_type']}")
                print(f"- direct_mismatch_indices: {mismatch_analysis['direct_mismatch_indices']}")
                print(
                    f"- suggested xml index per yolo index: {mismatch_analysis['suggested_xml_index_for_yolo_index']}"
                )
                if mismatch_analysis["unmatched_yolo_indices"]:
                    print(f"- unmatched_yolo_indices: {mismatch_analysis['unmatched_yolo_indices']}")
                if mismatch_analysis["unmatched_xml_indices"]:
                    print(f"- unmatched_xml_indices: {mismatch_analysis['unmatched_xml_indices']}")
                if mismatch_analysis["reordered_pairs_summary"]:
                    print(f"- reordered_pairs_summary: {mismatch_analysis['reordered_pairs_summary']}")

        row = {
            "yolo_id": yolo_id,
            "yolo_path": yp,
            "xml_path": xml_rec["xml_path"],
            "xml_image_name": xml_rec["image_name"],
            "n_yolo": cmp_res["n_yolo"],
            "n_xml": cmp_res["n_xml"],
            "order_match": cmp_res["order_match"],
            "missing_xml": False,
            "first_mismatch": "" if cmp_res["first_mismatch"] is None else int(cmp_res["first_mismatch"]),
            "mismatch_count": len(cmp_res["mismatch_indices"]),
            "max_bbox_abs_err": cmp_res["max_bbox_abs_err"],
            "max_xy_abs_err": "" if cmp_res["max_xy_abs_err"] is None else cmp_res["max_xy_abs_err"],
            "mismatch_type": "" if mismatch_analysis is None else mismatch_analysis["mismatch_type"],
            "direct_mismatch_indices": "" if mismatch_analysis is None else json.dumps(
                mismatch_analysis["direct_mismatch_indices"]
            ),
            "suggested_xml_index_for_yolo_index": "" if mismatch_analysis is None else json.dumps(
                mismatch_analysis["suggested_xml_index_for_yolo_index"]
            ),
            "unmatched_yolo_indices": "" if mismatch_analysis is None else json.dumps(
                mismatch_analysis["unmatched_yolo_indices"]
            ),
            "unmatched_xml_indices": "" if mismatch_analysis is None else json.dumps(
                mismatch_analysis["unmatched_xml_indices"]
            ),
            "reordered_pairs_summary": "" if mismatch_analysis is None else json.dumps(
                mismatch_analysis["reordered_pairs_summary"]
            ),
        }
        rows.append(row)

        if json_f is not None and (args.report_all or (not cmp_res["order_match"])):
            if cmp_res["order_match"]:
                mapping_list: list[int | None] = list(range(len(yolo_dets)))
            else:
                mapping_list = (
                    mismatch_analysis["suggested_xml_index_for_yolo_index"] if mismatch_analysis is not None else []
                )

            if args.report_format == "minimal":
                rec = {
                    "yolo_id": yolo_id,
                    "yolo_path": yp,
                    "xml_path": xml_rec["xml_path"],
                    "xml_image_name": xml_rec["image_name"],
                    "mapping": _mapping_obj_from_yolo_to_xml_list(mapping_list),
                }
            else:
                rec = {
                    "yolo_id": yolo_id,
                    "yolo_path": yp,
                    "xml_path": xml_rec["xml_path"],
                    "xml_image_name": xml_rec["image_name"],
                    "compare": cmp_res,
                    "mismatch_analysis": mismatch_analysis,
                }
            json_f.write(json.dumps(rec) + "\n")

    if json_f is not None:
        json_f.close()

    if args.out_csv:
        out_csv = args.out_csv
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    total = len(yolo_paths)
    print(f"YOLO label files: {total}")
    print(f"XML files loaded: {len(xml_paths)} (image records: {len(by_yolo_id)}, duplicates skipped: {dupes})")
    print(f"Missing XML matches: {n_missing_xml}")
    print(f"Order matches: {n_ok}")
    print(f"Order mismatches: {n_bad}")
    if bad_ids:
        show = 10
        tail = "" if len(bad_ids) <= show else f" (showing first {show})"
        print(f"Mismatched IDs: {', '.join(bad_ids[:show])}{tail}")
    if args.out_csv:
        print(f"Wrote CSV: {args.out_csv}")
    if out_jsonl:
        print(f"Wrote JSONL: {out_jsonl}")

    return 0 if n_bad == 0 and n_missing_xml == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
