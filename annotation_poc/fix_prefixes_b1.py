from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger("prefix_replace")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _norm_prefix(p: str) -> str:
    """
    Normalize prefixes to a consistent form:
    - use forward slashes
    - ensure ends with '/'
    """
    p = p.replace("\\", "/").strip()
    if p and not p.endswith("/"):
        p += "/"
    return p


def _replace_one(path_str: str, old_prefixes: List[str], new_prefix: str) -> Tuple[str, str | None, List[str]]:
    """
    Replace the first matching old prefix with new_prefix.
    Returns: (new_path, used_prefix_or_None, matched_prefixes_list)
    """
    s = (path_str or "").replace("\\", "/")
    old = [_norm_prefix(x) for x in old_prefixes]
    newp = _norm_prefix(new_prefix)

    matched = [p for p in old if s.startswith(p)]
    if not matched:
        return s, None, []

    # If more than one matches, it can happen if prefixes overlap (e.g. "2em/" and "2em/sub/")
    used = matched[0]  # "first in provided order" after normalization
    # But keep deterministic wrt input order:
    # pick the first prefix from user's order that matches
    for p in old:
        if s.startswith(p):
            used = p
            break

    return newp + s[len(used):], used, matched


# -------------------------------------------------------------------
# 1) COCO JSON (CVAT instances_default.json) prefix replace
# -------------------------------------------------------------------
def replace_prefixes_coco_json(
    input_json_path: str | Path,
    output_json_path: str | Path,
    old_prefixes: List[str],
    new_prefix: str,
) -> None:
    """
    Replace prefixes in COCO images[*].file_name.

    Warnings:
      - no prefix match for an image path
      - multiple prefixes match (ambiguous overlap)
      - duplicates created after replacement
    """
    in_path = Path(input_json_path)
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data: Dict[str, Any] = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "images" not in data or not isinstance(data["images"], list):
        raise ValueError(f"Not a COCO instances JSON (missing 'images' list): {in_path}")

    before_to_after: Dict[str, str] = {}
    after_counts: Dict[str, int] = {}

    no_match = 0
    ambiguous = 0
    changed = 0

    for img in data["images"]:
        if not isinstance(img, dict):
            continue
        if "file_name" not in img:
            continue

        old_name = str(img["file_name"])
        new_name, used, matched = _replace_one(old_name, old_prefixes, new_prefix)

        if used is None:
            no_match += 1
            logger.warning(f"COCO: no prefix match for file_name='{old_name}'")
        else:
            if len(matched) > 1:
                ambiguous += 1
                logger.warning(
                    f"COCO: multiple prefixes match for '{old_name}': {matched}. "
                    f"Using '{used}' (based on your prefix order)."
                )
            if new_name != old_name:
                img["file_name"] = new_name
                changed += 1

        before_to_after[old_name] = new_name
        after_counts[new_name] = after_counts.get(new_name, 0) + 1

    # detect duplicates after replacement
    dups = [k for k, v in after_counts.items() if v > 1]
    if dups:
        logger.warning(f"COCO: {len(dups)} duplicate file_name(s) after replacement. Example: {dups[:5]}")

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"COCO prefix replace done: changed={changed}, no_match={no_match}, ambiguous={ambiguous}, "
        f"written to {out_path}"
    )


# -------------------------------------------------------------------
# 2) CVAT XML prefix replace
# -------------------------------------------------------------------
def replace_prefixes_cvat_xml(
    input_xml_path: str | Path,
    output_xml_path: str | Path,
    old_prefixes: List[str],
    new_prefix: str,
) -> None:
    """
    Replace prefixes in CVAT XML:
      - <image name="...">
    (For video-mode exports, CVAT still uses <image name=...> for frames in some exports;
     if your XML has no <image>, nothing will change.)

    Warnings:
      - no prefix match for an image name
      - multiple prefixes match (ambiguous overlap)
      - duplicates created after replacement
    """
    in_path = Path(input_xml_path)
    out_path = Path(output_xml_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(in_path)
    root = tree.getroot()
    if root.tag != "annotations":
        raise ValueError(f"Not a CVAT XML (<annotations> root expected): {in_path}")

    after_counts: Dict[str, int] = {}

    no_match = 0
    ambiguous = 0
    changed = 0

    for img in root.findall("image"):
        old_name = img.get("name", "") or ""
        new_name, used, matched = _replace_one(old_name, old_prefixes, new_prefix)

        if used is None:
            no_match += 1
            logger.warning(f"XML: no prefix match for image name='{old_name}'")
        else:
            if len(matched) > 1:
                ambiguous += 1
                logger.warning(
                    f"XML: multiple prefixes match for '{old_name}': {matched}. "
                    f"Using '{used}' (based on your prefix order)."
                )
            if new_name != old_name:
                img.set("name", new_name)
                changed += 1

        after_counts[new_name] = after_counts.get(new_name, 0) + 1

    dups = [k for k, v in after_counts.items() if v > 1]
    if dups:
        logger.warning(f"XML: {len(dups)} duplicate image name(s) after replacement. Example: {dups[:5]}")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    logger.info(
        f"CVAT XML prefix replace done: changed={changed}, no_match={no_match}, ambiguous={ambiguous}, "
        f"written to {out_path}"
    )


# -------------------------------------------------------------------
# Example
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example: "2em/4.2_H_cam_229_Kitchen_1039.tif" -> "tmp/4.2_H_cam_229_Kitchen_1039.tif"
    old_prefixes = [
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
    ]
    new_prefix = "4.2_batch12/"

    BASE_PTH = "/home/klymy/data/cam4_revB/4.2_batch12"
    CORRECTED_PTH = "/home/klymy/data/cam4_revB/batch12_corrected"

    replace_prefixes_coco_json(
        os.path.join(BASE_PTH, "instances_default.json"),
        os.path.join(CORRECTED_PTH, "batch12_corrected.json"),
        old_prefixes, new_prefix
    )
    replace_prefixes_cvat_xml(
        os.path.join(BASE_PTH, "annotations.xml"),
        os.path.join(CORRECTED_PTH, "batch12_corrected.xml"),
        old_prefixes, new_prefix
    )
