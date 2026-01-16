from __future__ import annotations

import glob
import os
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import xml.etree.ElementTree as ET


# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("cvat_merge")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# -----------------------------
# Helpers
# -----------------------------
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _canonicalize_label_xml(label_el: ET.Element) -> Dict[str, Any]:
    """Best-effort canonical representation of a CVAT XML <label>."""
    def txt(el: ET.Element | None) -> str:
        return (el.text or "").strip() if el is not None else ""

    name = txt(label_el.find("name"))
    color = txt(label_el.find("color"))
    ltype = txt(label_el.find("type"))

    attrs = []
    attrs_el = label_el.find("attributes")
    if attrs_el is not None:
        for a in attrs_el.findall("attribute"):
            attrs.append(
                {
                    "name": txt(a.find("name")),
                    "mutable": txt(a.find("mutable")),
                    "input_type": txt(a.find("input_type")),
                    "default_value": txt(a.find("default_value")),
                    "values": txt(a.find("values")),
                }
            )
    # stable ordering
    attrs.sort(key=lambda d: (d["name"], d["input_type"], d["values"]))
    return {"name": name, "type": ltype, "color": color, "attributes": attrs}


def _same_image_dims_xml(a: ET.Element, b: ET.Element) -> bool:
    return (a.get("width"), a.get("height")) == (b.get("width"), b.get("height"))


def _indent_xml(root: ET.Element) -> None:
    # Python 3.9+: ET.indent exists
    try:
        ET.indent(root, space="  ")  # type: ignore[attr-defined]
    except Exception:
        pass


def _is_coco_dict(d: Dict[str, Any]) -> bool:
    return isinstance(d.get("images"), list) and isinstance(d.get("annotations"), list) and isinstance(d.get("categories"), list)


# -----------------------------
# 1) Merge COCO "instances_default.json"-style files
# -----------------------------
def merge_coco_instances_json(
    input_paths: List[str | Path],
    output_path: str | Path,
) -> None:
    """
    Merge multiple COCO instances JSON files (e.g. CVAT export "instances_default.json")
    into one big COCO JSON.

    Collision handling:
      - Category IDs are remapped by category name.
      - Image IDs are reassigned. Images are merged by file_name when dimensions match.
      - If same file_name but different size -> file_name renamed with __dupN and WARNING logged.
      - Annotation IDs are reassigned, and image_id/category_id remapped.

    Writes output JSON to output_path.
    """
    paths = [Path(p) for p in input_paths]
    out_path = Path(output_path)
    _ensure_parent_dir(out_path)

    merged: Dict[str, Any] = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    # category name -> new_id
    cat_name_to_new_id: Dict[str, int] = {}
    # (file_name) -> new_image_id, plus stored dims to validate
    file_to_new_image: Dict[str, Dict[str, Any]] = {}

    next_cat_id = 1
    next_img_id = 1
    next_ann_id = 1

    # keep some global info/licenses (best-effort)
    first_info_set = False
    licenses_seen: set[Tuple] = set()

    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)

        if not isinstance(d, dict) or not _is_coco_dict(d):
            raise ValueError(f"Not a COCO instances JSON (images/annotations/categories missing): {p}")

        if not first_info_set and isinstance(d.get("info"), dict):
            merged["info"] = copy.deepcopy(d["info"])
            first_info_set = True
        elif isinstance(d.get("info"), dict) and merged["info"] and d["info"] != merged["info"]:
            logger.warning(f"COCO 'info' differs in {p.name}; keeping first file's info.")

        for lic in d.get("licenses", []) or []:
            if isinstance(lic, dict):
                key = tuple(sorted(lic.items()))
                if key not in licenses_seen:
                    merged["licenses"].append(copy.deepcopy(lic))
                    licenses_seen.add(key)

        # ---- categories: merge by name ----
        old_cat_id_to_new: Dict[int, int] = {}
        for c in d["categories"]:
            if not isinstance(c, dict) or "name" not in c:
                logger.warning(f"Bad category entry in {p.name}: {c!r} (skipping)")
                continue
            name = str(c["name"])
            supercat = str(c.get("supercategory", ""))
            old_id = int(c.get("id", -1))

            if name not in cat_name_to_new_id:
                new_id = next_cat_id
                next_cat_id += 1
                cat_name_to_new_id[name] = new_id
                merged["categories"].append({"id": new_id, "name": name, "supercategory": supercat})
            else:
                # check supercategory consistency
                existing = next(x for x in merged["categories"] if x["id"] == cat_name_to_new_id[name])
                if str(existing.get("supercategory", "")) != supercat:
                    logger.warning(
                        f"Category '{name}' has different supercategory in {p.name} "
                        f"('{supercat}' vs '{existing.get('supercategory','')}'). Keeping first."
                    )

            if old_id != -1:
                old_cat_id_to_new[old_id] = cat_name_to_new_id[name]

        # ---- images: merge by file_name ----
        old_img_id_to_new: Dict[int, int] = {}
        dup_counter: Dict[str, int] = {}

        for img in d["images"]:
            if not isinstance(img, dict) or "file_name" not in img:
                logger.warning(f"Bad image entry in {p.name}: {img!r} (skipping)")
                continue

            old_id = int(img.get("id", -1))
            file_name = str(img["file_name"])
            width = img.get("width")
            height = img.get("height")

            if file_name in file_to_new_image:
                existing = file_to_new_image[file_name]
                if existing.get("width") == width and existing.get("height") == height:
                    # merge: map to existing image id
                    old_img_id_to_new[old_id] = int(existing["id"])
                else:
                    # collision: same file_name, different dims -> rename
                    dup_counter[file_name] = dup_counter.get(file_name, 0) + 1
                    new_file_name = f"{file_name}__dup{dup_counter[file_name]}"
                    logger.warning(
                        f"Image file_name collision with different size: '{file_name}' in {p.name}. "
                        f"Renaming to '{new_file_name}'."
                    )
                    new_id = next_img_id
                    next_img_id += 1
                    new_img = copy.deepcopy(img)
                    new_img["id"] = new_id
                    new_img["file_name"] = new_file_name
                    merged["images"].append(new_img)
                    file_to_new_image[new_file_name] = new_img
                    old_img_id_to_new[old_id] = new_id
            else:
                new_id = next_img_id
                next_img_id += 1
                new_img = copy.deepcopy(img)
                new_img["id"] = new_id
                merged["images"].append(new_img)
                file_to_new_image[file_name] = new_img
                old_img_id_to_new[old_id] = new_id

        # ---- annotations: remap ids ----
        for ann in d["annotations"]:
            if not isinstance(ann, dict):
                logger.warning(f"Bad annotation entry in {p.name}: {ann!r} (skipping)")
                continue

            old_img_id = int(ann.get("image_id", -1))
            old_cat_id = int(ann.get("category_id", -1))

            if old_img_id not in old_img_id_to_new:
                logger.warning(f"Annotation references unknown image_id={old_img_id} in {p.name} (skipping ann)")
                continue
            if old_cat_id not in old_cat_id_to_new:
                logger.warning(f"Annotation references unknown category_id={old_cat_id} in {p.name} (skipping ann)")
                continue

            new_ann = copy.deepcopy(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = old_img_id_to_new[old_img_id]
            new_ann["category_id"] = old_cat_id_to_new[old_cat_id]

            merged["annotations"].append(new_ann)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(
        f"COCO merge done: {len(paths)} files -> "
        f"{len(merged['images'])} images, {len(merged['annotations'])} annotations, {len(merged['categories'])} categories "
        f"written to {out_path}"
    )


# -----------------------------
# 2) Merge CVAT XML annotation files
# -----------------------------
def merge_cvat_xml_annotations(
    input_paths: List[str | Path],
    output_path: str | Path,
) -> None:
    """
    Merge multiple CVAT XML files (<annotations>...</annotations>) into one big CVAT XML.

    Prints stats:
      - images count
      - annotations count (shapes under images + shapes inside tracks)
      - categories count (labels in meta)
    """
    paths = [Path(p) for p in input_paths]
    out_path = Path(output_path)
    _ensure_parent_dir(out_path)

    if not paths:
        raise ValueError("input_paths is empty")

    base_tree = ET.parse(paths[0])
    base_root = base_tree.getroot()
    if base_root.tag != "annotations":
        raise ValueError(f"Not a CVAT XML (<annotations> root expected): {paths[0]}")

    out_root = ET.Element("annotations")

    base_version = base_root.find("version")
    if base_version is not None:
        out_root.append(copy.deepcopy(base_version))

    base_meta = base_root.find("meta")
    if base_meta is not None:
        out_root.append(copy.deepcopy(base_meta))
    else:
        logger.warning(f"No <meta> found in {paths[0].name}; output will not have meta.")

    def get_labels_el(r: ET.Element) -> ET.Element | None:
        meta = r.find("meta")
        if meta is None:
            return None
        job = meta.find("job")
        task = meta.find("task")
        if job is not None and job.find("labels") is not None:
            return job.find("labels")
        if task is not None and task.find("labels") is not None:
            return task.find("labels")
        return None

    out_labels_el = get_labels_el(out_root)
    if out_labels_el is None and base_meta is not None:
        out_labels_el = get_labels_el(out_root)

    label_by_name: Dict[str, Dict[str, Any]] = {}
    if out_labels_el is not None:
        for lab in out_labels_el.findall("label"):
            rep = _canonicalize_label_xml(lab)
            if rep["name"]:
                label_by_name[rep["name"]] = rep

    image_name_to_el: Dict[str, ET.Element] = {}
    image_name_to_dims: Dict[str, Tuple[str | None, str | None]] = {}
    image_dup_counter: Dict[str, int] = {}

    next_image_id = 0
    next_ann_id = 1

    def assign_new_ann_ids(image_el: ET.Element) -> None:
        nonlocal next_ann_id
        for child in list(image_el):
            if "id" in child.attrib:
                child.set("id", str(next_ann_id))
                next_ann_id += 1
            for sub in child.iter():
                if sub is child:
                    continue
                if "id" in sub.attrib:
                    sub.set("id", str(next_ann_id))
                    next_ann_id += 1

    def add_new_image_from(src_image: ET.Element) -> ET.Element:
        nonlocal next_image_id
        new_image = ET.Element("image", attrib=dict(src_image.attrib))
        new_image.set("id", str(next_image_id))
        next_image_id += 1
        for ch in list(src_image):
            new_image.append(copy.deepcopy(ch))
        assign_new_ann_ids(new_image)
        out_root.append(new_image)
        return new_image

    # ---- track merging + id reassignment ----
    def merge_tracks(src_root: ET.Element) -> None:
        nonlocal next_ann_id
        for tr in src_root.findall("track"):
            tr_copy = copy.deepcopy(tr)
            if "id" in tr_copy.attrib:
                tr_copy.set("id", str(next_ann_id))
                next_ann_id += 1
            for node in tr_copy.iter():
                if "id" in node.attrib and node is not tr_copy:
                    node.set("id", str(next_ann_id))
                    next_ann_id += 1
            out_root.append(tr_copy)

    # seed images + tracks from base
    for img in base_root.findall("image"):
        name = img.get("name", "")
        out_img = add_new_image_from(img)
        image_name_to_el[name] = out_img
        image_name_to_dims[name] = (img.get("width"), img.get("height"))

    merge_tracks(base_root)

    # merge remaining files
    for p in paths[1:]:
        tree = ET.parse(p)
        root = tree.getroot()
        if root.tag != "annotations":
            logger.warning(f"Skipping non-CVAT XML (root != annotations): {p.name}")
            continue

        # labels union / validation
        labels_el = get_labels_el(root)
        if labels_el is not None and out_labels_el is not None:
            for lab in labels_el.findall("label"):
                rep = _canonicalize_label_xml(lab)
                nm = rep["name"]
                if not nm:
                    continue
                if nm not in label_by_name:
                    logger.warning(f"New label '{nm}' found in {p.name}; adding to output meta/labels.")
                    out_labels_el.append(copy.deepcopy(lab))
                    label_by_name[nm] = rep
                else:
                    if rep != label_by_name[nm]:
                        logger.warning(
                            f"Label '{nm}' differs in {p.name} vs base meta. "
                            f"Keeping base definition; annotations will still be merged."
                        )

        # images
        for img in root.findall("image"):
            name = img.get("name", "")
            dims = (img.get("width"), img.get("height"))

            if name in image_name_to_el:
                existing_dims = image_name_to_dims.get(name)
                if existing_dims == dims:
                    dst_img = image_name_to_el[name]
                    for ch in list(img):
                        dst_img.append(copy.deepcopy(ch))
                    assign_new_ann_ids(dst_img)
                else:
                    image_dup_counter[name] = image_dup_counter.get(name, 0) + 1
                    new_name = f"{name}__dup{image_dup_counter[name]}"
                    logger.warning(
                        f"Image name collision with different size in {p.name}: '{name}' "
                        f"{dims} vs existing {existing_dims}. Renaming to '{new_name}'."
                    )
                    img_copy = copy.deepcopy(img)
                    img_copy.set("name", new_name)
                    out_img = add_new_image_from(img_copy)
                    image_name_to_el[new_name] = out_img
                    image_name_to_dims[new_name] = (img_copy.get("width"), img_copy.get("height"))
            else:
                out_img = add_new_image_from(img)
                image_name_to_el[name] = out_img
                image_name_to_dims[name] = dims

        # tracks
        merge_tracks(root)

    # -----------------------------
    # Stats (XML)
    # -----------------------------
    def count_categories() -> int:
        labels = get_labels_el(out_root)
        return len(labels.findall("label")) if labels is not None else 0

    def count_image_shapes() -> int:
        # In CVAT image-mode: shapes are direct children of <image> (box, polygon, points, polyline, mask, cuboid, ellipse, tag, etc.)
        # We'll count all direct children except <attribute> (attributes are nested under shapes).
        shapes = 0
        for img in out_root.findall("image"):
            for ch in list(img):
                if ch.tag == "attribute":
                    continue
                shapes += 1
        return shapes

    def count_track_shapes() -> int:
        # In CVAT video-mode: <track> contains shapes over time (box/polygon/points/...), plus possibly <attribute>
        shapes = 0
        for tr in out_root.findall("track"):
            for ch in list(tr):
                if ch.tag == "attribute":
                    continue
                shapes += 1
        return shapes

    images_count = len(out_root.findall("image"))
    annotations_count = count_image_shapes() + count_track_shapes()
    categories_count = count_categories()

    _indent_xml(out_root)
    ET.ElementTree(out_root).write(out_path, encoding="utf-8", xml_declaration=True)

    logger.info(
        f"CVAT XML merge done: {len(paths)} files -> "
        f"{images_count} images, {annotations_count} annotations, {categories_count} categories "
        f"written to {out_path}"
    )


# -----------------------------
# Example usage (optional)
# -----------------------------
if __name__ == "__main__":
    # Example:
    # merge_cvat_xml_annotations(["annotations.xml", "annotations 2.xml"], "merged_annotations.xml")
    # merge_coco_instances_json(["instances_default.json", "instances_default_2.json"], "merged_instances.json")
    ROOT_PTH = "/home/klymy/Downloads/b11_annotation"
    xml_names = glob.glob(os.path.join(ROOT_PTH, "*/*.xml"))
    json_names = glob.glob(os.path.join(ROOT_PTH, "*/*/*.json"))
    print(len(xml_names))
    print(len(json_names))
    print("\n----")

    merge_cvat_xml_annotations(xml_names, os.path.join(ROOT_PTH, "annotations.xml"))
    merge_coco_instances_json(json_names, os.path.join(ROOT_PTH, "instances_default.json"))
