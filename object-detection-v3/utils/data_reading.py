import os
import xml.etree.ElementTree as ET
import uuid
import random

from cvat_sdk import make_client


rd = random.Random()
rd.seed(42)
def gen_uuid() -> str:
    return str(uuid.UUID(int=rd.getrandbits(128), version=4))


def is_img(pth):
    return os.path.splitext(pth)[-1].lower() in [".png", ".jpg", ".jpeg", ".bmp"]


def parse_text(txt):
    text = txt.lower()
    if text == "true":
        return True
    elif text == "false":
        return False
    else:
        return str(text).replace("none", "")


def split_point(s):
    return s.split(";")[0].split(",")


def split_corners(s):
    return [float(j)  for i in s.split(";") for j in i.split(",")]


def process_img_node(node: ET.Element) -> list:
    img_attributes = {
        "id": gen_uuid(),
        "name": node.attrib["name"],
        "width": int(node.attrib["width"]),
        "height": int(node.attrib["height"])
    }
    if node.attrib["name"] == 'PG1_CIR_H270_office_HotDesk_13_10_27_07_OS_2016-7_5953.jpg':
        a = 1

    boxes = []
    positions = []
    corners = []
    for child in node:
        attributes = {}
        for meta in child:
            attributes[meta.attrib["name"]] = parse_text(meta.text)

        obj = {
            "label": child.attrib["label"],
            **attributes
        }
        if child.tag == "points":
            if ";" in child.attrib["points"]:
                obj["corners"] = split_corners(child.attrib["points"])
                corners.append(obj)
            else:
                obj["x"], obj["y"] = [float(i) for i in split_point(child.attrib["points"])]
                positions.append(obj)
        elif "xtl" in child.attrib:
            obj["xtl"] = float(child.attrib["xtl"])
            obj["ytl"] = float(child.attrib["ytl"])
            obj["xbr"] = float(child.attrib["xbr"])
            obj["ybr"] = float(child.attrib["ybr"])
            boxes.append(obj)

    objects = {}
    for box in boxes:
        idx = f"{box['ID']}-{box['label']}"
        objects[idx] = box
    for pos in positions:
        idx = f"{pos['ID']}-{pos['label']}"
        if idx in objects:
            objects[idx]["x"] = pos["x"]
            objects[idx]["y"] = pos["y"]
        else:
            objects[idx] = pos

    for tmp_corners in corners:
        idx = f"{tmp_corners['ID']}-{tmp_corners['label']}"
        if idx in objects:
            objects[idx]["corners"] = tmp_corners["corners"]
        else:
            objects[idx] = tmp_corners

    for i in objects.values():
        i["img_id"] = img_attributes["id"]
        i["id"] = gen_uuid()

    return list(objects.values()), img_attributes


def process_meta_node(node: ET.Element, cvat_user: str, cvat_password: str) -> dict:
    meta = {}
    for child in node:
        if child.tag not in ["task", "job"]:
            continue
        for meta_field in child:
            if meta_field.tag == "segments":
                for segment_data in meta_field[0]:
                    if segment_data.tag != "url":
                        continue
                    url = segment_data.text
                    job_id = int(url.split("/")[-1])
                    with make_client('https://app.cvat.ai', credentials=(cvat_user, cvat_password)) as client:
                        job = client.jobs.retrieve(job_id)
                        task_id = job.task_id
                    job_url = f"https://app.cvat.ai/tasks/{task_id}/jobs/{job_id}"
                    meta["task_id"] = task_id
                    meta["job_id"] = job_id
                    meta["job_url"] = job_url

    return meta


def process_annotation_file(pth: str, cvat_user: str, cvat_password: str, batch: str = None, folder: str = None) -> list:
    tree = ET.parse(pth)
    root = tree.getroot()
    objects = []
    img_attributes = []
    meta = {}
    for child in root:
        if child.tag == "meta":
            meta = process_meta_node(child, cvat_user, cvat_password)
        elif child.tag == "image":
            tmp_objects, tmp_img_attributes = process_img_node(child)
            objects += tmp_objects
            img_attributes.append(tmp_img_attributes)

    for img_attrib in img_attributes:
        img_attrib["batch"] = batch

    for img_attrib in img_attributes:
        img_attrib["folder"] = folder
    
    for img_attrib in img_attributes:
        img_attrib.update(meta)

    return objects, img_attributes