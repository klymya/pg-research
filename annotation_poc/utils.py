import os
import random

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


# Function to parse the XML file and merge bounding boxes and points with the same ID
def parse_and_merge_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = {}

    for image in root.findall('image'):
        _ = image.get('id')
        image_name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))

        annotations[image_name] = {
            'width': width,
            'height': height,
            'objects': []
        }

        boxes = {}
        points = {}

        for box in image.findall('box'):
            obj_id = box.find("attribute[@name='ID']").text
            if obj_id not in boxes:
                boxes[obj_id] = []
            boxes[obj_id].append(box)

        for point in image.findall('points'):
            obj_id = point.find("attribute[@name='ID']").text
            if obj_id not in points:
                points[obj_id] = []
            points[obj_id].append(point)

        for obj_id in boxes:
            if obj_id in points:
                for box in boxes[obj_id]:
                    for point in points[obj_id]:
                        annotations[image_name]['objects'].append({
                            'label': box.get('label'),
                            'bounding_box': [
                                float(box.get('xtl')),
                                float(box.get('ytl')),
                                float(box.get('xbr')),
                                float(box.get('ybr'))
                            ],
                            'points': [
                                (float(p.split(',')[0]), float(p.split(',')[1])) for p in point.get('points').split(';')
                            ],
                            'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
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


def draw_test_sample(annotations: dict, root_img_pth: str):
    count = 0
    N = 9

    n_rows = 3
    n_cols = 3

    plt.figure(figsize=(17, 12))
    for image_name, annotation in annotations.items():
        image_path = os.path.join(root_img_pth, image_name)  # Adjust this if images are in a different directory
        annotated_image = draw_annotations(image_path, annotation)

        plt.subplot(n_rows, n_cols, count+1)
        plt.imshow(annotated_image)
        plt.title(image_name, fontsize=7)
        plt.axis('off')
        count += 1
        if count == N:
            break
