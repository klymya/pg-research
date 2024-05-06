import os
import argparse

import numpy as np
import cv2

import models


RES_FOLDER = "output"
MARGIN = 5
COLORS = np.random.uniform(0, 255, size=(len(models.CLASSES), 3))

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draws bounding boxes on the input image based on the provided arguments.

    Args:
        img (numpy.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{models.CLASSES[class_id]} ({confidence:.2f})"
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_img(det, seg, img):
    detections = det.infer(img)

    overlay = np.zeros_like(img, dtype=np.uint8)
    for det in detections:
        if det["class_name"] == "table":
            box = det["box"]
            patch = img[
                box[1] - MARGIN: box[1] + box[3] + MARGIN,
                box[0] - MARGIN: box[0] + box[2] + MARGIN,
            ]
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                continue

            output = seg.infer(patch)
            output = np.round(output).astype(np.uint8) * 255

            overlay[
                box[1] - MARGIN: box[1] + box[3] + MARGIN,
                box[0] - MARGIN: box[0] + box[2] + MARGIN,
                1
            ] = output


    alpha = 0.4  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    output_image = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Iterate through NMS results to draw bounding boxes and labels
    for det in detections:
        if det["class_name"] not in ['chair', 'table', 'armchair', 'tv']:
            continue
        box = det['box']

        draw_bounding_box(
            output_image,
            det["class_id"],
            det["confidence"],
            round(box[0]),
            round(box[1]),
            round(box[0] + box[2]),
            round(box[1] + box[3]),
        )

    return output_image


def main(args):
    if args.save:
        os.makedirs(RES_FOLDER, exist_ok=True)
    
    detector = models.Detector(args.yolo)
    segmentor = models.Segmentator(args.pan)

    if os.path.isfile(args.input):
        img_paths = [args.input]
    else:
        img_paths = [
            os.path.join(args.input, i) for i in os.listdir(args.input)
            if os.path.isfile(os.path.join(args.input, i)) and os.path.splitext(i)[-1] in [".png", ".jpg", ".jpeg"]
        ]

    for img_pth in img_paths:
        img = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
        output_img = process_img(detector, segmentor, img)

        if args.vis:
            cv2.imshow("results", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

        if args.save:
            pth = os.path.split(img_pth)[-1]
            cv2.imwrite(os.path.join(RES_FOLDER, pth), cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))


parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--yolo")
parser.add_argument("--pan")
parser.add_argument("--vis", action="store_true", default=False)
parser.add_argument("--save", action="store_true", default=False)

args = parser.parse_args()

main(args)
