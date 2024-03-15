import json
import os

import cv2
import numpy as np


with open("detections_tmp.json", "rt") as f:
    detections = json.load(f)

cap = cv2.VideoCapture(os.environ["VIDEO_PTH"])
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("vis.avi", cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height), True)

idx = -1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1

    dets = np.array(detections[idx])

    for det in dets:
        cv2.rectangle(frame, det[:2].astype(int), det[:2].astype(int) + det[2:4].astype(int), (0, 255, 0), 3)
        cv2.circle(frame, det[-2:].astype(int), 4, (0, 255, 0), 5)
        cv2.putText(
            frame, f"{det[4]:.02f}", det[:2].astype(int) - [-10, 10], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 2), 2,
            cv2.LINE_AA
        )

    out.write(frame)
cap.release()
out.release()
