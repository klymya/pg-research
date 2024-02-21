import json
import os
import argparse

import cv2
import numpy as np


COLOR_DET = (255, 0, 0)
COLOR_TRACK = (0, 255, 0)
COLOR_TRACK_PRED = (0, 0, 255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("vis tracks")
    parser.add_argument("--video_path")
    parser.add_argument("--input_name", default="tracks.json")
    parser.add_argument("--output_name", default="vis.avi")
    parser.add_argument("--ignore_dets", action="store_true", default=False)
    parser.add_argument("--ignore_tracks", action="store_true", default=False)
    parser.add_argument("--ignore_kalman", action="store_true", default=False)
    args = parser.parse_args()

    with open(os.path.join("artifacts", args.input_name), "rt") as f:
        detections = json.load(f)

    cap = cv2.VideoCapture(args.video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join("artifacts", args.output_name), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height), True)

    idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        idx += 1
        
        cv2.putText(
                frame, f"{idx}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2,
                cv2.LINE_AA
            )
        
        if not args.ignore_dets:
            frame_data = detections[idx]
            for det in np.array(frame_data["detections"]):
                p1 = det[:2].astype(int)
                p2 = det[:2].astype(int) + det[2:4].astype(int)
                p1[0] = max(p1[0], 0)
                p1[1] = max(p1[1], 0)
                p2[0] = min(p2[0], w)
                p2[1] = min(p2[1], h)
                cv2.rectangle(
                    frame, 
                    p1,
                    p2,
                    COLOR_DET,
                    3
                )
                cv2.putText(
                    frame, f"{det[4]:.02f}", p1 + [5, 10], cv2.FONT_HERSHEY_PLAIN, 0.7, COLOR_DET, 1,
                    cv2.LINE_AA
                )
        
        for track in frame_data["tracks"]:
            if not args.ignore_kalman:
                box = np.array(track["tlwh_pred"]).astype(int)
                p1 = box[:2].astype(int)
                p2 = box[:2].astype(int) + box[2:4].astype(int)
                p1[0] = max(p1[0], 0)
                p1[1] = max(p1[1], 0)
                p2[0] = min(p2[0], w)
                p2[1] = min(p2[1], h)
                cv2.rectangle(
                    frame, 
                    p1,
                    p2,
                    COLOR_TRACK_PRED,
                    2
                )

            if not args.ignore_tracks:
                box = np.array(track["tlwh"]).astype(int)
                p1 = box[:2].astype(int)
                p2 = box[:2].astype(int) + box[2:4].astype(int)
                p1[0] = max(p1[0], 0)
                p1[1] = max(p1[1], 0)
                p2[0] = min(p2[0], w)
                p2[1] = min(p2[1], h)
                cv2.rectangle(
                    frame, 
                    p1,
                    p2,
                    COLOR_TRACK,
                    1
                )
                
                x = (p2[0] + p1[0]) // 2 - 15
                y = (p2[1] + p1[1]) // 2
                cv2.putText(
                    frame, f"{track['id']}. {track['state']}. {track['is_activated']}", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA
                )

        out.write(frame)
    cap.release()
    out.release()
