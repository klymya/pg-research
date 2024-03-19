import json
import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm

from model import Model


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("run yolov8 model")
    parser.add_argument("--video_path")
    parser.add_argument("--out_filename", default="detections.json")
    parser.add_argument("--conf_th", type=float, default=0.25)
    parser.add_argument("--num_threads", type=int, default=4)
    args = parser.parse_args()

    model = Model(
        "../leg_position_detectors/yolov8/run_results/model_2_5m_params/320/weights/best_saved_model/best_full_integer_quant.tflite",
        "../leg_position_detectors/yolov8/detection_head/2_5m_model/320/saved_model/head_320_float32.tflite",
        conf_th=args.conf_th,
        num_threads=args.num_threads
    )

    cap = cv2.VideoCapture(args.video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []
    pbar = tqdm(total=n_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = model.predict(frame)
        results.append({"detections": detections})
        pbar.update(1)

    with open(os.path.join("artifacts", args.out_filename), "wt") as f:
        json.dump(results, f, cls=NumpyEncoder)
