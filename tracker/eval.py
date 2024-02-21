import json
import os
import argparse

import numpy as np

from common.pg import pg_tracked_elements
from metricsevaluator import MetricsEvaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser("eval")
    parser.add_argument("--input_name", default="tracks.json")
    parser.add_argument("--video_path")
    parser.add_argument("--eval_dets", action="store_true", default=False)
    args = parser.parse_args()

    evaluator = MetricsEvaluator(
        is_relative_coordinates=False,
        is_xywh_coordinates=True,
    )

    gt_TEs_per_frame, _, _ = pg_tracked_elements.load_TEs(f"{args.video_path}.seq")

    for idx, tes in enumerate(gt_TEs_per_frame):
        for te in tes:
            rect = np.array(te.rect.ltwh())
            rect_fixed = rect.copy()
            rect_fixed[0] = 640 - rect_fixed[0]
            rect_fixed[0] = rect_fixed[0] - rect[2]
            rect_fixed[1] = 480 - rect_fixed[1]
            rect_fixed[1] = rect_fixed[1] - rect[3]

            evaluator.add_gt_box(
                img_path=idx,
                class_name="person",
                bbox=rect_fixed
            )

    with open(os.path.join("artifacts", args.input_name), "rt") as f:
        det_tes = json.load(f)

    for idx, tes in enumerate(det_tes):
        if args.eval_dets:
            for te in tes["detections"]:
                rect = np.array(te)

                evaluator.add_det_box(
                    img_path=idx,
                    class_name="person",
                    bbox=rect[:4],
                    confidence=te[4]
                )
        else:
            for te in tes["tracks"]:
                rect = np.array(te["tlwh"])

                evaluator.add_det_box(
                    img_path=idx,
                    class_name="person",
                    bbox=rect,
                    confidence=te["score"]
                )

    result = evaluator.eval_conf_vals(0.01)[0]
    precision = result["total TP"] / (result["total TP"] + result["total FP"])
    recall = result["total TP"] / (result["total positives"])
    f_score = 2 * precision * recall / (precision + recall)
    print(f"P = {precision:.03f}. R = {recall:.03f}. F1 = {f_score:.03f}. AP = {result['AP']:.03f}")
