import os
import argparse
import json
import numpy as np
from tqdm import tqdm

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run ByteTrack")
    parser.add_argument("--track_thresh", type=float, default=0.35, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=10, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    parser.add_argument("--input_name", default="detections.json")
    parser.add_argument("--output_name", default="tracks.json")
    args = parser.parse_args()

    info_imgs = 640, 480
    img_size = 640, 480
    tracker = BYTETracker(args)

    with open(os.path.join("artifacts", args.input_name), "rt") as f:
        detections = json.load(f)

    tracks = []
    for dets in tqdm(detections):
        dets_in = dets["detections"].copy()
        dets = np.array(dets["detections"], dtype=float)
        if dets.size:
            dets[:, 2:4] = dets[:, :2] + dets[:, 2:4]
            dets = np.hstack([dets[:, :5], dets[:, 6:]])
        else:
            dets = np.empty((1, 7))
        online_targets = tracker.update(dets, info_imgs, img_size)

        proceses_targets = []
        for online_target in online_targets:
            box = np.array(online_target.tlwh)
            box[box < 0 ] = 0
            proceses_targets.append({
                "id": online_target.track_id,
                "state": online_target.state,
                "is_activated": online_target.is_activated,
                "tlwh": box,
                "tlwh_pred": online_target.tlwh_predicted,
                "pos": online_target.pos,
                "pos_pred": online_target.pos_predicted,
                "score": online_target.score
            })
        tracks.append({"detections": dets_in, "tracks": proceses_targets})

    with open(os.path.join("artifacts", args.output_name), "wt") as f:
        json.dump(tracks, f, cls=NumpyEncoder)

