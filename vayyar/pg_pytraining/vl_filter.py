import argparse
import glob
import json

import numpy as np

from common.pg.pg_geometry import PgDetectionArea, PgCamera
import common.pg.cam_params as cp
from ml_training.detectors.data_loaders import extract_height_from_filename
from common.data.interface import virtual_line
# 1 read vls
# 2 read da
# 3 check each vl:
# a depth
# b width
# c distance to da borders

MIN_DEPTH_TH = 50
MAX_DEPTH_TH = 200
WIDTH_TH = 90
DA_DIST_TH = 50


def get_depth(vl_points):
    return np.linalg.norm(vl_points[0] - vl_points[1])


def get_length(vl_points):
    return np.linalg.norm(vl_points[1] - vl_points[2])


def from_line_to_point(p1, p2, p3):
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)


def process_video(video_path, rm_list, thresholds):
    vls_path = video_path + '.json'
    video_name = video_path.split('/')[-1]

    cam_height_cm = extract_height_from_filename(video_path)
    cam_params = cp.get_camparams(video_path)
    camera = PgCamera(ceil_h_mm=(10 * cam_height_cm), cam_type=cam_params.NAME)

    da_poly = np.array(PgDetectionArea(camera).get_cm_polygon())
    vlds = virtual_line.read_VLs_from_json(vls_path)
    n0 = len(vlds)
    if rm_list:
        vlds = [v for v in vlds if v.id not in rm_list]
        print(f"Removed VL, because in rm list. Vide name = {video_name}. rm list = {rm_list}")
    vlds_cm = [vld.lens.pixel2cm(vld.vertices, cam_height_cm).astype(np.int) for vld in vlds]

    verified_vlds = []
    for idx, vld in enumerate(vlds_cm):
        depth = int(get_depth(vld))
        if not (thresholds["min_depth"] <= depth <= thresholds["max_depth"]):
            print(
                f"Filtered VL, because depth value. Video name = {video_name}. "
                f"VL name = {vlds[idx].name}. depth = {depth}. "
                f"threshold = {(thresholds['min_depth'], thresholds['max_depth'])}"
            )
            continue

        width = int(get_length(vld))
        if get_length(vld) < thresholds["width"]:
            print(
                f"Filtered VL, because width value. Video name = {video_name}. "
                f"VL name = {vlds[idx].name}. width = {width}. "
                f"threshold = {thresholds['width']}"
            )
            continue
        min_da_dist = 1e6
        for da_idx in range(len(da_poly) - 1):
            for v in vld:
                tmp_dist = int(from_line_to_point(da_poly[da_idx], da_poly[da_idx + 1], v))
                if tmp_dist < min_da_dist:
                    min_da_dist = tmp_dist

        if min_da_dist < thresholds["da_dist"]:
            print(
                f"Filtered VL, because min da distance value. Video name = {video_name}. "
                f"VL name = {vlds[idx].name}. min_da_dist = {min_da_dist}. "
                f"threshold = {thresholds['da_dist']}"
            )
            continue

        verified_vlds.append(vlds[idx])

    virtual_line.write_VLs_to_json(vls_path, verified_vlds, cam_height_cm, override_id=False)

    print()
    print(f"{video_name}. # of VL before = {n0}. after = {len(verified_vlds)}")
    print("==================================================================")


def process_rm_lists(filepath, video_paths):
    with open(filepath, 'rt') as f:
        data = json.load(f)

    rm_lists = []
    for video_path in video_paths:
        name = video_path.split('/')[-1]
        rm_lists.append(data.get(name, []))

    return rm_lists


def main(args):
    video_paths = sorted(glob.glob(f"{args.input_folder}/*.avi"))
    rm_list = process_rm_lists(args.rm_list, video_paths)

    thresholds = {
        "min_depth": args.min_depth_th,
        "max_depth": args.max_depth_th,
        "width": args.width_th,
        "da_dist": args.da_dist_th
    }

    for video_path, rm_list in zip(video_paths, rm_list):
        process_video(video_path, rm_list, thresholds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("vl filter")
    parser.add_argument("--input-folder", '-i')
    parser.add_argument("--rm-list", required=False)
    parser.add_argument("--min-depth-th", type=int, default=MIN_DEPTH_TH)
    parser.add_argument("--max-depth-th", type=int, default=MAX_DEPTH_TH)
    parser.add_argument("--width-th", type=int, default=WIDTH_TH)
    parser.add_argument("--da-dist-th", type=int, default=DA_DIST_TH)

    args = parser.parse_args()

    main(args)
    # TODO: add list of ids for skip
