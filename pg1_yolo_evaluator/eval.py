import os
import glob
import sys
from multiprocessing import Pool
import argparse
from datetime import datetime

import cv2
from ultralytics import YOLO
from tqdm import tqdm
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../PG_PyTraining"))

import simber
from common.pg.cam_params import get_camparams
from ml_training.detectors.data_loaders import extract_height_from_filename
from common.pg import pg_geometry


DATASET_PATHS = [
    "/mnt/pg-fs01/Current/Benchmarks/PG1_AlgoDev_DLtest02_RV9",
    "/mnt/pg-fs01/Current/Benchmarks/PG1_AlgoDev_DLtest_RV59",
    "/mnt/pg-fs01/Current/Benchmarks/PG1_AlgoDev_VLC_RV5_V2",
    "/mnt/pg-fs01/Current/DB/CV_PeopleTracking/Benchmarks/Mixed/VLC_MultiplePeopleCrossing/10fps",
]


LIINE_TEMPLATE = \
    "frame {frame_n} PG_OCSManagerImpl::PrintOneTrackToPos TRACK_USER_{IDX} {l} {t} {w} {h}  "\
    "POSITION_{x_cm}*{y_cm} STATUS_STANDING HEIGHT_0 ORIGINALPOSITION_{x}*{y}"


class Processor:
    def __init__(self, model_path, conf, iou, device, output_root_dir):
        self.model_path = model_path
        self.output_root_dir = output_root_dir

        self.args = dict(
            task="pose",
            device=[device],
            batch=128,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=320,
            model="val",
            rect=False,
            single_cls=False,
            stream=True,
        )

    @staticmethod
    def flip_video(in_path, out_path, flip_code):
        # Open the video file
        cap = cv2.VideoCapture(in_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the frame
            flipped_frame = cv2.flip(frame, flip_code)

            # Write the flipped frame to output
            out.write(flipped_frame)

        # Release everything
        cap.release()
        out.release()

    def process_video(self, video_path):
        os.makedirs(self.output_root_dir, exist_ok=True)
        output_path = os.path.basename(video_path)

        model = YOLO(self.model_path)

        cam_params = get_camparams(video_path)
        cam_height = extract_height_from_filename(video_path)
        cam = pg_geometry.PgCamera(cam_height * 10, cam_params.NAME)

        self.flip_video(video_path, output_path, cam_params.FLIP_CODE)

        lines = []
        frame_count = 0
        for i in model.predict(output_path, **self.args):
            xywhs = i.boxes.xywh.detach().cpu().numpy()
            probs = i.boxes.conf.detach().cpu().numpy()
            xys = i.keypoints.xy.detach().cpu().numpy()

            for idx, (xywh, xy) in enumerate(zip(xywhs, xys)):
                l, t, w, h = xywh
                x, y = xy[0]
                x_cm, y_cm = cam.pixel2cm([[x, y]])[0]
                lines.append(
                    LIINE_TEMPLATE.format(
                        frame_n=frame_count,
                        IDX=idx,
                        l=int(l - w / 2),
                        t=int(t - h / 2),
                        w=int(w),
                        h=int(h),
                        x_cm=int(x_cm),
                        y_cm=int(y_cm),
                        x=int(x),
                        y=int(y)
                    )
                )
            frame_count += 1

        torch.cuda.empty_cache()
        os.remove(output_path)

        header_lines = [
            video_path,
            f";META ceiling_height_cm {cam_height}",
            "frame 0 PG_MainThread::Run START",
        ]

        with open(os.path.join(self.output_root_dir, output_path.replace('.avi', '.pos')), "w") as f:
            f.write("\n".join(header_lines + lines + [f"frame {frame_count} PG_MainThread::Run END"]))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--weights",
        type=str,
        default="../people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt",
        help="Path to the YOLOv8 model file."
    )
    argparser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for detection."
    )
    argparser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for NMS."
    )
    argparser.add_argument(
        "--device",
        type=int,
        default=1,
        help="Device ID for GPU processing."
    )
    argparser.add_argument(
        "--output_root_dir",
        type=str,
        default=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output directory for processed files."
    )
    args = argparser.parse_args()

    for dataset_pth in DATASET_PATHS:
        dataset_name = os.path.basename(dataset_pth)

        seq_paths = glob.glob(os.path.join(dataset_pth, "*.seq"))
        video_paths = simber.videos_from_seq_files(seq_paths)

        processor = Processor(
            model_path=args.weights,
            conf=args.conf,
            iou=args.iou,
            device=int(args.device),
            output_root_dir=os.path.join(args.output_root_dir, dataset_name),
        )

        with Pool(4) as p:
            r = list(tqdm(p.imap(processor.process_video, video_paths), total=len(video_paths), desc=dataset_name))
