import argparse
import os

import numpy as np
import torch

from ultralytics_yolov8.ultralytics import YOLO
from detection_head import DetectionHead, detection_head2tflite
from tflite_inference import TFLiteInferer


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--format', type=str, default='tflite', help='export format')
    parser.add_argument('--quantize', action='store_true', help='int8 export')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--max_ncalib_imgs', type=int, default=100, help='calibration image set size')

    return parser.parse_args()


def export_head(yolo_pth, output_pth, imgsz):
    inferer = TFLiteInferer(yolo_pth)
    y = []
    for i in inferer.yolo_output_details:
        y.append(np.zeros(i["shape"], dtype=np.float32))

    head = DetectionHead([i.shape for i in y], imgsz)
    head([torch.Tensor(i) for i in y])

    detection_head2tflite(output_pth, head, y)


if __name__ == '__main__':

    args = parser_arguments()
    # Create a dictionary of kwargs
    kwargs = vars(args)
    print(kwargs)

    model = YOLO(args.weights)

    success = model.export(
        format=args.format,
        simplify=True,
        imgsz=args.imgsz,
        data=args.data,
        int8=args.quantize,
        separate_outputs=True,
        export_hw_optimized=True,
        uint8_io_dtype=True,
        max_ncalib_imgs=args.max_ncalib_imgs
    )

    base_pth = os.path.dirname(args.weights)
    tflite_pth = os.path.join(base_pth, "best_saved_model/best_full_integer_quant.tflite")
    head_pth = os.path.join(base_pth, "head_saved_model")
    export_head(tflite_pth, head_pth, args.imgsz)
