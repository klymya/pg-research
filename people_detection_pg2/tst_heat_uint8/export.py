import argparse
import os
import sys
import glob

import numpy as np
import torch
import cv2

sys.path.append("..")
from ultralytics_yolov8.ultralytics import YOLO
from detection_head import DetectionHead, detection_head2tflite, DetectionHeadSplitted, detection_head2tflite_uint8, detection_head2tflite_separated
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


def get_broadcast_shape(input_shape):
    input_shape = np.array(input_shape)
    
    # Remove batch dimension (assumed to be first)
    # shape_wo_batch = input_shape[1:]
    shape_wo_batch = input_shape
    
    # For shape length > 1, preserve second dimension (channel), rest -> 1
    if len(shape_wo_batch) > 1:
        output_shape = [1 if i != 1 else dim for i, dim in enumerate(shape_wo_batch)]
    else:
        # For shape like [n, x], return [1]
        output_shape = [1]

    return output_shape


def export_head(yolo_pth, output_pth, imgsz):
    os.makedirs(output_pth, exist_ok=True)

    inferer = TFLiteInferer(yolo_pth)
    y = []
    for i in inferer.yolo_output_details:
        y.append(np.zeros(i["shape"], dtype=np.float32))

    head = DetectionHead([i.shape for i in y], imgsz)
    head([torch.Tensor(i) for i in y])

    detection_head2tflite(output_pth, head, y)


def export_head_separated(yolo_pth, output_pth, imgsz):
    os.makedirs(output_pth, exist_ok=True)

    inferer = TFLiteInferer(yolo_pth)
    y = []
    for i in inferer.yolo_output_details:
        y.append(np.zeros(i["shape"], dtype=np.float32))

    head = DetectionHeadSplitted([i.shape for i in y], imgsz)
    head([torch.Tensor(i) for i in y])

    detection_head2tflite_separated(output_pth, head, y)


def norm_data(x: np.ndarray, limit_min: float, limit_max: float) -> np.ndarray:
    """
    Normalize data to the range [0, 255].
    """
    return np.clip((x - limit_min) / (limit_max - limit_min), 0, 1) * 255


def export_head_uint8(yolo_pth, output_pth, imgsz, images, quantized_output=False):
    os.makedirs(output_pth, exist_ok=True)

    inferer = TFLiteInferer(yolo_pth)
    y = []
    for i in inferer.yolo_output_details:
        y.append(np.zeros(i["shape"], dtype=np.float32))

    head = DetectionHeadSplitted([i.shape for i in y], imgsz)
    head([torch.Tensor(i) for i in y])

    np_data_examples = [[] for i in range(len(y))]
    for img in images:
        output = inferer.infer_no_head(img, quantized_output=quantized_output)
        for idx, i in enumerate(output):
            np_data_examples[idx].append(i)

    if quantized_output:
        suffix = "_quantized"
    else:
        suffix = ""

    np_data = []
    for idx, i in enumerate(np_data_examples):
        name = f"input_{idx}"
        tmp_file = os.path.join(output_pth, f"{name}{suffix}.npy")
        tmp_norm_file = os.path.join(output_pth, f"{name}_norm{suffix}.npy")
        mean_file = os.path.join(output_pth, f"{name}_mean{suffix}.npy")
        std_file = os.path.join(output_pth, f"{name}_std{suffix}.npy")
        data = np.concatenate(i, axis=0).astype(np.float32)

        val_min = data.min(axis=(0, 1))
        val_max = data.max(axis=(0, 1))

        data_norm = norm_data(data, val_min, val_max)
        mean = 255 * np.abs(val_min) / (val_max - val_min)
        std = 255 / (val_max - val_min)

        np.save(str(tmp_file), data)
        np.save(str(tmp_norm_file), data_norm)
        np.save(str(mean_file), mean)
        np.save(str(std_file), std)

        if quantized_output:
            np_data.append([
                name,
                tmp_norm_file,
                mean[None, None],
                std[None, None],
            ])
        else:
            np_data.append([
                name,
                tmp_norm_file,
                mean[None, None],
                std[None, None],
            ])

    detection_head2tflite_uint8(output_pth, head, y, np_data)


def get_images(data_pth, max_ncalib_imgs):
    base_pth = os.path.dirname(data_pth)
    images_pth = os.path.join(base_pth, "images/val")
    img_pths = glob.glob(os.path.join(images_pth, "*.jpg"))[:max_ncalib_imgs]
    imgs = [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in img_pths]

    return imgs


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

    imgs = get_images(args.data, args.max_ncalib_imgs)

    base_pth = os.path.dirname(args.weights)
    tflite_pth = os.path.join(base_pth, "best_saved_model/best_full_integer_quant.tflite")
    head_pth = os.path.join(base_pth, "head_saved_model")
    head_separated_pth = os.path.join(base_pth, "head_sep_saved_model")
    head_uint8_pth = os.path.join(base_pth, "head_uint8_saved_model")

    export_head(tflite_pth, head_pth, args.imgsz)
    export_head_separated(tflite_pth, head_separated_pth, args.imgsz)
    # export_head_uint8(tflite_pth, head_uint8_norm_pth, args.imgsz, imgs, True)
    export_head_uint8(tflite_pth, head_uint8_pth, args.imgsz, imgs, False)
