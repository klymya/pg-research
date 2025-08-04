import argparse
import os
import glob

import cv2
import onnx
import numpy as np
import torch
import onnx2tf

from ultralytics_yolov8.ultralytics import YOLO
from detection_head import detection_head2onnx, DetectionHeadSplitted
from tflite_inference import TFLiteInferer
from merger import merge_yolo


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--max_ncalib_imgs', type=int, default=100, help='calibration image set size')
    parser.add_argument('--output-folder-name', type=str, default='best_saved_model', help='output folder name')

    return parser.parse_args()


def export_head(yolo_pth, output_pth, imgsz):
    yolo_model = onnx.load(yolo_pth)

    y = []
    for output in yolo_model.graph.output:
        shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        y.append(np.zeros(shape, dtype=np.float32))

    head = DetectionHeadSplitted([i.shape for i in y], imgsz)
    head([torch.Tensor(i) for i in y])

    detection_head2onnx(output_pth, head, y, True, True)


def get_input_data(data_pth: str, max_ncalib_imgs: int, imgsz: int) -> np.ndarray:
    """
    Load input data from the specified path and return it as a numpy array.
    """
    base_pth = os.path.dirname(data_pth)
    images_pth = os.path.join(base_pth, "images/val")
    img_pths = glob.glob(os.path.join(images_pth, "*.jpg"))[:max_ncalib_imgs]
    imgs = [cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in img_pths]

    inferer = TFLiteInferer(imgsz=imgsz)
    img_processed = [inferer.pre_process(img) for img in imgs]

    return np.array(img_processed)


def export_combined_model(weights: str, output_folder: str, data_pth: str, imgsz: int, max_ncalib_imgs: int):
    os.makedirs(output_folder, exist_ok=True)

    images = get_input_data(data_pth, max_ncalib_imgs, imgsz)
    tmp_file = "calibration_images.npy"

    np.save(tmp_file, images.astype(np.float32))  # BHWC
    np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]

    onnx2tf.convert(
        input_onnx_file_path=weights,
        output_folder_path=output_folder,
        not_use_onnxsim=True,
        verbosity="info",
        output_integer_quantized_tflite=True,
        quant_type="per-channel",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
        custom_input_op_name_np_data_path=np_data,
        input_output_quant_dtype="uint8",
        disable_group_convolution=True,  # for end-to-end model compatibility
        enable_batchmatmul_unfold=True,  # for end-to-end model compatibility
    )

    os.remove(tmp_file)  # Clean up temporary file


if __name__ == '__main__':

    args = parser_arguments()
    # Create a dictionary of kwargs
    kwargs = vars(args)
    print(kwargs)

    model = YOLO(args.weights)

    success = model.export(
        format="onnx",
        simplify=True,
        imgsz=args.imgsz,
        data=args.data,
        int8=False,
        separate_outputs=True,
        export_hw_optimized=True,
        max_ncalib_imgs=args.max_ncalib_imgs
    )

    base_pth = os.path.dirname(args.weights)
    model_onnx_pth = args.weights.replace(".pt", ".onnx")
    head_pth = os.path.join(base_pth, "head.onnx")

    export_head(model_onnx_pth, head_pth, args.imgsz)

    merge_yolo(model_onnx_pth, head_pth, model_onnx_pth)

    export_combined_model(
        weights=model_onnx_pth,
        output_folder=os.path.join(base_pth, args.output_folder_name),
        data_pth=args.data,
        imgsz=args.imgsz,
        max_ncalib_imgs=args.max_ncalib_imgs
    )
