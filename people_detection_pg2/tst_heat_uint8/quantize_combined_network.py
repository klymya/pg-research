import os
import glob
import sys
import argparse

import cv2
import numpy as np
import onnx2tf

sys.path.append("..")
from tflite_inference import TFLiteInferer


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


def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--max_ncalib_imgs', type=int, default=100, help='calibration image set size')

    return parser.parse_args()


def main():
    args = parser_arguments()

    images = get_input_data(args.data, args.max_ncalib_imgs, args.imgsz)
    tmp_file = "calibration_images.npy"

    np.save(tmp_file, images.astype(np.float32))  # BHWC
    np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]

    f = args.weights.replace(".onnx", "")
    f = f + "_per_channel"
    os.makedirs(f, exist_ok=True)

    onnx2tf.convert(
        input_onnx_file_path=args.weights,
        output_folder_path=f,
        not_use_onnxsim=True,
        verbosity="info",
        output_integer_quantized_tflite=True,
        # quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
        quant_type="per-channel",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
        custom_input_op_name_np_data_path=np_data,
        input_output_quant_dtype="uint8",
        disable_group_convolution=True,  # for end-to-end model compatibility
        enable_batchmatmul_unfold=True,  # for end-to-end model compatibility
    )


if __name__ == "__main__":
    main()
