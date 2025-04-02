import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


def np2torch(x):
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def torch2np(x):
    return (x.squeeze(0).permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)


class InputAdapter(nn.Module):
    def __init__(self):
        super(InputAdapter, self).__init__()
        
        # 1x1 Conv layer to apply fixed grayscale conversion
        self.rgb2gray = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        gray_weights = torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]])  # Shape: (1, 3, 1, 1)
        self.rgb2gray.weight = nn.Parameter(gray_weights, requires_grad=False)

        # 3x3 Conv layer for Gaussian blurring
        self.gaussian_blur = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        gaussian_weights = torch.tensor(
            [[[[0.0625, 0.125, 0.0625],
               [0.125,  0.25,  0.125],
               [0.0625, 0.125, 0.0625]]]]
        )  # Shape: (1, 1, 3, 3)
        self.gaussian_blur.weight = nn.Parameter(gaussian_weights, requires_grad=False)

        # 1x1 Conv layer to duplicate grayscale into 3-channel RGB
        self.gray2rgb = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        duplicate_weights = torch.tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]]])  # Shape: (3, 1, 1, 1)
        self.gray2rgb.weight = nn.Parameter(duplicate_weights, requires_grad=False)

    def forward(self, x):
        x = self.rgb2gray(x)  # Convert RGB to grayscale
        x = self.gaussian_blur(x)  # Apply Gaussian blur
        x = self.gray2rgb(x)  # Convert grayscale to 3-channel RGB
        return x


def model2onnx(file_path: str, model: nn.Module, y: np.ndarray, simplify=False):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    torch.onnx.export(
        model,
        torch.Tensor(y),
        file_path,
        input_names=["input_0"],
        output_names=["output_0"],
    )

    if simplify:
        import onnx
        import onnxslim

        model_onnx = onnx.load(file_path)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, file_path)


def imgs2calib_data_pth(data_pth: str, calib_pth: str, imgs: np.ndarray, max_ncalib_imgs: int = 100) -> str:
    img_pths = glob.glob(os.path.join(data_pth, "images", "val", "*.jpg"))

    imgs = []
    n = 0
    for img_pth in img_pths:
        img = cv2.imread(img_pth)
        img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        n += 1
        if n >= max_ncalib_imgs:
            break
    imgs = np.stack(imgs, axis=0).astype(np.float32)

    np.save(calib_pth, imgs)


def model2tflite(folder_path: str, model: nn.Module, y: np.ndarray, data_pth: str, max_ncalib_imgs: int = 100):
    import onnx2tf

    f = Path(folder_path)
    os.makedirs(folder_path, exist_ok=True)

    onnx_model_path = str(f / "input_adapter.onnx")
    model2onnx(onnx_model_path, model, y, simplify=True)

    calib_data_pth = f / "calib_data.npy"
    imgs2calib_data_pth(data_pth, str(calib_data_pth), y, max_ncalib_imgs)
    onnx2tf.convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=str(f),
        not_use_onnxsim=True,
        verbosity=True,
        output_integer_quantized_tflite=True,
        quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
        custom_input_op_name_np_data_path=[["input_0", str(calib_data_pth), [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]],
        input_output_quant_dtype="uint8",
        disable_group_convolution=True,  # for end-to-end model compatibility
        enable_batchmatmul_unfold=True,  # for end-to-end model compatibility
    )

    # Remove/rename TFLite models
    calib_data_pth.unlink(missing_ok=True)
    for file in f.rglob("*_dynamic_range_quant.tflite"):
        file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix))
    for file in f.rglob("*_integer_quant_with_int16_act.tflite"):
        file.unlink()  # delete extra fp16 activation TFLite files
