import os

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

    def forward(self, x):
        x = self.rgb2gray(x)  # Convert RGB to grayscale
        x = self.gaussian_blur(x)  # Apply Gaussian blur
        x = x.repeat(1, 3, 1, 1)  # Duplicate the grayscale channel to create an RGB image
        return x


def model2onnx(file_path: str, model: nn.Module, y: np.ndarray, simplify=False):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    torch.onnx.export(
        model,
        [torch.Tensor(y)],
        file_path,
        input_names=["input_0"]
    )

    if simplify:
        import onnx
        import onnxslim

        model_onnx = onnx.load(file_path)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, file_path)


def model2tflite(folder_path: str, model: nn.Module, y: np.ndarray):
    import onnx2tf

    os.makedirs(folder_path, exist_ok=True)

    onnx_model_path = os.path.join(folder_path, "input_adapter.onnx")
    model2onnx(onnx_model_path, model, y, simplify=True)

    onnx2tf.convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=str(folder_path),
        not_use_onnxsim=True,
        verbosity=True,
    )