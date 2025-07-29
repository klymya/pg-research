import math
import os

import onnx
import torch
import torch.nn as nn
import numpy as np

from ultralytics_yolov8.ultralytics.utils.tal import make_anchors


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16, num_classes=1):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
        self.sigm = nn.Sigmoid()
        self.num_classes = num_classes

    def forward(self, x0, x1, x2, x3, x4, x5):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        x = torch.permute(torch.cat([x0, x1, x2], 1), (0, 2, 1))
        y = torch.permute(torch.cat([x3, x4, x5], 1), (0, 2, 1))

        b, c, a = x.shape  # batch, channels, anchors
        out = self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        return out, self.sigm(y)


def dist2bbox_lt_wh(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    wh = x2y2 - x1y1
    return torch.cat((x1y1, wh), dim)


class DetectionHead(nn.Module):
    def __init__(self, shapes, img_size=256, num_classes=1):
        super().__init__()

        # self.box_idxs = [1, 0, 5, 4, 6, 2]
        # self.kp_idx = 3

        mcv = float('-inf')
        self.kp_idx = -1
        for idx, s in enumerate(shapes):
            dim_1 = s[1]
            if dim_1 > mcv:
                mcv = dim_1
                self.kp_idx = idx

        self.box_idxs = [
            i for i, _ in sorted(enumerate(shapes), key=lambda x: (x[1][2] if num_classes > 64 else -x[1][2], -x[1][1]))
        ]
        self.box_idxs.remove(self.kp_idx)

        strides = [
            int(math.sqrt(img_size * img_size / shapes[p][1])) for p in self.box_idxs if shapes[p][2] != 64]
        dims = [(img_size // s, img_size // s) for s in strides]
        fake_feats = [torch.zeros((1, 1, h, w)) for h, w in dims]
        self.anchors, self.strides = (x.transpose(0, 1)
                        for x in make_anchors(fake_feats, strides, 0.5))  # generate anchors and strides

        self.kpt_shape = (shapes[self.kp_idx][-1] // 2, 2)
        reg_max = 16
        self.num_classes = num_classes
        self.dfl = DFL(reg_max, self.num_classes)

    def forward(self, x):
        dbox, y = self.dfl(*[x[i] for i in self.box_idxs])

        dbox = dist2bbox_lt_wh(dbox, self.anchors.unsqueeze(0), xywh=False, dim=1) * self.strides
        dbox = torch.cat((dbox, y), 1)

        kps = torch.permute(x[self.kp_idx], (0, 2, 1))
        kps = (kps * 2.0 + (self.anchors - 0.5)) * self.strides
        # kps[:, 0] = (kps[:, 0] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        # kps[:, 1] = (kps[:, 1] * 2.0 + (self.anchors[1] - 0.5)) * self.strides

        return torch.permute(torch.cat([dbox, kps], 1), (0, 2, 1))


class DetectionHeadSplitted(nn.Module):
    def __init__(self, shapes, img_size=256, num_classes=1):
        super().__init__()

        mcv = float('-inf')
        self.kp_idx = -1
        for idx, s in enumerate(shapes):
            dim_1 = s[1]
            if dim_1 > mcv:
                mcv = dim_1
                self.kp_idx = idx

        self.box_idxs = [
            i for i, _ in sorted(enumerate(shapes), key=lambda x: (x[1][2] if num_classes > 64 else -x[1][2], -x[1][1]))
        ]
        self.box_idxs.remove(self.kp_idx)

        strides = [
            int(math.sqrt(img_size * img_size / shapes[p][1])) for p in self.box_idxs if shapes[p][2] != 64]
        dims = [(img_size // s, img_size // s) for s in strides]
        fake_feats = [torch.zeros((1, 1, h, w)) for h, w in dims]
        self.anchors, self.strides = (x.transpose(0, 1)
                        for x in make_anchors(fake_feats, strides, 0.5))  # generate anchors and strides

        self.kpt_shape = (shapes[self.kp_idx][-1] // 2, 2)
        reg_max = 16
        self.num_classes = num_classes
        self.dfl = DFL(reg_max, self.num_classes)

    def forward(self, x):
        dbox, y = self.dfl(*[x[i] for i in self.box_idxs])

        dbox = dist2bbox_lt_wh(dbox, self.anchors.unsqueeze(0), xywh=False, dim=1) * self.strides

        kps = torch.permute(x[self.kp_idx], (0, 2, 1))
        kps = (kps * 2.0 + (self.anchors - 0.5)) * self.strides
        # kps[:, 0] = (kps[:, 0] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
        # kps[:, 1] = (kps[:, 1] * 2.0 + (self.anchors[1] - 0.5)) * self.strides

        return torch.permute(torch.cat([dbox, kps], 1), (0, 2, 1)), torch.permute(y, (0, 2, 1))


def detection_head2onnx(file_path: str, head: nn.Module, y: list[np.ndarray], simplify=False, separate_outputs=False):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if separate_outputs:
        torch.onnx.export(
            head,
            [torch.Tensor(i) for i in y],
            file_path,
            input_names=[f"input_{i}" for i in range(7)],
            output_names=["bboxes", "scores"]
        )
    else:
        torch.onnx.export(
            head,
            [torch.Tensor(i) for i in y],
            file_path,
            input_names=[f"input_{i}" for i in range(7)]
        )

    if simplify:
        import onnxslim

        model_onnx = onnx.load(file_path)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, file_path)


def detection_head2tflite(folder_path: str, head: nn.Module, y: list[np.ndarray]):
    import onnx2tf

    os.makedirs(folder_path, exist_ok=True)

    onnx_model_path = os.path.join(folder_path, "head.onnx")
    detection_head2onnx(onnx_model_path, head, y, simplify=True)

    onnx2tf.convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=str(folder_path),
        not_use_onnxsim=True,
        verbosity=True,
        keep_ncw_or_nchw_or_ncdhw_input_names=[f"input_{i}" for i in range(7)]
    )
    
def get_skip_nchw_input_indexes(inputs: list[np.ndarray]) -> list[int]:
    """
    Returns indexes of input arrays that should be skipped from
    keep_ncw_or_nchw_or_ncdhw_input_names in onnx2tf conversion.

    Parameters:
    - inputs: List of np.ndarray inputs, each with shape [batch_size, ...]

    Returns:
    - List[int]: indexes to skip (not suitable for NCHW preservation)
    """
    skip_indexes = []
    for idx, arr in enumerate(inputs):
        # Remove batch dimension
        shape = arr.shape[1:]
        if not (3 <= len(shape) <= 5):
            skip_indexes.append(idx)
    return skip_indexes


def detection_head2tflite_uint8(folder_path: str, head: nn.Module, y: list[np.ndarray], np_data: list):
    import onnx2tf

    os.makedirs(folder_path, exist_ok=True)

    onnx_model_path = os.path.join(folder_path, "head.onnx")
    detection_head2onnx(onnx_model_path, head, y, simplify=True, separate_outputs=True)

    onnx2tf.convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=str(folder_path),
        not_use_onnxsim=True,
        verbosity=True,
        keep_ncw_or_nchw_or_ncdhw_input_names=[f"input_{i}" for i in range(7)],
        output_integer_quantized_tflite=True,
        input_output_quant_dtype="uint8",
        quant_type="per-tensor",
        custom_input_op_name_np_data_path=np_data,
        disable_group_convolution=True,  # for end-to-end model compatibility
        enable_batchmatmul_unfold=True,  # for end-to-end model compatibility
    )
