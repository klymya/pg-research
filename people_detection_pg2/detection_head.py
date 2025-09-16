import math
import os

import onnx
import torch
import torch.nn as nn
import numpy as np

from ultralytics_yolov8.ultralytics.utils.tal import make_anchors
from ultralytics_yolov8.ultralytics.utils.torch_utils import TORCH_1_13, get_latest_opset


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


class DFL_FC_NoSlice_NoDiv(nn.Module):
    """
    DFL integral without gather/slice and without explicit division.
    Uses Softmax along the last dim (16) and a single Linear ([0..15]) dot-product.
    Shapes assumed right after your two lines:
      x: (B, 64, A)   # 64 == 4*c1 with c1=16
      y: (B,  1, A)
    Returns:
      out: (B, 4, A)              # distances per side (matches Ultralytics)
      y  : (B, A, num_classes)    # same as your original (sigmoid applied)
    """
    def __init__(self, c1=16, num_classes=1):
        super().__init__()
        self.c1 = c1
        self.num_classes = num_classes

        # FullyConnected that computes dot with [0..c1-1] on the LAST axis (size c1)
        self.fc = nn.Linear(c1, 1, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.arange(c1, dtype=torch.float32).view(1, c1))
        self.fc.requires_grad_(False)

        self.sigm = nn.Sigmoid()

    def forward(self, x0, x1, x2, x3, x4, x5):
        # Your original prep:
        x = torch.permute(torch.cat([x0, x1, x2], dim=1), (0, 2, 1))   # (B, 64, A)
        y = torch.permute(torch.cat([x3, x4, x5], dim=1), (0, 2, 1))   # (B,  1, A)

        B, C, A = x.shape
        assert C == 4 * self.c1, f"Expected channels=4*c1={4*self.c1}, got {C}"

        # Put anchors before channels (3D transpose only, which embedded NPUs allow)
        x = x.permute(0, 2, 1).contiguous()          # (B, A, 64)

        # Pack (4 sides × 16 bins) into the last dim, with bins LAST for TFLite Softmax:
        # (B, A, 64) -> (B, A*4, 16)
        x = x.view(B, A * 4, self.c1)                # no slice/gather, just reshape

        # Softmax over bins (last dim = 16)
        p = torch.softmax(x, dim=-1)                 # (B, A*4, 16)

        # Dot with [0..15] using a single FC (no explicit sum/div)
        e = self.fc(p).squeeze(-1)                   # (B, A*4)

        # Restore sides and anchors
        out = e.view(B, A, 4).permute(0, 2, 1).contiguous()   # (B, 4, A)

        # Class/conf branch: keep your original layout (B, A, num_classes)
        y = self.sigm(y.permute(0, 2, 1).contiguous())        # (B, A, num_classes)

        return out, y


class DFL_GroupedNoSlice(nn.Module):
    """
    DFL integral via grouped 1x1 convs (no slice/gather, no 4D transpose, no explicit sum).
    Assumes (right after your two lines):
      x: (B, 64, A)  == (B, 4*c1, A) with c1=16
      y: (B,  1, A)  == (B, num_classes, A) with num_classes=1
    Returns:
      out: (B, 4, A)               # distances per side
      y  : (B, A, num_classes)     # classes/conf after sigmoid (same as your original DFL)
    """
    def __init__(self, c1=16, num_classes=1, eps=1e-9):
        super().__init__()
        self.c1 = c1
        self.num_classes = num_classes
        self.eps = eps

        in_ch = 4 * c1
        groups = 4

        # Conv to compute Σ i * exp(logit_i) per group (i=0..c1-1)
        self.conv_num = nn.Conv1d(in_channels=in_ch, out_channels=groups,
                                  kernel_size=1, groups=groups, bias=False)
        # Conv to compute Σ exp(logit_i) per group
        self.conv_den = nn.Conv1d(in_channels=in_ch, out_channels=groups,
                                  kernel_size=1, groups=groups, bias=False)

        with torch.no_grad():
            # Each group sees its own 16 input channels
            w_num = torch.zeros(groups, c1, 1, dtype=torch.float32)
            w_den = torch.ones(groups, c1, 1, dtype=torch.float32)
            idx = torch.arange(c1, dtype=torch.float32)  # [0,1,...,15]
            for g in range(groups):
                w_num[g, :, 0] = idx
            self.conv_num.weight.copy_(w_num)
            self.conv_den.weight.copy_(w_den)
        self.conv_num.requires_grad_(False)
        self.conv_den.requires_grad_(False)

        self.sigm = nn.Sigmoid()

    def forward(self, x0, x1, x2, x3, x4, x5):
        # Your exact pre-DFL ops:
        x = torch.permute(torch.cat([x0, x1, x2], dim=1), (0, 2, 1))   # (B, 64, A)
        y = torch.permute(torch.cat([x3, x4, x5], dim=1), (0, 2, 1))   # (B,  1, A)

        B, C, A = x.shape
        assert C == 4 * self.c1, f"Expected channels={4*self.c1}, got {C}"

        # Softmax via normalized exp using group-wise 1x1 convs (no slicing)
        ex = torch.exp(x)                              # (B, 64, A)
        num = self.conv_num(ex)                        # (B, 4, A)  -> Σ i*e^{logit_i}
        den = self.conv_den(ex)                        # (B, 4, A)  -> Σ e^{logit_i}
        out = num / (den + self.eps)                   # (B, 4, A)  expectation per side

        # Class/conf branch: same as your original — return (B, A, num_classes) after sigmoid
        y = self.sigm(y.permute(0, 2, 1).contiguous()) # (B, A, num_classes)

        return out, y


class DFL_NoTranspose(nn.Module):
    def __init__(self, c1=16, num_classes=1):
        super().__init__()
        self.c1 = c1
        self.num_classes = num_classes
        self.register_buffer("bins", torch.arange(c1, dtype=torch.float32).view(1,1,1,c1))
        self.sigm = nn.Sigmoid()

    def forward(self, x0, x1, x2, x3, x4, x5):
        # ---- distances (DFL) ----
        z = torch.cat([x0, x1, x2], dim=1)  # either (B, 4*c1, A) or (B, A, 4*c1)
        four_c1 = 4 * self.c1
        B = z.shape[0]

        if z.dim() != 3:
            raise RuntimeError(f"Expected 3D tensor for z, got {z.shape}")

        if z.shape[1] == four_c1:             # (B, 4*c1, A) -> (B, A, 4*c1)
            _, _, A = z.shape
            z = z.permute(0, 2, 1).contiguous()
        elif z.shape[2] == four_c1:           # already (B, A, 4*c1)
            _, A, _ = z.shape
        else:
            raise RuntimeError(f"Cannot find 4*c1={four_c1} on axis 1 or 2 for z.shape={z.shape}")

        # Now z = (B, A, 4*c1). Reshape to (B, A, 4, c1), softmax over bins, weighted sum.
        z = z.reshape(B, A, 4, self.c1)
        p = torch.softmax(z, dim=-1)                  # (B, A, 4, c1)
        out = (p * self.bins).sum(dim=-1)             # (B, A, 4)  expectation over bins
        out = out.permute(0, 2, 1).contiguous()       # (B, 4, A)  — matches Ultralytics head

        # ---- classes/conf ----
        y = torch.cat([x3, x4, x5], dim=1)            # often (B, Ccls, A) or (B, A, Ccls)
        # Normalize to (B, A, Ccls) then apply sigmoid
        if y.dim() == 3 and y.shape[1] <= y.shape[2]:
            # likely (B, A, Ccls) already
            y = self.sigm(y)
        else:
            # likely (B, Ccls, A)
            y = self.sigm(y.permute(0, 2, 1).contiguous())
        return out, y


class DFL_PerSide(nn.Module):
    """
    DFL integral computed per side (L,T,R,B) to avoid 4D transpose and explicit sum.
    Input assumptions (right after your two lines):
      x: (B, 64, A)  == (B, 4*c1, A) with c1=16
      y: (B,  1, A)  == (B, num_classes, A) with num_classes=1
    Returns:
      out: (B, 4, A)              # distances per side
      y  : (B, A, num_classes)    # classes/conf after sigmoid
    """
    def __init__(self, c1=16, num_classes=1):
        super().__init__()
        self.c1 = c1
        self.num_classes = num_classes

        # 1x1 Conv1d to implement dot([0..c1-1]) along the bin/channel axis
        self.side_conv = nn.Conv1d(c1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            w = torch.arange(c1, dtype=torch.float32).view(1, c1, 1)  # (out=1, in=c1, k=1)
            self.side_conv.weight.copy_(w)
        self.side_conv.requires_grad_(False)

        self.sigm = nn.Sigmoid()

    def forward(self, x0, x1, x2, x3, x4, x5):
        # ----- Your exact pre-DFL ops -----
        x = torch.permute(torch.cat([x0, x1, x2], dim=1), (0, 2, 1))   # expected (B, 64, A)
        y = torch.permute(torch.cat([x3, x4, x5], dim=1), (0, 2, 1))   # expected (B, 1, A)

        B, C, A = x.shape
        four_c1 = 4 * self.c1
        assert C == four_c1, f"Expected x.shape[1]==4*c1={four_c1}, got {C}"

        # ----- Reshape to 4 sides of 16 bins each: (B, 4, c1, A) -----
        z = x.reshape(B, 4, self.c1, A)  # no 4D transpose introduced

        # ----- Per-side: softmax over bins (dim=1 in Conv1d input), then 1x1 Conv1d -----
        outs = []
        for i in range(4):
            side = z[:, i, :, :]                      # (B, c1, A)
            side = torch.softmax(side, dim=1)         # softmax over bins
            val  = self.side_conv(side)               # (B, 1, A) = E[bins]
            outs.append(val)

        out = torch.cat(outs, dim=1)                  # (B, 4, A)  distances (matches Ultralytics layout)

        # ----- Classes/conf → (B, A, num_classes) after sigmoid -----
        # You reported y as (B, 1, A). Convert to (B, A, 1) and apply sigmoid.
        if y.dim() == 3 and y.shape[1] == self.num_classes and y.shape[2] == A:
            y = y.permute(0, 2, 1).contiguous()       # (B, A, num_classes)
        elif y.dim() == 3 and y.shape[1] == A and y.shape[2] == self.num_classes:
            # already (B, A, num_classes)
            pass
        else:
            # Fallback: ensure (B, A, num_classes)
            y = y.permute(0, 2, 1).contiguous() if y.shape[2] == A else y

        y = self.sigm(y)                               # (B, A, num_classes)

        return out, y


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
        self.dfl = DFL_FC_NoSlice_NoDiv(reg_max, self.num_classes)

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
        self.dfl = DFL_FC_NoSlice_NoDiv(reg_max, self.num_classes)

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
            output_names=["bboxes", "scores"],
            opset_version=get_latest_opset(),
        )
    else:
        torch.onnx.export(
            head,
            [torch.Tensor(i) for i in y],
            file_path,
            input_names=[f"input_{i}" for i in range(7)],
            opset_version=get_latest_opset(),
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


def detection_head2tflite_separated(folder_path: str, head: nn.Module, y: list[np.ndarray]):
    import onnx2tf

    os.makedirs(folder_path, exist_ok=True)

    onnx_model_path = os.path.join(folder_path, "head.onnx")
    detection_head2onnx(onnx_model_path, head, y, simplify=True, separate_outputs=True)

    onnx2tf.convert(
        input_onnx_file_path=onnx_model_path,
        output_folder_path=str(folder_path),
        not_use_onnxsim=True,
        verbosity=True,
        keep_ncw_or_nchw_or_ncdhw_input_names=[f"input_{i}" for i in range(7)]
    )



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
