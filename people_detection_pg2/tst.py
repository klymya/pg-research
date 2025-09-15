import torch
import torch.nn as nn
import numpy as np
import onnx

from detection_head import DetectionHeadSplitted


N = 16
M = 4
SHAPE = (1, 2100, 64)

class Model0(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        b, c, a = x.shape  # batch, channels, anchors
        return x.view(b, M, N, a).transpose(2, 1)


class Model1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        b, c, a = x.shape  # batch, channels, anchors
        return x.view(b, N, M, a)


class Model3(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))  # This is 3D, so permute is allowed
        b, c, a = x.shape  # batch, channels, anchors
        
        # Model0 does: x.view(b, M, N, a).transpose(2, 1) 
        # To avoid transpose on 4D tensor, we use stack/unstack approach
        
        x_reshaped = x.view(b, M, N, a)  # (b, M, N, a)
        
        # Split along M dimension, then stack along N dimension
        # This achieves the same result as transpose(2, 1) without using transpose on 4D
        slices = torch.unbind(x_reshaped, dim=1)  # Split into M tensors of shape (b, N, a)
        result = torch.stack(slices, dim=2)  # Stack along dim 2 to get (b, N, M, a)
        return result


x = torch.randn(SHAPE)

y0 = Model0()(x)
y1 = Model1()(x)
y3 = Model3()(x)

assert torch.allclose(y0, y3, atol=1e-6)
print("Model3 successfully produces the same result as Model0!")
