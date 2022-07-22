from torch import nn
import torch

class BatchNormDimSwap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if len(x.shape)==3:
            x = x.permute(0,2,1)
        return x
