"""
pytorch_retrieve.modules
========================

The 'pytorch_retrieve.modules' sub-module defines the custom Pytorch
modules provides by pytorch_retrieve.
"""
from torch import nn


class Mean(nn.Module):
    def forward(x):
        return MeanTensor(x)
