"""
pytorch_retrieve.modules.transformations
========================================

Defines transfomations modules to be used in output modules.
"""

import torch
from torch import nn


class SquareRoot(nn.Module):
    """
    SquareRoot transformation.

    This transformation can be used in an output layer to predict the square root of the
    target quantity instead of the original quantity.
    """
    def forward(self, x):
        return torch.sqrt(x)

    def invert(self, x: torch.Tensor):
        return x ** 2


class Log(nn.Module):
    """
    Log transformation.

    This transformation can be used in an output layer to predict the logarithm of the
    target quantity instead of the original quantity.
    """
    def forward(self, x):
        return torch.log(x)

    def invert(self, x: torch.Tensor):
        return torch.exp(x)


class LogLinear(nn.Module):
    """
    Log transformation.

    This transformation can be used in an output layer to predict the logarithm of the
    target quantity instead of the original quantity.
    """
    def forward(self, x):
        return torch.where(x > 1, x - 1, torch.log(x))

    def invert(self, x: torch.Tensor):
        return torch.where(x > 0, x + 1, torch.exp(x))
