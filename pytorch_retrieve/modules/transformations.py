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


class MinMax(nn.Module):
    """
    Min-max transformation.

    Linearly map the range -1, 1 to the range [min, max]
    """
    def __init__(self, x_min: float, x_max: float):
        super().__init__()
        self.x_min = torch.tensor(x_min)
        self.x_max = torch.tensor(x_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        x_min = self.x_min.to(device=device, dtype=dtype)
        x_max = self.x_max.to(device=device, dtype=dtype)
        return -1.0 + 2.0 * (x - x_min) / (x_max - x_min)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        x_min = self.x_min.to(device=device, dtype=dtype)
        x_max = self.x_max.to(device=device, dtype=dtype)
        return x_min + (x_max - x_min) * (0.5 * x + 0.5)
