"""
pytorch_retrieve.modules.conv.padding
=====================================

Defines padding factories for creating specific padding layers.
"""
from typing import Tuple, Union
import torch
from torch import nn


def calculate_padding(
        kernel_size: Union[int, Tuple[int]],
        dilation: Union[int, Tuple[int]] = 1
) -> Tuple[int]:
    """
    Calculate padding for a kernel filter with given kernel size and dilation.

    Args:
        kernel_size: The filters kernel isze.
        dilations: The dilation of the kernel.

    Return:
        Tuple specifying the padding to apply along each of the n-last dimensions,
        where n is determined from the length of 'kernel_size' or set to 2 if
        'kernel_size' is an integer.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2
        n_dim = 2
    else:
        n_dim = len(kernel_size)


    if isinstance(dilation, int):
        dilation = (dilation,) * n_dim
    else:
        assert len(dilation) == n_dim


    padding = tuple([
        (s_k - 1) * dil // 2 for s_k, dil in zip(kernel_size, dilation)
    ])

    return padding


class Zero(nn.Module):
    """
    Pad input by padding zeros.
    """
    def __init__(self, pad: Union[int, Tuple[int]]):
        """
        Instantiates a padding layer.

        Args:
            pad: N-tuple defining the padding added to the n-last dimensions
                of the tensor. If an int, the same padding will be added to the
                two last dimensions of the tensor.
        """
        super().__init__()
        if isinstance(pad, int):
            pad = (pad,) * 2

        full_pad = []
        for n_elems in pad:
            full_pad += [n_elems, n_elems]

        full_pad = tuple(full_pad[::-1])
        self.pad = full_pad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add padding to tensor.

        Args:
            x: The input tensor.

        Return:
            The padded tensor.
        """
        return nn.functional.pad(x, self.pad, "constant", value=0)


class Reflect(nn.Module):
    """
    Pad input by reflecting the input tensor.
    """
    def __init__(self, pad: Union[int, Tuple[int]]):
        """
        Instantiates a padding layer.

        Args:
            pad: N-tuple defining the padding added to the n-last dimensions
                of the tensor. If an int, the same padding will be added to the
                two last dimensions of the tensor.
        """
        super().__init__()
        if isinstance(pad, int):
            pad = (pad,) * 2

        full_pad = []
        for n_elems in pad:
            full_pad += [n_elems, n_elems]

        full_pad = tuple(full_pad[::-1])
        self.pad = full_pad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add padding to tensor.

        Args:
            x: The input tensor.

        Return:
            The padded tensor.
        """
        return nn.functional.pad(x, self.pad, "reflect")


class Global(nn.Module):
    """
    Padding for global lat/lon grids that performs circular padding
    along the last dimension and reflect padding along the second-to-last
    dimensions.
    """
    def __init__(self, pad: Union[int, Tuple[int]]):
        """
        Instantiates a padding layer.

        Args:
            pad: N-tuple defining the padding added to the n-last dimensions
                of the tensor. If an int, the same padding will be added to the
                two last dimensions of the tensor.
        """
        super().__init__()
        if isinstance(pad, int):
            pad = (pad,) * 2

        full_pad = []
        for n_elems in pad:
            full_pad += [n_elems, n_elems]

        full_pad = tuple(full_pad[::-1])
        self.pad = full_pad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add padding to tensor.

        Args:
            x: The input tensor.

        Return:
            The padded tensor.
        """
        x_1 = nn.functional.pad(x, self.pad[:2] + (0, 0), "circular")
        return nn.functional.pad(x_1, (0, 0) + self.pad[2:], "reflect")
