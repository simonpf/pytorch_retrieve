"""
pytorch_retrieve.modules.conv.heads
===================================

Provides head modules for CNNs.
"""
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import nn

from pytorch_retrieve.modules.conv import blocks


class BasicConv(nn.Sequential):
    """
    A generic head consisting of sequenctial convolution block with
    activation and normalization layers and a final 1x1 convolution
    to map the output to the expected shape.
    """

    def __init__(
        self,
        in_channels: int,
        out_shape: Union[List[int], Tuple[int], int],
        depth: int = 1,
        kernel_size: int = 1,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        residual_connections: bool = True,
    ):
        if isinstance(out_shape, int):
            out_shape = (out_shape,)

        super().__init__()
        padding = kernel_size // 2
        head_blocks = [
            blocks.BasicConvBlock(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                activation_factory=activation_factory,
                normalization_factory=normalization_factory,
                residual_connection=residual_connections,
            )
        ]
        head_blocks.append(nn.Conv2d(in_channels, np.prod(out_shape), kernel_size=1))
        super().__init__(*head_blocks)
        self.out_shape = tuple(out_shape)

    def forward(self, x: torch.Tensor):
        """
        Forward tensor through head and shape into expected shape.
        """
        y = super().forward(x)
        curr_shape = y.shape
        new_shape = y.shape[:1] + self.out_shape + y.shape[2:]
        y = y.view(new_shape)
        return y
