"""
pytorch_retrieve.modules.conv_blocks
====================================

This module provide convolution block factories for reproducing various
CNN architectures.
"""
from typing import Callable, Optional

import torch
from torch import nn

from pytorch_retrieve.modules.utils import ParamCount


class BasicConvBlock(nn.Module, ParamCount):
    """
    Implements a basic convolution block with an optional residual
    connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        residual_connection: bool = False,
        downsample: Optional[int] = None,
    ):
        super().__init__()

        self.residual_connection = residual_connection

        if isinstance(downsample, int):
            downsample = (downsample,) * 2

        stride = (1, 1)
        if downsample is not None and max(downsample) > 1:
            stride = downsample

        if normalization_factory is not None:
            bias = False
        else:
            bias = True

        blocks = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                stride=stride,
                padding=padding,
            )
        ]
        if normalization_factory is not None:
            blocks.append(
                normalization_factory(out_channels),
            )
        if activation_factory is not None:
            blocks.append(
                activation_factory(),
            )

        self.body = nn.Sequential(*blocks)
        if self.residual_connection and in_channels != out_channels or max(stride) > 1:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=stride, stride=stride
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate tensor through block.
        """
        y = self.body(x)
        if self.residual_connection:
            y = y + self.projection(x)
        return y


class BasicConv:
    """
    Factory for basic convolution blocks.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        residual_connection: bool = False,
    ):
        """
        Args:
            kernel_size: Kernel size of the convolution blocks
            padding: Padding applied to the input.
            activation_factory: Factory to use for the instantiation of the
                activation functions.
            normalization_factory: Factory to use for the instantition of the
                normalization layers in the convolution block.


        """
        self.kernel_size = kernel_size
        if padding is None:
            padding = self.kernel_size // 2
        self.padding = padding
        self.activation_factory = activation_factory
        self.normalization_factory = normalization_factory
        self.residual_connection = residual_connection

    def __call__(
        self, in_channels: int, out_channels: int, downsample: int = 1, **kwargs
    ):
        return BasicConvBlock(
            in_channels,
            out_channels,
            self.kernel_size,
            padding=self.padding,
            downsample=downsample,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
        )


class ResNetBlock(nn.Module, ParamCount):
    """
    Implements a basic ResNet block with bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        downsample: Optional[int] = None,
        bottleneck: int = 4,
    ):
        super().__init__()

        if isinstance(downsample, int):
            downsample = (downsample,) * 2

        stride = (1, 1)
        if downsample is not None and max(downsample) > 1:
            stride = downsample

        blocks = []
        if bottleneck > 1:
            blocks += [
                nn.Conv2d(in_channels, out_channels // bottleneck, kernel_size=1),
                normalization_factory(out_channels),
                activation_factory(inplace=True),
            ]
            in_channels = out_channels // bottleneck

        blocks += [
            nn.Conv2d(in_channels, out_channels // bottleneck, kernel_size=3),
            normalization_factory(out_channels // bottleneck),
            activation_factory(inplace=True),
        ]

        if bottleneck > 1:
            blocks += [
                nn.Conv2d(out_channels // bottleneck, out_channels, kernel_size=1),
                normalization_factory(out_channels),
                activation_factory(inplace=True),
            ]
        else:
            blocks += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3),
                normalization_factory(out_channels),
                activation_factory(inplace=True),
            ]

        self.body = nn.Sequential(blocks)

        self.body = nn.Sequential(*blocks)
        if self.residual_connection and in_channels != out_channels or max(stride) > 1:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=stride, stride=stride
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate tensor through block.
        """
        y = self.body(x)
        if self.residual_connection:
            y = y + self.projection(x)
        return y
