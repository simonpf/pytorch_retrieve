from typing import Callable, Union

import torch
from torch import nn

from pytorch_retrieve.modules.utils import ParamCount


class BasicConv(ParamCount, nn.Sequential):
    """
    Stem consisting of a convolution layer followed by optional normalization
    layer and activation function.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        depth: int = 1,
        downsampling: int = 1,
        activation_factory: Callable[[], nn.Module] = None,
        normalization_factory: Callable[[int], nn.Module] = None,
    ):
        """
        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels to extract from the input.
            kernel_size: The size of the convolution kernel.
            depth: The number of sequential convolution layers.
            activation: Optional factory used to create activation functions
                that are applied following every convolution layer.
            activation: Optional factory used to create normalization layers
                following every activation or convolution layer.
        """
        blocks = []

        if depth == 0 and in_channels != out_channels:
            raise ValueError(
                "If the depth of a stem is 0, the numbers of incoming and"
                " outcoming channels must be identical."
            )

        for _ in range(depth):
            blocks.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=downsampling,
                )
            )
            if normalization_factory is not None:
                blocks.append(normalization_factory)
            if activation_factory is not None:
                blocks.append(activation_factory())

            in_channels = out_channels
            downsampling = 1

        super().__init__(*blocks)
        self.out_channels = out_channels


class BasicConv3d(ParamCount, nn.Sequential):
    """
    Simple stem consisting of (potentially multiple) 3D convolution
    layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        depth: int = 1,
        downsampling: int = 1,
        activation_factory: Callable[[], nn.Module] = None,
        normalization_factory: Callable[[int], nn.Module] = None,
    ):
        """
        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels to extract from the input.
            kernel_size: The size of the convolution kernel.
            depth: The number of sequential convolution layers.
            activation: Optional factory used to create activation functions
                that are applied following every convolution layer.
            activation: Optional factory used to create normalization layers
                following every activation or convolution layer.
        """
        blocks = []

        if depth == 0 and in_channels != out_channels:
            raise ValueError(
                "If the depth of a stem is 0, the numbers of incoming and"
                " outcoming channels must be identical."
            )

        for _ in range(depth):
            blocks.append(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=downsampling,
                )
            )
            if normalization_factory is not None:
                blocks.append(normalization_factory)
            if activation_factory is not None:
                blocks.append(activation_factory())

            in_channels = out_channels
            downsampling = 1

        super().__init__(*blocks)
        self.out_channels = out_channels
