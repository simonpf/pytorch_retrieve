"""
pytorch_retrieve.modules.conv.downsampling
==========================================

This module provides downsampling block factories for the use in generic
encoder-decoder architecture.
"""
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from pytorch_retrieve.modules.utils import NoNorm


class BlurPool(nn.Module):
    """
    A blur pool layer for 2D and 3D downsampling.

    See 'Making Convolutional Networks Shift-Invariant Again' for motivation.
    """
    def __init__(self, in_channels: int, stride: Tuple[int], filter_size: Tuple[int]):
        """
        Instantiate blur pool filter.

        Args:
            in_channels: The number of channels in the input tensor.
            stride: Tuple specifying the stride applied along each dimension.
            filter_size: Tuple specifying the size of the low-pass filter
                applied along the time and spatial dimensions.
        """
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        if len(filter_size) == 2:
            x = torch.tensor(np.array(np.poly1d((0.5, 0.5)) ** (filter_size[1] - 1)))
            x = x.to(dtype=torch.float32)
            y = torch.tensor(np.array(np.poly1d((0.5, 0.5)) ** (filter_size[0] - 1)))
            y = y.to(dtype=torch.float32)
            k = y[:, None] * x[None, :]
            self.filter = nn.Parameter(
                k.repeat((in_channels, 1, 1, 1)),
                requires_grad=False
            )
        elif len(filter_size) == 3:
            x = torch.tensor(np.array(np.poly1d((0.5, 0.5)) ** (filter_size[2] - 1)))
            x = x.to(dtype=torch.float32)
            y = torch.tensor(np.array(np.poly1d((0.5, 0.5)) ** (filter_size[1] - 1)))
            y = y.to(dtype=torch.float32)
            z = torch.tensor(np.array(np.poly1d((0.5, 0.5)) ** (filter_size[0] - 1)))
            z = z.to(dtype=torch.float32)
            k = (z[:, None, None] * y[None, :, None] * x[None, None, :]
            )
            self.filter = nn.Parameter(
                k.repeat(in_channels, 1, 1, 1, 1),
                requires_grad=False
            )
        else:
            raise ValueError("Filter size must be a tuple of length 2 or 3.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply blur pool to input."""
        if self.filter.ndim == 5:
            return nn.functional.conv3d(x, self.filter, stride=self.stride, groups=self.in_channels)
        return nn.functional.conv2d(x, self.filter, stride=self.stride, groups=self.in_channels)




class DownsamplingFactoryBase(nn.Module):
    """
    Generic downsampling factory that includes a linear projection layer
    to adapt channels.
    """

    def __init__(
        self,
        downsampling_class: Callable[[], None],
        normalization_factory: Optional[Callable[[int], nn.Module]] = None,
    ):
        """
        Args:
            downsampling_class: The PyTorch class to use to instantiate the
                downsampling layer.
            normalization_factory: An optional normalization factory that is
                applied after the projection.
        """
        self.downsampling_class = downsampling_class
        self.normalization_factory = normalization_factory

    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        downsampling_factor: Union[int, Tuple[int, int]],
    ) -> nn.Module:
        """
        Create downsampling module.

        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels in the output.
            downsampling_factor: The degree of downsampling to apply.

        Return:
            The downsampling module.
        """
        blocks = []
        if in_channels != out_channels:
            blocks.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        blocks.append(
            self.downsampling_class(
                kernel_size=downsampling_factor, stride=downsampling_factor
            )
        )
        if self.normalization_factory is not None:
            blocks.append(self.normalization_factory(out_channels))

        return nn.Sequential(*blocks)


class MaxPool2d(DownsamplingFactoryBase):
    """
    Factory creating max-pooling downsampling blocks.
    """

    def __init__(
        self, normalization_factory: Optional[Callable[[int], nn.Module]] = None
    ):
        super().__init__(nn.MaxPool2d, normalization_factory=normalization_factory)


class AvgPool2d(DownsamplingFactoryBase):
    """
    Factory creating max-pooling downsampling blocks.
    """

    def __init__(
        self, normalization_factory: Optional[Callable[[int], nn.Module]] = None
    ):
        super().__init__(nn.AvgPool2d, normalization_factory=normalization_factory)


class Space2DepthModule(nn.Module):
    def __init__(self, downsampling_factor):
        super().__init__()
        self.downsampling_factor = downsampling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Downsample tensor by transforming spatial features to channels.

        Args:
            x: The input tensor to downsample.
            downsampling_factor: The factor by which to decrease the spatial dimensions.

        Return:
            A spatially-downsampled tensor with the number of channels increased by
            the square of the downsampling factor.
        """
        x = x.contiguous()
        in_shape = x.shape
        new_shape = list(in_shape)
        new_shape[-1] //= self.downsampling_factor
        new_shape[-2] //= self.downsampling_factor
        new_shape.insert(-1, self.downsampling_factor)
        new_shape.append(self.downsampling_factor)
        x = x.reshape(new_shape)
        out_shape = (
            in_shape[0],
            -1,
            in_shape[2] // self.downsampling_factor,
            in_shape[3] // self.downsampling_factor,
        )
        x = x.permute((0, 1, 3, 5, 2, 4)).reshape(out_shape)
        return x


class Space2Depth:
    def __init__(
        self, normalization_factory: Optional[Callable[[int], nn.Module]] = None
    ):
        """
        Args:
            down: The degree of downsampling to apply.
        """
        self.normalization_factory = normalization_factory

    def __call__(self, in_channels: int, out_channels: int, downsampling_factor: int):
        down_channels = in_channels * downsampling_factor**2
        if down_channels == out_channels:
            return Space2DepthModule(downsampling_factor=downsampling_factor)
        blocks = [
            Space2DepthModule(downsampling_factor=downsampling_factor),
            nn.Conv2d(down_channels, out_channels, kernel_size=1),
        ]
        if self.normalization_factory is not None:
            blocks.append(self.normalization_factory(out_channels))
        return nn.Sequential(*blocks)
