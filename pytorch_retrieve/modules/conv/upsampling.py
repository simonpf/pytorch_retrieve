"""
mDefines factories for creating upsampling block for use in convolutional
neural network architectures.
"""
from typing import Tuple, Union

from torch import nn


class Bilinear:
    """
    Factory for creating bilinear upsampling modules.
    """

    def __call__(
        self, in_channels: int, out_channels: int, factor: Union[float, Tuple[float]]
    ) -> nn.Module:
        """
        Args:
            factor: A scalar factor defining the factor by which to upsample
                the resolution of a tensor. A tuple of upsampling factors can
                provided to upsample the input tensor by different factors
                along its height and width dimensions.

        Return:
            A pytorch.nn.Module object that upsamples a given 4D tensor along
            the last two dimensions by 'factor'.
        """
        blocks = []
        if in_channels != out_channels:
            blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        if isinstance(factor, (int, float)):
            factor = (factor,) * 2
        factor = tuple(factor)

        blocks.append(
            nn.Upsample(
                scale_factor=factor,
                mode="bilinear",
            )
        )
        return nn.Sequential(*blocks)


class Trilinear:
    """
    Factory for upsampling of 3d data.
    """
    def __call__(
        self, in_channels: int, out_channels: int, factor: Union[float, Tuple[float]]
    ) -> nn.Module:
        """
        Args:
            factor: A scalar factor defining the factor by which to upsample
                the resolution of a tensor. A tuple of upsampling factors can
                provided to upsample the input tensor by different factors
                along its height and width dimensions.

        Return:
            A pytorch.nn.Module object that upsamples a given 4D tensor along
            the last two dimensions by 'factor'.
        """
        blocks = []
        if in_channels != out_channels:
            blocks.append(nn.Conv3d(in_channels, out_channels, kernel_size=1))

        if isinstance(factor, (int, float)):
            factor = (factor,) * 3
        factor = tuple(factor)

        blocks.append(
            nn.Upsample(
                scale_factor=factor,
                mode="trilinear",
            )
        )
        return nn.Sequential(*blocks)


class ConvTranspose:
    """
    Upsampling using transposed convolutions.
    """
    def __call__(
            self, in_channels: int, out_channels: int, factor: Union[float, Tuple[float]]
    ) -> nn.Module:
        stride = int(factor)
        kernel_size = stride
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
