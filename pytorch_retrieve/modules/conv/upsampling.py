"""
mDefines factories for creating upsampling block for use in convolutional
neural network architectures.
"""

from typing import Tuple, Union

import torch
from torch import nn

from pytorch_retrieve.modules.normalization import LayerNormFirst


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


class BilinearWNorm:
    """
    Factory for creating bilinear upsampling modules with normalization layers.
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

        blocks += [
            nn.Upsample(
                scale_factor=factor,
                mode="bilinear",
            ),
            LayerNormFirst(out_channels)
        ]
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
        stride = factor
        kernel_size = stride
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)


class StepwiseBilinearBlock(nn.Module):
    """
    Bilinear upsampling for sequence data.
    """

    def __init__(self, scale_factor):
        """
        Args:
            scale_factor: The factor by which to upsample the input.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample input.
        """
        n_batch, n_seq, *rem = x.shape
        x_up = self.up(x.view((-1,) + tuple(rem)))
        return x_up.view(
            (
                n_batch,
                n_seq,
            )
            + x_up.shape[1:]
        )


class StepwiseBilinear:
    """
    Factory for upsampling of 3d data along spatial dimensions only.
    """

    def __call__(
        self, in_channels: int, out_channels: int, factor: Union[float, Tuple[float]]
    ) -> nn.Module:
        """
        Args:
            in_channels: The number of incoming channels.
            out_channels: The number of outgoing channels.
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
            factor = (factor,) * 2
        factor = tuple(factor)

        blocks.append(
            StepwiseBilinearBlock(
                scale_factor=factor,
            )
        )
        return nn.Sequential(*blocks)
