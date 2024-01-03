"""
pytorch_retrieve.modules.conv.downsampling
==========================================

This module provides downsampling block factories for the use in generic
encoder-decoder architecture.
"""
from typing import Callable, Optional, Tuple, Union

from torch import nn


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
    ):
        """
        Create downsampling module.

        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels in the output.
            downsampling_factory:

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
            blocks.append(normalization_factory(out_channels))

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
