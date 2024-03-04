"""
pytorch_retrieve.modules.conv.projection
========================================

Define funcitonality for composing projection modules used for skip connections.
"""
from typing import Callable, Tuple
from torch import nn

from .downsampling import BlurPool
from .padding import Reflect


def get_projection(
        in_channels: int,
        out_channels: int,
        stride: Tuple[int],
        anti_aliasing: bool = False,
        padding_factory: Callable[[Tuple[int]], nn.Module] = Reflect
):
    """
    Get a projection module that adapts an input tensor to a smaller input tensor that is
    downsampled using the strides defined in 'stride'.

    Args:
        in_channels: The number of channels in the input tensor.
        out_channels: The number of channels in the output tensor.
        stride: The stride by which the input should be downsampled.
        anti_aliasing: Wether or not to apply anti-aliasing before downsampling.
        padding_factory: A factor for producing the padding blocks used in the model.

    Return:
        A projection module to project the input to the dimensions of the output.
    """
    if max(stride) == 1:
        if in_channels == out_channels:
            return nn.Identity()
        if len(stride) == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size=1)
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)


    blocks = []

    if anti_aliasing:
        pad = tuple([1 if strd > 1 else 0 for strd in stride])
        filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
        strd = (1,) * len(stride)
        blocks += [
            padding_factory(pad),
            BlurPool(in_channels, strd, filter_size)
        ]

    if len(stride) == 3:
        blocks.append(
            nn.Conv3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        )
    else:
        blocks.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride)
        )
    return nn.Sequential(*blocks)
