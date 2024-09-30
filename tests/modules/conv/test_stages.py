"""
Tests for the pytorch_retrieve.modules.conv.stages module.
"""
from typing import Optional, Tuple

import torch
from torch import nn

from pytorch_retrieve.modules.conv.encoders import BasicConv
from pytorch_retrieve.modules.conv.stages import (
    SequentialStage,
    SequentialWKeywordsStage
)


def test_sequential_stage():
    """
    Build a stage consisting of multiple conv block and ensure that propagating
    a tensor through it yields a tensor of the expected size.
    """
    block_factory = BasicConv(
        kernel_size=3, normalization_factory=None, activation_factory=None
    )
    stage_factory = SequentialStage(block_factory)

    stage = stage_factory(8, 8, 4)
    x = torch.rand(1, 8, 8, 8)
    y = stage(x)
    assert y.shape == (1, 8, 8, 8)

    stage = stage_factory(8, 8, 4, (2, 4))
    x = torch.rand(1, 8, 8, 8)
    y = stage(x)
    assert y.shape == (1, 8, 4, 2)


class KeywordBlock(nn.Module):
    """
    Dummy block that returns the y keyword instead of processing its inputs.

    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downsample: Tuple[int, int],
            block_index: int
    ):
        super().__init__()

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        return y


def test_sequential_w_keyword_stage():
    """
    Test that passing a keyword to the block in a sequential stage works.
    """
    block_factory = KeywordBlock
    stage_factory = SequentialWKeywordsStage(block_factory)

    stage = stage_factory(8, 8, 4)
    x = torch.zeros(1, 8, 8, 8)
    y = torch.rand(1, 8, 8, 8)
    y = stage(x, y=y)
    assert torch.all(y == y)
