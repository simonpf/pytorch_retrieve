"""
Tests for the pytorch_retrieve.modules.conv.stages module.
"""
import torch

from pytorch_retrieve.modules.conv.encoders import BasicConv
from pytorch_retrieve.modules.conv.stages import SequentialStage


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
