"""
Tests for the pytorch_retrieve.module.conv.padding module.
"""
import pytest
import torch

from pytorch_retrieve.modules.conv.padding import (
    Zero,
    Reflect,
    Global
)

@pytest.mark.parametrize("padding_factory", [Zero, Reflect, Global])
def test_padding(padding_factory):
    """
    Ensure that padding is performed correctly along the last two dimensions.
    """
    x = torch.rand(1, 2, 6, 4)
    padding = padding_factory((1, 2))
    y = padding(x)
    assert y.shape == (1, 2, 8, 8)


def test_global_padding():
    """
    Ensure that global padding is periodic along last dimension.
    """
    x = torch.arange(8).repeat((1, 1, 4, 1))
    padding = Global((2, 3))
    y = padding(x)
    assert (y[0, 0, 0, :3] == torch.tensor([5.0, 6.0, 7.0])).all()
    assert (y[0, 0, 0, -3:] == torch.tensor([0.0, 1.0, 2.0])).all()
