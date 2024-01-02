"""
Tests for the pytorch_retrieve.modules.heads module.
"""
import torch

from pytorch_retrieve.modules.conv.heads import BasicConv


def test_basic_conv():
    """
    Test a basic head made up of convolution layers.

    """
    head = BasicConv(
        in_channels=16,
        out_shape=(4, 4),
        depth=2,
    )
    x = torch.rand(1, 16, 32, 32)
    y = head(x)
    assert y.shape == (1, 4, 4, 32, 32)

    head = BasicConv(
        in_channels=16,
        out_shape=(4),
        depth=2,
    )
    y = head(x)
    assert y.shape == (1, 4, 32, 32)
