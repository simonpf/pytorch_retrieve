"""
Tests for the pytorch_retrieve.modules.conv.stems module.
"""
import torch

from pytorch_retrieve.modules.conv.stems import (
    BasicConv,
    BasicConv3d
)


def test_basic_stem():
    """
    Ensure that:
        - A simple stem with one layer yields the expected number of output
            channels
        - A stem with multiple layers works.
        - A stem with zero layers leaves the input unchanged.

    """
    stem = BasicConv(8, 16)
    x = torch.rand(1, 8, 16, 16)
    y = stem(x)
    assert y.shape == (1, 16, 16, 16)

    stem = BasicConv(8, 16, depth=3)
    x = torch.rand(1, 8, 16, 16)
    y = stem(x)
    assert y.shape == (1, 16, 16, 16)

    stem = BasicConv(8, 8, depth=0)
    x = torch.rand(1, 8, 16, 16)
    y = stem(x)
    assert y.shape == (1, 8, 16, 16)


def test_basic_3d_stem():
    """
    Ensure that:
        - A simple stem with one layer yields the expected number of output
            channels
        - A stem with multiple layers works.
        - A stem with zero layers leaves the input unchanged.

    """
    stem = BasicConv3d(8, 16)
    x = torch.rand(1, 8, 16, 16, 16)
    y = stem(x)
    assert y.shape == (1, 16, 16, 16, 16)

    stem = BasicConv3d(8, 16, depth=3)
    x = torch.rand(1, 8, 16, 16, 16)
    y = stem(x)
    assert y.shape == (1, 16, 16, 16, 16)

    stem = BasicConv3d(8, 8, depth=0)
    x = torch.rand(1, 8, 16, 16, 16)
    y = stem(x)
    assert y.shape == (1, 8, 16, 16, 16)
