"""
Tests for the pytorch_retrieve.modules.conv.upsampling module.
"""
import torch

from pytorch_retrieve.modules.conv.upsampling import Bilinear


def test_bilinear():
    """
    Instantiate an upsampler factory, create upsampler modules, and ensure that
    they correctly upsample inputs.
    """
    upsampler_factory = Bilinear()

    upsampler = upsampler_factory(8, 8, 2)
    x = torch.rand(1, 8, 8, 8)
    y = upsampler(x)
    assert y.shape == (1, 8, 16, 16)

    upsampler = upsampler_factory(8, 16, 2)
    x = torch.rand(1, 8, 8, 8)
    y = upsampler(x)
    assert y.shape == (1, 16, 16, 16)
