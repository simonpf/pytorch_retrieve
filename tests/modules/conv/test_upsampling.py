"""
Tests for the pytorch_retrieve.modules.conv.upsampling module.
"""

import pytest
import torch

from pytorch_retrieve.modules.conv.upsampling import (
    Bilinear,
    BilinearWNorm,
    Trilinear,
    ConvTranspose,
    StepwiseBilinear,
)


@pytest.mark.parametrize("upsampler_factory", (Bilinear, BilinearWNorm, ConvTranspose))
def test_bilinear(upsampler_factory):
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


def test_trilinear():
    """
    Instantiate a trilinear upsampler factory, create upsampler modules, and ensure that
    they correctly upsample inputs.
    """
    upsampler_factory = Trilinear()

    upsampler = upsampler_factory(8, 8, 2)
    x = torch.rand(1, 8, 8, 8, 8)
    y = upsampler(x)
    assert y.shape == (1, 8, 16, 16, 16)

    upsampler = upsampler_factory(8, 16, 2)
    x = torch.rand(1, 8, 8, 8, 8)
    y = upsampler(x)
    assert y.shape == (1, 16, 16, 16, 16)


def test_stepwise_bilinear():
    """
    Instantiate an upsampler factory, create upsampler modules, and ensure that
    they correctly upsample inputs.
    """
    upsampler_factory = StepwiseBilinear()

    upsampler = upsampler_factory(8, 8, 2)
    x = torch.rand(1, 8, 4, 8, 8)
    y = upsampler(x)
    assert y.shape == (1, 8, 4, 16, 16)
