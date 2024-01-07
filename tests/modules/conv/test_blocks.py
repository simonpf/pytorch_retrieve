"""
Tests for the pytorch_retrieve.modules.conv.block module.
"""
import torch

from pytorch_retrieve.modules.conv.blocks import BasicConv, ResNet, ResNeXt


def test_basic_conv():
    """
    Test basic convolution.
    """
    factory = BasicConv()
    module = factory(16, 32)

    x = torch.rand(1, 16, 32, 32)
    y = module(x)
    assert y.shape == (1, 32, 32, 32)

    module = factory(16, 32, downsample=(1, 2))
    y = module(x)
    assert y.shape == (1, 32, 32, 16)


def test_resnet():
    """
    Tests for the ResNet factory.
    """
    factory = ResNet()
    module = factory(16, 32)

    x = torch.rand(1, 16, 32, 32)
    y = module(x)
    assert y.shape == (1, 32, 32, 32)

    module = factory(16, 32, downsample=(1, 2))
    y = module(x)
    assert y.shape == (1, 32, 32, 16)


def test_resnext():
    """
    Tests for the ResNeXt factory.
    """
    factory = ResNeXt()
    module = factory(64, 64)

    x = torch.rand(1, 64, 32, 32)
    y = module(x)
    assert y.shape == (1, 64, 32, 32)

    module = factory(64, 64, downsample=(1, 2))
    y = module(x)
    assert y.shape == (1, 64, 32, 16)
