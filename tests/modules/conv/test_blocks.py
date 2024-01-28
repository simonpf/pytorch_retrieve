"""
Tests for the pytorch_retrieve.modules.conv.block module.
"""
import torch

from pytorch_retrieve.modules.conv.blocks import (
    BasicConv,
    BasicConv3d,
    ResNet,
    ResNeXt,
    ResNeXt2Plus1
)


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

    factory = BasicConv(anti_aliasing=True)
    module = factory(16, 32, downsample=(1, 2))
    y = module(x)
    assert y.shape == (1, 32, 32, 16)


def test_basic_conv3d():
    """
    Test basic convolution.
    """
    factory = BasicConv3d()
    module = factory(16, 32)

    x = torch.rand(1, 16, 32, 32, 32)
    y = module(x)
    assert y.shape == (1, 32, 32, 32, 32)

    module = factory(16, 32, downsample=(2, 2, 1))
    y = module(x)
    assert y.shape == (1, 32, 16, 16, 32)

    factory = BasicConv3d(anti_aliasing=True)
    module = factory(16, 32, downsample=(2, 2, 1))
    y = module(x)
    assert y.shape == (1, 32, 16, 16, 32)

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

    module = factory(16, 32, downsample=(1, 2), anti_aliasing=True)
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

    module = factory(64, 64, downsample=(1, 2), anti_aliasing=True)
    y = module(x)
    assert y.shape == (1, 64, 32, 16)


def test_resnext2plus1():
    """
    Tests for the ResNeXt2Plus1 factory.
    """
    factory = ResNeXt2Plus1()
    module = factory(64, 64)

    x = torch.rand(1, 64, 32, 32, 32)
    y = module(x)
    assert y.shape == (1, 64, 32, 32, 32)

    module = factory(64, 64, downsample=(1, 2, 1))
    y = module(x)
    assert y.shape == (1, 64, 32, 16, 32)
