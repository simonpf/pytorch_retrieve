"""
Tests for the pytorch_retrieve.modules.conv.block module.
"""
import torch

from pytorch_retrieve.modules.conv.blocks import (
    BasicConv,
    BasicConv3d,
    ResNet,
    ResNeXt,
    ResNeXt2Plus1,
    InvertedBottleneck,
    InvertedBottleneck2Plus1,
    Satformer
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


def test_inverted_bottleneck():
    """
    Tests for the InvertedBottleneck factory.
    """
    factory = InvertedBottleneck()
    module = factory(64, 64)

    x = torch.rand(1, 64, 32, 32)
    y = module(x)
    assert y.shape == (1, 64, 32, 32)

    factory = InvertedBottleneck(anti_aliasing=True)
    module = factory(64, 64, downsample=(2, 1))
    y = module(x)
    assert y.shape == (1, 64, 16, 32)

    # Test squeeze-and-excite block
    factory = InvertedBottleneck(
        expansion_factor=6,
        excitation_ratio=0.25,
        anti_aliasing=True

    )
    module_2 = factory(
        64, 64, downsample=(2, 1), expansion_factor=6,
        excitation_ratio=0.25, anti_aliasing=True
    )
    assert module_2.n_params > module.n_params

    y = module_2(x)
    assert y.shape == (1, 64, 16, 32)

    # Test fused block
    factory = InvertedBottleneck(
        expansion_factor=6,
        excitation_ratio=0.25,
        anti_aliasing=True,
        fused=True

    )
    module_3 = factory(
        64, 64, downsample=(2, 1), expansion_factor=6,
        excitation_ratio=0.25, anti_aliasing=True
    )
    assert module_3.n_params > module_2.n_params

    y = module_3(x)
    assert y.shape == (1, 64, 16, 32)

    # Stochastic depth.
    factory = InvertedBottleneck(
        expansion_factor=6,
        excitation_ratio=0.25,
        anti_aliasing=True,
        fused=True,
        stochastic_depth=0.0
    )
    module = factory(
        64, 64, downsample=(2, 1), expansion_factor=6,
        excitation_ratio=0.25, anti_aliasing=True
    )
    y = module(x)
    assert y.shape == (1, 64, 16, 32)


def test_inverted_bottleneck_2p1():
    """
    Tests for the InvertedBottleneck2Plus1 factory.
    """
    factory = InvertedBottleneck2Plus1()
    module = factory(64, 64)

    x = torch.rand(1, 64, 8, 32, 32)
    y = module(x)
    assert y.shape == (1, 64, 8, 32, 32)

    factory = InvertedBottleneck2Plus1(anti_aliasing=True)
    module = factory(64, 64, downsample=(2, 2, 1))
    y = module(x)
    assert y.shape == (1, 64, 4, 16, 32)

    # Test squeeze-and-excite block
    factory = InvertedBottleneck2Plus1(
        expansion_factor=6,
        anti_aliasing=True

    )
    module_2 = factory(
        64, 64, downsample=(2, 2, 1), expansion_factor=6,
        anti_aliasing=True
    )
    assert module_2.n_params > module.n_params

    y = module_2(x)
    assert y.shape == (1, 64, 4, 16, 32)

    # Test fused block
    factory = InvertedBottleneck2Plus1(
        expansion_factor=6,
        anti_aliasing=True,
        fused=True

    )
    module_3 = factory(
        64, 64, downsample=(1, 2, 1),
        expansion_factor=6,
        excitation_ratio=0.25,
        anti_aliasing=True
    )
    assert module_3.n_params > module_2.n_params

    y = module_3(x)
    assert y.shape == (1, 64, 8, 16, 32)

    # Test with squeeze-and-excite block.
    factory = InvertedBottleneck2Plus1(
        expansion_factor=6,
        excitation_ratio=0.25,
    )
    module_4 = factory(
        64, 64, downsample=(1, 2, 1), expansion_factor=6,
        excitation_ratio=0.25, anti_aliasing=True
    )
    assert module_4.n_params > module_2.n_params

    y = module_4(x)
    assert y.shape == (1, 64, 8, 16, 32)

    factory = InvertedBottleneck2Plus1(
        expansion_factor=6,
        excitation_ratio=0.25,
        stochastic_depth=0.1
    )
    module = factory(
        64, 64, downsample=(1, 2, 1), expansion_factor=6,
        excitation_ratio=0.25, anti_aliasing=True
    )
    y = module(x)
    assert y.shape == (1, 64, 8, 16, 32)


def test_satformer_block():
    """
    Tests for the Satformer block.
    """
    block_factory = Satformer(attention=True, n_heads=4)
    block = block_factory(32, 64)
    x = torch.rand(1, 32, 16, 64, 64)
    y = block(x)

    block_factory = Satformer(attention=True, n_heads=4, stochastic_depth=0.5)
    block = block_factory(32, 64)
    x = torch.rand(1, 32, 16, 64, 64)
    y = block(x)
