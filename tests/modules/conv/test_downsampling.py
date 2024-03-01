import torch
from torch import nn

from pytorch_retrieve.modules.conv.downsampling import (
    BlurPool,
    MaxPool2d,
    AvgPool2d,
    Space2Depth
)

def test_blurpool():

    x = torch.ones(1, 8, 5, 5)
    x[0, :, 2, 2] = 1.0
    blur_pool = BlurPool(8, (2, 2), (3, 3))
    y = blur_pool(x)
    assert torch.isclose(y, torch.tensor(1.0)).all()



def test_maxpool2d():
    """
    Use MaxPool2d downsampling factory to create downsampling module and use
    it to downsample tensor. Esure output has expected size.
    """
    factory = MaxPool2d()

    x = torch.ones((1, 8, 16, 16))

    module = factory(8, 16, 2)
    y = module(x)
    assert y.shape == (1, 16, 8, 8)

    module = factory(8, 8, 2)
    y = module(x)
    assert y.shape == (1, 8, 8, 8)
    assert torch.isclose(y.sum(), torch.tensor(8.0 * 8.0 * 8.0))

    factory = MaxPool2d(normalization_factory=nn.BatchNorm2d)
    x = torch.ones((1, 8, 16, 16))
    module = factory(8, 16, 2)
    y = module(x)
    assert y.shape == (1, 16, 8, 8)


def test_avgpool2d():
    """
    Use MaxPool2d downsampling factory to create downsampling module and use
    it to downsample tensor. Esure output has expected size.
    """
    factory = AvgPool2d()
    module = factory(8, 16, 2)

    x = torch.ones(1, 8, 16, 16)
    y = module(x)
    assert y.shape == (1, 16, 8, 8)

    module = factory(8, 8, 2)
    y = module(x)
    assert y.shape == (1, 8, 8, 8)
    assert torch.isclose(y.sum(), torch.tensor(8.0 * 8.0 * 8.0))

    factory = AvgPool2d(normalization_factory=nn.BatchNorm2d)
    x = torch.ones((1, 8, 16, 16))
    module = factory(8, 16, 2)
    y = module(x)
    assert y.shape == (1, 16, 8, 8)


def test_space2depth():
    """
    Use Space2Depth downsampling factory to create downsampling module and use
    it to downsample tensor. Esure output has expected size and that order of
    elements is correct.
    """
    factory = Space2Depth()
    module = factory(8, 16, 2)

    x = torch.ones(1, 8, 16, 16)
    y = module(x)
    assert y.shape == (1, 16, 8, 8)

    module = factory(8, 32, 2)
    y = module(x)
    assert y.shape == (1, 32, 8, 8)
    assert torch.isclose(y.sum(), torch.tensor(8.0 * 16.0 * 16.0))

    module = factory(1, 4, 2)
    x = torch.arange(8).repeat(1, 1, 8, 1)
    y = module(x)
    assert torch.all(torch.isclose(y[0, :, 0, 0], torch.tensor([0, 1, 0, 1])))
    x = torch.arange(8).unsqueeze(-1).repeat(1, 1, 1, 8)
    y = module(x)
    assert torch.all(torch.isclose(y[0, :, 0, 0], torch.tensor([0, 0, 1, 1])))

    factory = Space2Depth(normalization_factory=nn.BatchNorm2d)
    module = factory(8, 32, 2)
    x = torch.ones(1, 8, 16, 16)
    y = module(x)
    assert y.shape == (1, 32, 8, 8)
    assert torch.isclose(y.sum(), torch.tensor(8.0 * 16.0 * 16.0))
