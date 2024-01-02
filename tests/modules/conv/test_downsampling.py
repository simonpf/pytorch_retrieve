import torch

from pytorch_retrieve.modules.conv.downsampling import MaxPool2d


def test_maxpool2d():
    """
    Use MaxPool2d downsampling factory to create downsampling module and use
    it to downsample tensor. Esure output has expected size.
    """
    factory = MaxPool2d()
    module = factory(8, 16, 2)

    x = torch.rand(1, 8, 16, 16)
    y = module(x)
    assert y.shape == (1, 16, 8, 8)
