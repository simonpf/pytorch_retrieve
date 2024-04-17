"""
Tests for the pytorch_retrieve.module.conv.padding module.
"""
import pytest
import torch

from pytorch_retrieve.modules.conv.padding import (
    calculate_padding,
    get_padding_factory,
    Crop
)

def test_calculate_padding():
    """
    Ensure that calculate padding for given kernel filter configuration matches
    expected values.
    """
    padding = calculate_padding((3, 1))
    assert padding == (1, 0)


@pytest.mark.parametrize("padding_factory", ["Zero", "Reflect", "Global"])
def test_padding(padding_factory):
    """
    Ensure that padding is performed correctly along the last two dimensions.
    """
    padding_factory = get_padding_factory(padding_factory)
    x = torch.rand(1, 2, 6, 4)
    padding = padding_factory((1, 2))
    y = padding(x)
    assert y.shape == (1, 2, 8, 8)


def test_global_padding():
    """
    Ensure that global padding is periodic along last dimension.
    """
    padding_factory = get_padding_factory("Global")
    x = torch.arange(8).repeat((1, 1, 4, 1)).to(torch.float32)
    padding = padding_factory((2, 3))
    y = padding(x)
    assert (y[0, 0, 0, :3] == torch.tensor([5.0, 6.0, 7.0])).all()
    assert (y[0, 0, 0, -3:] == torch.tensor([0.0, 1.0, 2.0])).all()

    x = torch.arange(8).repeat((1, 1, 1, 4, 1)).to(torch.float32)
    padding = padding_factory((0, 2, 3))
    y = padding(x)
    assert (y[0, 0, 0, 0, :3] == torch.tensor([5.0, 6.0, 7.0])).all()
    assert (y[0, 0, 0, 0, -3:] == torch.tensor([0.0, 1.0, 2.0])).all()



def test_zero_padding():
    """
    Test that zero padding pads zeros.
    """
    padding_factory = get_padding_factory("Zero")
    padding = padding_factory((1, 3))
    x = torch.arange(8).repeat((1, 1, 4, 1)).to(torch.float32)
    y = padding(x)
    assert (y[0, 0, 0, :3] == torch.tensor([0.0, 0.0, 0.0])).all()
    assert (y[0, 0, 0, -3:] == torch.tensor([0.0, 0.0, 0.0])).all()


def test_crop_padding():
    """
    Test that cropping removes the expected number of pixels.

    """
    crop = Crop(((1, 3), (2, 4)))
    x = torch.arange(8).repeat((1, 1, 16, 1)).to(torch.float32)
    y = crop(x)
    assert y.shape == (1, 1, 12, 2)
    assert (y[0, 0, 0] == torch.tensor([2, 3])).all()
