"""
Tests for the pytorch_retrieve.modules.metnet module.
"""
import torch

from pytorch_retrieve.modules.metnet import Stem, SpatialAggregator


def test_stem():
    """
    Test Metnet stem and ensure that:
        - Downsampling works with average pooling
        - Downsampling works with space-to-depth downsampling and channel
          numbers are increased as expected
        - Downsampling works without appending the center crop
    """
    stem = Stem(16)
    x = torch.Tensor(1, 16, 32, 32)
    y = stem(x)
    assert y.shape == (1, 32, 8, 8)

    stem = Stem(16, first_stage_kind="space2depth")
    x = torch.Tensor(1, 16, 32, 32)
    y = stem(x)
    assert y.shape == (1, 32 * 4, 8, 8)

    stem = Stem(16, first_stage_kind="space2depth", center_crop=False)
    x = torch.Tensor(1, 16, 32, 32)
    y = stem(x)
    assert y.shape == (1, 32 * 2, 8, 8)


def test_spatial_aggregator():
    """
    Ensure that propagating tensors through spatial aggregator works.
    """
    agg = SpatialAggregator(32, 32, 4, 4)
    x = torch.rand(1, 32, 32, 32)
    y = agg(x)

    assert y.shape == (1, 32, 32, 32)
