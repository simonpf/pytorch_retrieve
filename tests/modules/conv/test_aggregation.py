"""
Tests for the pytorch_retrieve.modules.aggregation module.
"""
import torch

from pytorch_retrieve.modules.conv.aggregation import Linear


def test_linear_aggregator():
    inputs = {"x_1": 16, "x_2": 8}
    agg = Linear(inputs, 32)

    inputs = {"x_1": torch.rand(1, 16, 32, 32), "x_2": torch.rand(1, 8, 32, 32)}
    y = agg(inputs)
    assert y.shape == (1, 32, 32, 32)


def test_linear_3d_aggregator():
    inputs = {"x_1": 16, "x_2": 8}
    agg = Linear(inputs, 32)

    inputs = {"x_1": torch.rand(1, 16, 32, 32, 32), "x_2": torch.rand(1, 8, 32, 32, 32)}
    y = agg(inputs)
    assert y.shape == (1, 32, 32, 32, 32)
