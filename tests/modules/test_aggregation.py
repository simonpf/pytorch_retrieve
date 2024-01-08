"""
Tests for the pytorch_retrieve.modules.aggregation module.
"""
import torch

from pytorch_retrieve.modules.aggregation import MLPAggregator


def test_mlp_aggregator():
    """
    Create an MLP aggregator to merge two inputs with different number of
    channels. Propagate input through aggregator and ensure that output
    has expected shape.
    """
    inputs = {"x_1": 16, "x_2": 8}
    agg = MLPAggregator(inputs=inputs, out_channels=32, n_layers=3)

    inputs = {"x_1": torch.rand(32, 16), "x_2": torch.rand(32, 8)}
    y = agg(inputs)
    assert y.shape == (32, 32)
