"""
Tests for the pytorch_retrieve.modules.stats module.
"""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_retrieve.data.synthetic import Synthetic1d, Synthetic3d
from pytorch_retrieve.modules.stats import StatsTracker
from pytorch_retrieve.architectures import load_model, load_and_compile_model


def data_loader_1d(n_samples: int, batch_size: int) -> DataLoader:
    """
    A DataLoader providing batches of Synthetic1D data.
    """
    data = Synthetic1d(n_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def test_input_1d(tmp_path):
    """
    Calculate statistic of 1D dataset and ensure that NANs are ignored
    in the stats calculation.
    """
    data_loader = data_loader_1d(2048, 64)

    stats_layer = StatsTracker(n_features=1)
    for x, y in data_loader:
        stats_layer.track_stats(x)
    stats_1 = stats_layer.compute_stats()

    stats_layer = StatsTracker(n_features=1)
    for x, y in data_loader:
        mask = torch.rand(*x.shape) < 0.5
        x[mask] = torch.nan
        stats_layer.track_stats(x)

    stats_2 = stats_layer.compute_stats()
    assert np.isclose(stats_1["mean"].data, stats_2["mean"].data, rtol=0.05)

    # Ensure that histograms are included in stats after second pass
    # throught dataset.
    stats_layer.epoch_finished()
    for x, y in data_loader:
        mask = torch.rand(*x.shape) < 0.5
        x[mask] = torch.nan
        stats_layer.track_stats(x)

    stats_2 = stats_layer.compute_stats()
    assert "bin_boundaries" in stats_2


def test_hist_1d(tmp_path):
    """
    Test calculation of histograms for 1D data.
    """
    data_loader = data_loader_1d(2048, 64)

    stats_layer = StatsTracker(n_features=1)
    for x, y in data_loader:
        x = x.to(dtype=torch.float32)
        stats_layer.track_stats(x)
    stats_1 = stats_layer.compute_stats()

    stats_layer.epoch_finished()

    for x, y in data_loader:
        stats_layer.track_stats(x)

    assert len(stats_layer.bins) == stats_layer.n_features


def data_loader_1d_nan(n_samples: int, batch_size: int) -> DataLoader:
    """
    A DataLoader providing batches of Synthetic1D data.
    """
    data = Synthetic1d(n_samples)
    data.x[:] = np.nan
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def test_hist_1d_nan(tmp_path):
    """
    Test calculation of histograms for 1D data.
    """
    data_loader = data_loader_1d_nan(2048, 64)
    stats_layer = StatsTracker(n_features=1)
    for x, y in data_loader:
        x = x.to(dtype=torch.float32)
        stats_layer.track_stats(x)
    stats_1 = stats_layer.compute_stats()

    stats_layer.epoch_finished()

    for x, y in data_loader:
        x = x.to(torch.bfloat16)
        stats_layer.track_stats(x)
