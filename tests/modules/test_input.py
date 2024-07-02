"""
Tests for the pytorch_retrieve.modules.input module.
"""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pytorch_retrieve.modules.stats import save_stats
from pytorch_retrieve.data.synthetic import Synthetic1d, Synthetic3d
from pytorch_retrieve.modules.input import InputLayer, StandardizationLayer
from pytorch_retrieve.architectures import load_model, load_and_compile_model


def data_loader_1d(n_samples: int, batch_size: int) -> DataLoader:
    """
    A DataLoader providing batch of Synthetic1D data.
    """
    data = Synthetic1d(n_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def test_input_1d(tmp_path):
    """
    Calculate statistic of 1D dataset and ensure that NANs are ignored
    in the stats calculation.
    """
    data_loader = data_loader_1d(2048, 64)

    input_layer = InputLayer("x", n_features=1, stats_path=tmp_path / "stats")
    for x, y in data_loader:
        input_layer(x)
    stats_1 = input_layer.compute_stats()

    input_layer = InputLayer("x", n_features=1, stats_path=tmp_path / "stats")
    for x, y in data_loader:
        mask = torch.rand(*x.shape) < 0.5
        x[mask] = torch.nan
        input_layer(x)

    stats_2 = input_layer.compute_stats()

    assert np.isclose(stats_1["mean"], stats_2["mean"], rtol=0.05)


def test_hist_1d(tmp_path):
    """
    Test calculation of histograms for 1D data.
    """
    data_loader = data_loader_1d(2048, 64)

    input_layer = InputLayer("x", n_features=1, stats_path=tmp_path / "stats")
    for x, y in data_loader:
        input_layer(x)
    stats_1 = input_layer.compute_stats()

    input_layer.epoch_finished()

    for x, y in data_loader:
        input_layer(x)

    assert len(input_layer.bins) == input_layer.n_features


def test_finalization(tmp_path):
    """
    Test saving and loading of EDA results.
    """
    data_loader = data_loader_1d(2048, 64)

    input_layer_1 = InputLayer("x", n_features=1, stats_path=tmp_path / "stats")
    for x, y in data_loader:
        input_layer_1(x)
    input_layer_1.epoch_finished()

    for x, y in data_loader:
        input_layer_1(x)
    input_layer_1.epoch_finished()

    stats = input_layer_1.compute_stats()
    save_stats(stats, tmp_path / "stats" / "input", "x")
    assert (tmp_path / "stats" / "input" / "x.nc").exists()

    input_layer_2 = InputLayer("x", n_features=1, stats_path=tmp_path / "stats")
    assert input_layer_2.finalized


def test_normalization_1d(tmp_path):
    """
    Perform EDA on synthetic dataset and use input layers to normalize inputs.
    Ensure that:
        - Mean and std. dev. of standardized inputs are close to 0
          and 1, respectively.
        - Minimum and maximum of min-max normalized inputs are close to -1.0
          and 1.0, respectively.
    """
    data_loader = data_loader_1d(2048, 64)

    input_layer_1 = StandardizationLayer(
        "x", n_features=1, stats_path=tmp_path / "stats", kind="standardize"
    )
    for x, y in data_loader:
        input_layer_1(x)
    input_layer_1.epoch_finished()
    for x, y in data_loader:
        input_layer_1(x)
    input_layer_1.epoch_finished()

    # Standardization
    input_layer_2 = StandardizationLayer("x", n_features=1, stats_path=None)
    for x, y in data_loader:
        x = input_layer_1(x)
        input_layer_2(x)
    input_layer_2.epoch_finished()

    stats = input_layer_2.compute_stats()
    assert (np.isclose(stats["mean"], 0.0, atol=1e-2)).all()
    assert (np.isclose(stats["std_dev"], 1.0, atol=1e-2)).all()

    # Min-max normalization
    input_layer_2 = StandardizationLayer("x", n_features=1, stats_path=None)
    input_layer_1.kind = "minmax"
    for x, y in data_loader:
        x = input_layer_1(x)
        input_layer_2(x)
    input_layer_2.epoch_finished()

    stats = input_layer_2.compute_stats()
    assert (np.isclose(stats["min"], -1.0, atol=1e-2)).all()
    assert (np.isclose(stats["max"], 1.0, atol=1e-2)).all()


def data_loader_3d(n_samples: int, batch_size: int) -> DataLoader:
    """
    Create DataLoader providing batchs of Synthetic3D data.

    Args:
        n_samples: The number of samples in the dataset.
        batch_size: The size of the batches into which to combine the inputs.

    Return:
        A 'torch.utils.data.Dataloader' providing access to the training
        batches.
    """
    data = Synthetic3d(n_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def test_input_3d(tmp_path):
    """
    Test calculation of input statistics for 3D input.
    """
    data_loader = data_loader_3d(256, 4)

    input_layer = InputLayer("x", n_features=4, stats_path=tmp_path)
    for x, y in data_loader:
        input_layer(x)
    stats_1 = input_layer.compute_stats()

    input_layer = InputLayer("x", n_features=4, stats_path=tmp_path)
    for x, y in data_loader:
        mask = torch.rand(*x.shape) < 0.5
        x[mask] = torch.nan
        input_layer(x)

    stats_2 = input_layer.compute_stats()

    assert np.isclose(stats_1["std_dev"], stats_2["std_dev"], rtol=0.05).all()


def test_hist_3d(tmp_path):
    """
    Test calculation of histograms for 3D (spectral + spatial) input data and
    ensure that:
        - The number of bin arrays matches the number of input channels of the
          data.
    """
    data_loader = data_loader_3d(256, 4)

    input_layer = InputLayer("x", n_features=4, stats_path=tmp_path)
    for x, y in data_loader:
        input_layer(x)
    input_layer.epoch_finished()

    for x, y in data_loader:
        input_layer(x)
    input_layer.epoch_finished()

    assert len(input_layer.bins) == input_layer.n_features


def test_normalization_3d(tmp_path):
    """
    Perform EDA on synthetic dataset and use input layers to normalize inputs.
    Ensure that:
        - Mean and std. dev. of standardized inputs are close to 0
          and 1, respectively.
        - Minimum and maximum of min-max normalized inputs are close to -1.0
          and 1.0, respectively.
    """
    data_loader = data_loader_3d(256, 8)

    input_layer_1 = StandardizationLayer(
        "x", n_features=4, stats_path=tmp_path, kind="standardize"
    )
    for x, y in data_loader:
        input_layer_1(x)
    input_layer_1.epoch_finished()
    for x, y in data_loader:
        input_layer_1(x)
    input_layer_1.epoch_finished()

    # Standardization
    input_layer_2 = StandardizationLayer("x", n_features=4, stats_path=None)
    for x, y in data_loader:
        x = input_layer_1(x)
        input_layer_2(x)
    input_layer_2.epoch_finished()

    stats = input_layer_2.compute_stats()
    assert (np.isclose(stats["mean"], 0.0, atol=1e-2)).all()
    assert (np.isclose(stats["std_dev"], 1.0, atol=1e-2)).all()

    # Min-max normalization
    input_layer_2 = StandardizationLayer("x", n_features=4, stats_path=None)
    input_layer_1.kind = "minmax"
    for x, y in data_loader:
        x = input_layer_1(x)
        input_layer_2(x)
    input_layer_2.epoch_finished()

    stats = input_layer_2.compute_stats()
    assert (np.isclose(stats["min"], -1.0, atol=1e-2)).all()
    assert (np.isclose(stats["max"], 1.0, atol=1e-2)).all()


def test_load_input_modules(
        encoder_decoder_model_config_file,
        encoder_decoder_training_config_file,
        tmp_path
):
    """
    Calculate input statstics. Save model with statistics and ensure that:
        - Input data statistics are correctly loaded
    """
    data_loader = data_loader_3d(256, 8)

    input_layer = StandardizationLayer(
        "x", n_features=4, stats_path=tmp_path, kind="standardize"
    )
    for x, y in data_loader:
        input_layer(x)
    input_layer.epoch_finished()
    for x, y in data_loader:
        input_layer(x)
    input_layer.epoch_finished()

    model = load_and_compile_model(encoder_decoder_model_config_file)
    model.config_dict["input"]["x"]["normalize"] = "minmax"
    model.stems["x"].insert(0, input_layer)

    model.save(tmp_path / "model.pt")

    model_loaded = load_model(tmp_path / "model.pt")

    assert (model_loaded.stems["x"][0].t_min == model.stems["x"][0].t_min).all()
    assert (model_loaded.stems["x"][0].t_max == model.stems["x"][0].t_max).all()
    assert (model_loaded.stems["x"][0].t_mean == model.stems["x"][0].t_mean).all()


def test_load_stats(tmp_path):
    """
    Test explicit loading of stats.
    """
    data_loader = data_loader_1d(2048, 64)

    input_layer_1 = InputLayer("x", n_features=1, stats_path=tmp_path / "stats")
    for x, y in data_loader:
        input_layer_1(x)
    input_layer_1.epoch_finished()

    for x, y in data_loader:
        input_layer_1(x)
    input_layer_1.epoch_finished()

    stats = input_layer_1.compute_stats()
    save_stats(stats, tmp_path / "hidden_stats" / "input", "x")
    assert (tmp_path / "hidden_stats" / "input" / "x.nc").exists()

    input_layer_2 = InputLayer("x", n_features=1, stats_path=tmp_path / "stats")
    assert not input_layer_2.finalized
    input_layer_2.load_stats(tmp_path / "hidden_stats" / "input" / "x.nc")
    assert input_layer_2.finalized
