"""
Tests for the pytorch_retrieve.modules.transformations module
"""
import pytest
import lightning as L
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from pytorch_retrieve.config import InputConfig, OutputConfig
from pytorch_retrieve.eda import EDAModule
from pytorch_retrieve.tensors import MaskedTensor
from pytorch_retrieve.modules.transformations import (
    SquareRoot,
    CubeRoot,
    Log,
    LogLinear,
    MinMax,
    HistEqual
)

@pytest.mark.parametrize("transformation", [SquareRoot(), CubeRoot(), Log(), LogLinear(), MinMax(100.0, 200.0)])
def test_transformations(transformation):

    x_ref = 1e3 * torch.rand(10, 10, 10) + 1e-3
    y = transformation(x_ref)
    x = transformation.invert(y)

    assert torch.isclose(x, x_ref, atol=1e-4).all()

    x_ref = MaskedTensor(x_ref, mask=torch.rand(x_ref.shape) > 0.5)
    y = transformation(x_ref)

    assert isinstance(y, MaskedTensor)
    assert (x_ref.mask == y.mask).all()


def test_hist_equal(monkeypatch, tmp_path):
    """


    """
    monkeypatch.chdir(tmp_path)

    x = torch.rand(10 * 1024, 4)
    y = 10.0 * (torch.rand((10 * 1024, 4)) + torch.arange(4)[None])

    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=1024)

    input_configs = {
        "x": InputConfig(n_features=4)
    }
    output_configs = {
        "y": OutputConfig("y", kind="Mean", shape=(4,))
    }

    stats_path = tmp_path / "stats"
    stats_path.mkdir()

    # Run EDA
    eda_module = EDAModule(input_configs, output_configs, stats_path=stats_path)
    trainer = L.Trainer(
        max_epochs=2,
        logger=None,
        precision=32,
        accelerator="cpu",
        devices=1
    )
    trainer.fit(
        eda_module,
        train_dataloaders=dl,
    )

    transformation = HistEqual(256, output_configs["y"])

    y_t = transformation(y)
    assert y_t.min() == -1
    assert y_t.max() == 1

    y_r = transformation.invert(y_t)

    assert y_r.min() == y.min()
    assert y_r.max() == y.max()
    assert torch.isclose(y_r, y, rtol=1e-2).all()

    bins = transformation.bins
    centers = 0.5 * (bins[1:] + bins[:-1])
    y_t = transformation(centers)
    centers = transformation.invert(y_t)
    assert torch.isclose(centers, 0.5 * (bins[1:] + bins[:-1])).all()
