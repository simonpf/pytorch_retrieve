"""
Tests for the pytorch_retrieve.modules.mlp module.
==================================================
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning as L

from pytorch_retrieve.modules.output import Mean
from pytorch_retrieve.lightning import LightningRetrieval
from pytorch_retrieve.data.synthetic import Synthetic1d
from pytorch_retrieve.modules.mlp import MLP
from pytorch_retrieve import config


def data_loader_1d(n_samples: int, batch_size: int) -> DataLoader:
    """
    A DataLoader providing batch of Synthetic1D data.
    """
    data = Synthetic1d(n_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def run_training(model: nn.Module) -> None:
    """
    Runs one training epoch on the given neural network model.

    """
    with TemporaryDirectory() as tmp:
        mod = LightningRetrieval(model, model_dir=Path(tmp))
        data_loader = data_loader_1d(256, 32)
        trainer = L.Trainer(max_epochs=1, accelerator="cpu")
        trainer.fit(mod, train_dataloaders=data_loader)


def test_mlp_no_residuals():
    """
    Create MLP without residual connections and ensure that
       - tensors can be forwarded through the MLP
       - training on tabular data works
       - the network has the expected number of parameters.
    """
    data_loader = data_loader_1d(1024, 32)
    x, y = next(iter(data_loader))

    mlp = MLP(1, 16, 4, hidden_channels=8)
    n_params = (8 + 8) + (8 * 8 + 8) * 2 + (8 * 16 + 16)
    assert mlp.n_params == n_params
    y_pred = mlp(x)
    assert y_pred.shape == (x.shape[:1] + (16,))
    model = nn.Sequential(mlp, Mean("y", 1))
    run_training(model)


def test_mlp_simple_residuals():
    """
    Create MLP with simple residual connections and ensure that
       - tensors can be forwarded through the MLP
       - training on tabular data works
       - the network has the expected number of parameters.
    """
    data_loader = data_loader_1d(1024, 32)
    x, y = next(iter(data_loader))

    mlp = MLP(1, 16, 4, hidden_channels=8, residual_connections="simple")
    n_params = (8 + 8) + (8 * 8 + 8) * 2 + (8 * 16 + 16)
    assert mlp.n_params == n_params
    y_pred = mlp(x)
    assert y_pred.shape == (x.shape[:1] + (16,))
    model = nn.Sequential(mlp, Mean("y", 1))
    run_training(model)


def test_mlp_hyper_residuals():
    """
    Create MLP with hyper/dense residual connections and ensure that
       - tensors can be forwarded through the MLP
       - training on tabular data works
       - the network has the expected number of parameters.
    """
    data_loader = data_loader_1d(1024, 32)
    x, y = next(iter(data_loader))

    mlp = MLP(1, 16, 4, hidden_channels=8, residual_connections="hyper")
    n_params = (8 + 8) + (8 * 8 + 8) * 2 + (8 * 16 + 16)
    assert mlp.n_params == n_params
    y_pred = mlp(x)
    assert y_pred.shape == (x.shape[:1] + (16,))
    model = nn.Sequential(mlp, Mean("y", 1))
    run_training(model)
