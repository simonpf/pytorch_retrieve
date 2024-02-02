"""
pytorch_retrieve.module.input
=============================

Provides input modules to normalize inputs and record data statistics.
"""
from typing import Dict, Optional
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pytorch_retrieve.tensors import MaskedTensor
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only

import xarray as xr


LOGGER = getLogger(__name__)


@rank_zero_only
def save_stats(dataset: xr.Dataset, model_path: Path, input_name: str) -> None:
    """
    Save statistics to disk.

    Args:
        dataset: The xarray.Dataset containing the input data statistics.
        model_path: The model path to which to write the calculated
            statistics.
        input_name: The name of the input.
    """
    stats_dir = model_path / "stats"
    stats_dir.mkdir(exist_ok=True, parents=True)
    dataset.to_netcdf(stats_dir / (input_name + ".nc"))


class InputLayer(nn.Module):
    """
    Base class for input layers. Input layers track input-data statistics
    and normalize input tensors.

    Prior to using an input layer in a neural network, it must first be used
    to record input data statistics. Subsequently, the recorded input data
    statistics are used to normalize the input data.
    """

    def __init__(
        self,
        name: str,
        n_features: int,
        n_bins: int = 100,
        model_path: Optional[Path] = None,
    ):
        """
        Args:
            name: The name of the input.
            n_features: The number of features in the input data.
            n_bins: The number of bins to use for the input data histograms.
            model_path: Path in which model-related files are stored.
        """
        super().__init__()

        if name is None:
            name = "input"
        self.name = name
        self.n_features = n_features
        self.n_bins = n_bins

        self.initialize()

        if model_path is None:
            model_path = Path(".")
        self.model_path = model_path
        stats_path = model_path / "stats" / (name + ".nc")
        if stats_path.exists():
            with xr.open_dataset(stats_path) as dataset:
                self._load_stats_tensors(dataset)

    def initialize(self) -> None:
        """
        Initializes attributes used to track input data statistics.
        """
        self.x = None
        self.xx = None
        self.counts = None
        self.min_vals = None
        self.max_vals = None

        self.initialized = False

        self.hists = {}
        self.bins = {}

        # Initialize torch parameters.
        self.finalized = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        self.p_mean = nn.Parameter(torch.zeros(self.n_features, dtype=torch.float32), requires_grad=False)
        self.p_std_dev = nn.Parameter(torch.zeros(self.n_features, dtype=torch.float32), requires_grad=False)
        self.p_min = nn.Parameter(torch.zeros(self.n_features, dtype=torch.float32), requires_grad=False)
        self.p_max = nn.Parameter(torch.zeros(self.n_features, dtype=torch.float32), requires_grad=False)

    def reset(self):
        """
        Reset state of input layer.
        """
        self.initialize()

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Propagate input tensor through layer.

        If the input layer state is not yet finalized, this method will
        record the input data statistics and return None. After two
        epochs of training data have been passed through the layer, it will
        be finalized and the input layer simply returns it inpu.

        Args:
             x: A torch.Tensor to propagate through the network.

        Return:
             None in the EDA phase, when the input layer is not yet finalized.
             The input tensor when the layer is finalized.
        """
        if self.finalized.item():
            return x

        if not isinstance(x, MaskedTensor):
            mask = torch.isnan(x)
            x = MaskedTensor(x, mask=mask)

        # 1st epoch: record basic statistics
        if not self.initialized:
            x = torch.transpose(x, 1, -1)
            xx = x[..., None] * x[..., None, :]
            dims = tuple(range(x.dim() - 1))

            if self.x is None:
                self.x = x.sum(dims).detach()
                self.xx = xx.sum(dims).detach()
                self.counts = (~x.mask).to(dtype=torch.float32).sum(dims)

                min_vals = [
                    torch.select(x, -1, ind).min() for ind in range(self.n_features)
                ]
                self.min_vals = torch.tensor(min_vals)
                max_vals = [
                    torch.select(x, -1, ind).max() for ind in range(self.n_features)
                ]
                self.max_vals = torch.tensor(max_vals)
            else:
                self.x += x.sum(dims).detach()
                self.xx += xx.sum(dims).detach()
                self.counts += (~x.mask).to(torch.float32).sum(dims)
                min_vals = [
                    torch.select(x, -1, ind).min() for ind in range(self.n_features)
                ]
                self.min_vals = torch.minimum(self.min_vals, torch.tensor(min_vals))
                max_vals = [
                    torch.select(x, -1, ind).max() for ind in range(self.n_features)
                ]
                self.max_vals = torch.maximum(self.max_vals, torch.tensor(max_vals))

        # 2nd epoch: calculate histograms.
        else:
            if len(self.bins) == 0:
                for ind in range(self.n_features):
                    self.bins[ind] = torch.linspace(
                        self.min_vals[ind], self.max_vals[ind], self.n_bins + 1
                    )

            x_nan = torch.transpose(torch.where(x.mask, torch.nan, x.base), 1, -1)
            for ind in range(self.n_features):
                x_nan = x_nan.to("cpu", torch.float32)

                hist = self.hists.setdefault(
                    ind,
                    torch.zeros(self.n_bins, dtype=x_nan.dtype, device=x_nan.device),
                )
                hist += torch.histogram(torch.select(x_nan, -1, ind), self.bins[ind])[0]

        return None

    def compute_stats(
        self, lightning_module: Optional[LightningModule] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute input data statistics.

        Args:
            lightning_module: An optional lightning module object to use to
                gather data from multiple processes.

        Return:
            A dictionary containing the mean, standard deviation, covariance matrix
            and minimum and maxium values for all features in the input.
        """
        if self.x is None:
            raise ValueError(
                f"The input layer '{self.name}' has not yet processed any input data."
            )
        if lightning_module is not None:
            x = lightning_module.all_gather(x).sum(0).cpu().numpy()
            xx = lightning_module.all_gather(xx).sum(0).cpu().numpy()
            counts = lightning_module.all_gather(counts).sum(0).cpu().numpy()
            min_vals = lightning_module.all_gather(min_vals).sum(0).cpu().numpy()
            max_vals = lightning_module.all_gather(max_vals).sum(0).cpu().numpy()
        else:
            x = self.x.cpu().numpy()
            xx = self.xx.cpu().numpy()
            counts = self.counts.cpu().numpy()
            min_vals = self.min_vals.cpu().numpy()
            max_vals = self.max_vals.cpu().numpy()

        mean = x / counts
        cov = xx / counts - (mean[None] * mean[..., None])
        std_dev = np.sqrt(cov.diagonal())
        corr = cov / (std_dev[None] * std_dev[..., None])

        return {
            "mean": mean,
            "std_dev": std_dev,
            "cov": cov,
            "corr": corr,
            "min": min_vals,
            "max": max_vals,
        }

    def epoch_finished(
        self, lightning_module: Optional[LightningModule] = None
    ) -> None:
        """
        Signal processing of an epoch of data has finished.
        """
        if self.initialized:
            device = self.finalized.device
            self.finalized.set_(torch.tensor(1.0).to(device=device))
            self.finalize()
            return None
        self.initialized = True

    def finalize(self, lightning_module: Optional[LightningModule] = None) -> None:
        """
        Finalize input layer.
        """
        stats = self.compute_stats(lightning_module)
        dims = ("features", "features_")
        dataset = xr.Dataset(
            {name: (dims[: data.ndim], data) for name, data in stats.items()}
        )

        boundaries = np.stack(list(self.bins.values()))
        counts = np.stack(list(self.hists.values()))

        dataset["bin_boundaries"] = (("features", "bin_boundaries"), boundaries)
        dataset["counts"] = (("features", "bins"), counts)

        save_stats(dataset, self.model_path, self.name)
        self._load_stats_tensors(dataset)

    def _load_stats_tensors(self, dataset: xr.Dataset) -> None:
        """
        Load tensors with input data statistics from dataset.
        """
        self.p_mean = nn.Parameter(
            torch.tensor(dataset["mean"].data.astype("float32")), requires_grad=False
        )
        self.p_std_dev = nn.Parameter(
            torch.tensor(dataset["std_dev"].data.astype("float32")), requires_grad=False
        )
        self.p_min = nn.Parameter(
            torch.tensor(dataset["min"].data.astype("float32")), requires_grad=False
        )
        self.p_max = nn.Parameter(
            torch.tensor(dataset["max"].data.astype("float32")), requires_grad=False
        )
        device = self.finalized.device
        self.finalized.set_(torch.tensor(1.0, dtype=torch.float32).to(device=device))


class StandardizationLayer(InputLayer):
    """
    An input layer that performs input feature normalization based on
    statistics recorded from the training data.
    """

    def __init__(
        self,
        name,
        n_features: int,
        n_bins: int = 100,
        model_path: Optional[Path] = None,
        kind: str = "minmax",
        sentinel: Optional[float] = None,
    ):
        """
        Args:
            n_features: The number of features in the input data.
            n_bins: The number of bins to use for the input data histograms.
            kind: The type of normalization to perform: 'standard' or
                'minmax' normalization.
            sentinel: Fixed value to replace NANs with. If not provided, the
                the mean value will be used.
        """
        super().__init__(name, n_features, n_bins=n_bins, model_path=model_path)
        self.kind = kind
        if sentinel is None:
            if kind == "standardize":
                sentinel = -6.0
            else:
                sentinel = -1.5

        self.sentinel = sentinel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_dims = x.dim() - 2

        if self.finalized.item() > 0.0:
            if self.kind == "standardize":
                mean = self.p_mean.__getitem__((...,) + (None,) * pad_dims)
                std_dev = self.p_std_dev.__getitem__((...,) + (None,) * pad_dims)
                x_n = (x - mean) / std_dev
            elif self.kind == "minmax":
                mins = self.p_min.__getitem__((...,) + (None,) * pad_dims)
                maxs = self.p_max.__getitem__((...,) + (None,) * pad_dims)
                x_n = -1.0 + 2.0 * (x - mins) / (maxs - mins)

            # Replace NANs
            if torch.isnan(x_n).any():
                mask = torch.isfinite(x_n)
                x_n = torch.where(mask, x_n, self.sentinel)
            return x_n

        return super().forward(x)
