"""
pytorch_retrieve.modules.stats
==============================

Defines modules for tracking basic statistics of model input and
output data.
"""

from pathlib import Path
from typing import Dict, Optional


from lightning.pytorch.utilities import rank_zero_only
from lightning import LightningModule
import numpy as np
import torch
from torch import nn
import xarray as xr


from pytorch_retrieve.tensors import MaskedTensor


@rank_zero_only
def save_stats(dataset: xr.Dataset, stats_path: Path, name: str) -> None:
    """
    Save statistics in NetCDF4 format

    Args:
        dataset: The xarray.Dataset containing the input data statistics.
        stats_path: The path to which to write the statistics.
        name: The name of the file.
    """
    stats_path.mkdir(exist_ok=True, parents=True)
    dataset.to_netcdf(stats_path / (name + ".nc"))


class StatsTracker:
    """
    A StatsTracker is used to track the statistics of specific tensor during
    EDA.
    """

    def __init__(
        self,
        n_features: int,
        n_bins: int = 1024,
    ):
        """
        Args:
            name: A name identifying the data to track. The name will also
                be used to store the recorded statistics.
            n_features: The number of features (channels) in the  data.
            n_bins: The number of bins to use for the input data histograms.
        """
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.initialize()

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

    def reset(self) -> None:
        """
        Reset state of input layer.
        """
        self.initialize()

    def track_stats(self, x: torch.Tensor) -> None:
        """
        Track stats of given tensor.

        This function should be called for every batch in an epoch. The StatsTacker
        object will keep track of the aggergate statistics of the dataset.

        Args:
             x: The tensors containing the data to track.
        """
        if isinstance(x, list):
            for x_i in x:
                self.track_stats(x_i)
            return None

        if not isinstance(x, MaskedTensor):
            mask = torch.isnan(x)
            x = MaskedTensor(x, mask=mask)

        if x.ndim in [1, 3]:
            x = x.unsqueeze(1)

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

    def compute_stats(
        self, lightning_module: Optional[LightningModule] = None
    ) -> xr.Dataset:
        """
        Compute summary statistics of the tracked data.

        Args:
            lightning_module: An optional lightning module object to use to
                gather data from multiple processes.

        Return:
            A xarray.Dataset containing the mean, standard deviation,
            covariance matrix, minimum and maxium and histograms
            for tracked features.
        """
        if self.x is None:
            raise ValueError(
                f"The input layer '{self.name}' has not yet processed any input data."
            )

        if lightning_module is not None and lightning_module.device != torch.device(
            "cpu"
        ):
            x = lightning_module.all_gather(self.x).sum(0).cpu().numpy()
            xx = lightning_module.all_gather(self.xx).sum(0).cpu().numpy()
            counts = lightning_module.all_gather(self.counts).sum(0).cpu().numpy()
            min_vals = lightning_module.all_gather(self.min_vals).sum(0).cpu().numpy()
            max_vals = lightning_module.all_gather(self.max_vals).sum(0).cpu().numpy()
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

        stats = {
            "mean": mean,
            "std_dev": std_dev,
            "cov": cov,
            "corr": corr,
            "min": min_vals,
            "max": max_vals,
        }

        dims = ("features", "features_")
        dataset = xr.Dataset(
            {name: (dims[: data.ndim], data) for name, data in stats.items()}
        )

        if len(self.bins) > 0:
            boundaries = np.stack(list(self.bins.values()))
            counts = np.stack(list(self.hists.values()))
            dataset["bin_boundaries"] = (("features", "bin_boundaries"), boundaries)
            dataset["counts"] = (("features", "bins"), counts)
        return dataset

    def epoch_finished(self) -> None:
        """
        Signal processing of an epoch of data has finished.
        """
        self.initialized = True
