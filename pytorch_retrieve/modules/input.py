"""
pytorch_retrieve.module.input
=============================

Provides input modules to normalize inputs and record data statistics.
"""
from typing import Dict, Optional
from logging import getLogger
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

from pytorch_retrieve.tensors import MaskedTensor
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only

import xarray as xr

from .stats import StatsTracker, save_stats


LOGGER = getLogger(__name__)


def _ensure_loaded(tensor: Union[torch.Tensor, "_NotYetLoadedTensor"]) -> torch.Tensor:
    """
    Ensures that a given tensor is loaded and not a _NotYetLoadedTensor.

    Args:
        tensor: A tensor or '_NotYetLoadedTensor'.

    Return:
        A tensor containing the given input data.
    """
    if tensor.__class__.__name__ == "_NotYetLoadedTensor":
        tensor = tensor._load_tensor()
    return tensor


class InputLayer(StatsTracker, nn.Module):
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
        stats_path: Optional[Path] = None,
    ):
        """
        Args:
            name: The name of the input.
            n_features: The number of features in the input data.
            n_bins: The number of bins to use for the input data histograms.
            stats_path: Path in which model-related files are stored.
        """
        StatsTracker.__init__(self, n_features, n_bins=n_bins)
        nn.Module.__init__(self)

        if name is None:
            name = "input"
        self.name = name
        self.n_features = n_features
        self.n_bins = n_bins

        self.initialize()

        if stats_path is None:
            stats_path = Path("stats")
        self.stats_path = stats_path
        stats_file = stats_path / "input" / f"{self.name}.nc"
        if stats_file.exists():
            with xr.open_dataset(stats_file) as dataset:
                self._load_stats_tensors(dataset)

    def load_stats(self, stats_file: Union[Path, str]) -> None:
        """
        Load input data statistics from stats file.

        Args:
            stats_file: A path object pointing to a stats file.
        """
        with xr.open_dataset(stats_file) as dataset:
            self._load_stats_tensors(dataset)

    def initialize(self) -> None:
        """
        Initializes attributes used to track input data statistics.
        """
        StatsTracker.initialize(self)

        # Initialize torch parameters.
        self.finalized = torch.tensor(0.0, dtype=torch.float32)
        self.t_mean = torch.zeros(self.n_features, dtype=torch.float32)
        self.t_std_dev = torch.zeros(self.n_features, dtype=torch.float32)
        self.t_min = torch.zeros(self.n_features, dtype=torch.float32)
        self.t_max = torch.zeros(self.n_features, dtype=torch.float32)

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

        StatsTracker.track_stats(self, x)

    def finalize(self, lightning_module: Optional[LightningModule] = None) -> None:
        """
        Finalize input layer.
        """
        stats = self.compute_stats()
        self._load_stats_tensors(stats)

    def epoch_finished(
        self,
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

    def get_extra_state(self) -> Dict[str, torch.Tensor]:
        """
        Get mean, std. dev., min. and max tensors stored in layer.

        Args:
             state: A dictionary containing the extra state obtained from 'get_extra_state'.
        """
        return {
            "mean": self.t_mean,
            "std_dev": self.t_std_dev,
            "min": self.t_min,
            "max": self.t_max,
            "finalized": self.finalized,
        }

    def set_extra_state(self, state) -> None:
        """
        Set mean, std. dev., min. and max tensors stored in layer.

        Args:
             state: A dictionary containing the extra state obtained from 'get_extra_state'.
        """
        self.t_mean.set_(_ensure_loaded(state["mean"]))
        self.t_std_dev.set_(_ensure_loaded(state["std_dev"]))
        self.t_min.set_(_ensure_loaded(state["min"]))
        self.t_max.set_(_ensure_loaded(state["max"]))
        self.finalized.set_(_ensure_loaded(state["finalized"]))

    def _load_stats_tensors(self, dataset: xr.Dataset) -> None:
        """
        Load tensors with input data statistics from dataset.
        """
        self.t_mean = torch.tensor(dataset["mean"].data.astype("float32"))
        self.t_std_dev = torch.tensor(dataset["std_dev"].data.astype("float32"))
        self.t_min = torch.tensor(dataset["min"].data.astype("float32"))
        self.t_max = torch.tensor(dataset["max"].data.astype("float32"))
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
        stats_path: Optional[Path] = None,
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
        super().__init__(name, n_features, n_bins=n_bins, stats_path=stats_path)
        self.kind = kind
        if sentinel is None:
            if kind == "standardize":
                sentinel = -6.0
            else:
                sentinel = -1.5

        self.sentinel = sentinel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply standardization to input if module is finalized.

        Args:
             x: The tensor containing the input data.

        Return:
             The standardized tensor.
        """
        pad_dims = x.dim() - 2

        if self.finalized > 0.0:
            if self.kind == "standardize":
                mean = self.t_mean.to(device=x.device, dtype=x.dtype).__getitem__(
                    (...,) + (None,) * pad_dims
                )
                std_dev = self.t_std_dev.to(device=x.device, dtype=x.dtype).__getitem__(
                    (...,) + (None,) * pad_dims
                )
                x_n = (x - mean) / std_dev
            elif self.kind == "minmax":
                mins = self.t_min.to(device=x.device, dtype=x.dtype).__getitem__(
                    (...,) + (None,) * pad_dims
                )
                maxs = self.t_max.to(device=x.device, dtype=x.dtype).__getitem__(
                    (...,) + (None,) * pad_dims
                )
                x_n = -1.0 + 2.0 * (x - mins) / (maxs - mins)
            elif self.kind == "atanh":
                mins = self.t_min.to(device=x.device, dtype=x.dtype).__getitem__(
                    (...,) + (None,) * pad_dims
                )
                maxs = self.t_max.to(device=x.device, dtype=x.dtype).__getitem__(
                    (...,) + (None,) * pad_dims
                )
                masked = mins == maxs
                masked = torch.broadcast_to(masked[None], x.shape)
                x_n = -2.0 + 4.0 * (x - mins) / (maxs - mins)
                x_n = torch.tanh(x_n)
                mean = torch.zeros(masked.sum()).to(device=x.device, dtype=x.dtype)
                #x_n = torch.where(masked, torch.normal(mean=torch.zeros_like(x)), x_n)

            # Replace NANs
            if torch.isnan(x_n).any():
                mask = torch.isfinite(x_n)
                x_n = torch.where(mask, x_n, self.sentinel)

            return x_n

        return super().forward(x)
