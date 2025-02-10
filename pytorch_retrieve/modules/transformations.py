"""
pytorch_retrieve.modules.transformations
========================================

Defines transfomations modules to be used in output modules.
"""
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
import xarray as xr

from pytorch_retrieve.modules.input import _ensure_loaded


class SquareRoot(nn.Module):
    """
    Square root transformation.

    This transformation can be used in an output layer to predict the square root of the
    target quantity instead of the original quantity.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.sqrt(x)

    def invert(self, x: torch.Tensor):
        return x ** 2

class CubeRoot(nn.Module):
    """
    Cube root transformation.

    This transformation can be used in an output layer to predict the cube root of the
    target quantity instead of the original quantity.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return torch.pow(x, 1/3)

    def invert(self, x: torch.Tensor):
        return x ** 3


class Log(nn.Module):
    """
    Log transformation.

    This transformation can be used in an output layer to predict the logarithm of the
    target quantity instead of the original quantity.
    """
    def __init__(self, threshold=1e-3, **kwargs):
        super().__init__()
        self.threshold = torch.tensor(threshold)

    def forward(self, x):
        dtype = x.dtype
        device = x.device
        threshold = self.threshold.to(dtype=dtype, device=device)
        x[x < threshold] = threshold
        x = x.to(dtype=torch.float32)
        return torch.log(x).to(dtype=dtype)

    def invert(self, x: torch.Tensor):
        dtype = x.dtype
        device = x.device
        threshold = self.threshold.to(dtype=dtype, device=device)
        inverted = torch.exp(x.to(torch.float32)).to(dtype)
        zero = inverted < threshold
        inverted[zero] = 0.0
        return inverted


class LogLinear(nn.Module):
    """
    Log transformation.

    This transformation can be used in an output layer to predict the logarithm of the
    target quantity instead of the original quantity.
    """
    def __init__(self, threshold: float = 1e-3, **kwargs):
        super().__init__()
        self.threshold = torch.tensor(threshold)

    def forward(self, x):
        dtype = x.dtype
        device = x.device
        threshold = self.threshold.to(dtype=dtype, device=device)
        x[x < threshold] = threshold
        x = x.to(dtype=torch.float32)
        return torch.where(x > 1, x - 1, torch.log(x).to(dtype=dtype))


    def invert(self, x: torch.Tensor):
        dtype = x.dtype
        return torch.where(x > 0, x + 1, torch.exp(x).to(dtype=dtype))


class MinMax(nn.Module):
    """
    Min-max transformation.

    Linearly map the range -1, 1 to the range [min, max]
    """
    def __init__(self, x_min: float, x_max: float, **kwargs):
        super().__init__()
        self.x_min = torch.tensor(x_min)
        self.x_max = torch.tensor(x_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        x_min = self.x_min.to(device=device, dtype=dtype)
        x_max = self.x_max.to(device=device, dtype=dtype)
        return (x - x_min) / (x_max - x_min) * 2.0 - 1.0

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        x_min = self.x_min.to(device=device, dtype=dtype)
        x_max = self.x_max.to(device=device, dtype=dtype)
        return x_min + (x_max - x_min) * (0.5 * x + 0.5)


class HistEqual(nn.Module):
    """
    Applies histogram equalization to a given output so that the training data distribution
    of output values is mapped to a uniform distribution over [-1, 1].
    """
    def __init__(
            self,
            n_steps: int,
            output_config: "pytorch_retrieve.config.OutputConfig"
    ):
        super().__init__()
        self.output_name = output_config.target
        self.n_steps = n_steps
        self.bins = torch.nan * torch.zeros((n_steps,))

        stats_path = Path(".") / "stats" / "output" / (self.output_name + ".nc")
        if stats_path.exists():
            self.load_stats(stats_path)


    def load_stats(self, stats_path: Path):
        """
        Load histogram statistics and initialize bin attribute.

        Args:
            stats_path: A Path object pointing to the statistics file.
        """
        tot_counts = np.zeros(self.n_steps)

        with xr.open_dataset(stats_path) as stats:

            bin_min = stats["min"].data.min()
            bin_max = stats["max"].data.max()
            new_bins = np.linspace(bin_min, bin_max, self.n_steps + 1)
            cntr = 0.5 * (new_bins[1:] + new_bins[:-1])

            for feature in range(stats.features.size):
                x_f = stats.bin_boundaries[feature]
                x_f = 0.5 * (x_f[1:] + x_f[:-1])
                y_f = stats.counts[feature]
                y_i = np.interp(cntr, x_f, y_f, left=y_f[0], right=y_f[-1])
                tot_counts += y_i

            x_i = np.linspace(-1, 1, self.n_steps)
            x_i = np.concatenate([[0.0], np.flip(1.0 - np.logspace(-6, 0, self.n_steps - 1)[:-1]), [1.0]])
            x_i = 2.0 * x_i - 1.0

            x_f = np.concatenate([np.array([0.0]), np.cumsum(tot_counts)])
            x_f /= x_f[-1]
            x_f = x_f * 2 - 1.0
            bins = np.interp(x_i, x_f, new_bins, left=new_bins[0], right=new_bins[-1])

        self.bins.set_(torch.tensor(bins.astype(np.float32)))
        incs = torch.diff(self.bins)
        self.incs = torch.cat([incs, incs[-1:]])


    def get_extra_state(self) -> Dict[str, torch.Tensor]:
        """
        Store bins in state dict.
        """
        return {
            "bins": self.bins,
        }

    def set_extra_state(self, state) -> None:
        """
        Set bins from state dict.
        """
        self.bins = _ensure_loaded(state["bins"])
        incs = torch.diff(self.bins)
        self.incs = torch.cat([incs, incs[-1:]])


    def forward(self, y) -> torch.Tensor:
        """
        Transform output values to range [-1, 1]
        """
        invalid = torch.isnan(y)

        bins = self.bins.to(device=y.device, dtype=y.dtype)
        incs = self.incs.to(device=y.device, dtype=y.dtype)

        inds = torch.clamp(torch.bucketize(y, bins, out_int32=True, right=True) - 1, min=0, max=self.n_steps - 1)
        delta = torch.clamp((y - bins[inds]) / incs[inds], 0.0, 1.0)
        inds = inds.to(dtype=y.dtype) + delta
        y_q = 2.0 * inds / (self.n_steps - 1) - 1.0
        y_q[invalid] = torch.nan
        return y_q


    def invert(self, x) -> torch.Tensor:
        """
        Map input values from [-1, 1] to original range.
        """
        bins = self.bins.to(device=x.device, dtype=x.dtype)
        incs = self.incs.to(device=x.device, dtype=x.dtype)

        y = torch.clamp((x + 1.0) / 2.0 * (self.n_steps - 1), 0, self.n_steps - 1)
        inds = torch.trunc(y).to(dtype=torch.int32)
        d_y = (y - inds) * incs[inds]
        d_y[self.n_steps - 1 <= inds] = 0.0

        return bins[inds] + d_y
