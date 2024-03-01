"""
pytorch_retrieve.modules.output
===============================

"""
from typing import Tuple, Union

import torch
from torch import nn

from pytorch_retrieve.tensors import MeanTensor, QuantileTensor
from . import stats


class Mean(stats.StatsTracker, nn.Module):
    """
    This output module transforms a tensor into an output tensor representing
    the conditional mean of a prediction.
    """
    def __init__(
            self,
            name,
            shape: Union[int, Tuple[int]],
    ):
        nn.Module.__init__(self)
        if isinstance(shape, int):
            n_features = shape
        else:
            n_features = shape[0]
        stats.StatsTracker.__init__(self, n_features)


    def forward(self, x):
        return MeanTensor(x)


class Quantiles(stats.StatsTracker, nn.Module):
    """
    The Quantiles layer produces tensors that represent probabilistic regression
    outputs.
    """
    def __init__(
            self,
            name: str,
            shape: Union[int, Tuple[int]],
            tau: torch.Tensor,
    ):
        """
        Create an output layer for quantile regression.

        Args:
            tau: A vector containing the quantile fractions to predict.
        """

        nn.Module.__init__(self)
        if isinstance(shape, int):
            n_features = shape
        else:
            n_features = shape[0]
        stats.StatsTracker.__init__(self, n_features)
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        """
        Produces a QuantileTensor from a model output.
        """
        return QuantileTensor(x, tau=self.tau)
