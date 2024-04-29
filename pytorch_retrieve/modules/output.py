"""
pytorch_retrieve.modules.output
===============================

"""
from typing import Optional, Union, Tuple

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
            transformation: Optional[nn.Module] = None
    ):
        """
        Args:
            name: The name of the output produced by this output layer.
            shape: The shape of the target quantity for a single input pixel.
            transformation: An optional transformation module that will be applied to the
                reference values prior to calculating the losses.
        """
        self.name = name
        nn.Module.__init__(self)
        if isinstance(shape, int):
            n_features = shape
        else:
            n_features = shape[0]
        stats.StatsTracker.__init__(self, n_features)
        self.transformation = transformation


    def forward(self, x):
        result = MeanTensor(x)
        if self.transformation is not None:
            result.__transformation__ = self.transformation
        return result


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
            transformation: Optional[nn.Module] = None
    ):
        """
        Create an output layer for quantile regression.

        Args:
            name: The name of the output produced by this output layer.
            shape: The shape of the target quantity for a single input pixel.
            tau: An array containing the quantile fraction of the quantiles to predict.
            transformation: An optional transformation module that will be applied to the
                reference values prior to calculating the losses.
        """
        self.name = name
        nn.Module.__init__(self)
        if isinstance(shape, int):
            n_features = shape
        else:
            n_features = shape[0]
        stats.StatsTracker.__init__(self, n_features)
        self.tau = torch.tensor(tau, dtype=torch.float32)
        self.transformation = transformation

    def forward(self, x):
        """
        Produces a QuantileTensor from a model output.
        """
        result = QuantileTensor(x, tau=self.tau.to(device=x.device))
        if self.transformation is not None:
            result.__transformation__ = self.transformation
        return result
