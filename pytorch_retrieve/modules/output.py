"""
pytorch_retrieve.modules.output
===============================

"""
from typing import List, Optional, Union, Tuple
import torch
from torch import nn

from pytorch_retrieve.tensors import (
    MeanTensor,
    QuantileTensor,
    ClassificationTensor,
    DetectionTensor
)
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
        tau = self.tau.to(device=x.device)
        if self.training:
            result = QuantileTensor(x, tau=tau, transformation=self.transformation)
        else:
            if self.transformation is not None:
                result = QuantileTensor(self.transformation.invert(x), tau=tau, transformation=None)
            else:
                result = QuantileTensor(x, tau=tau, transformation=None)
        return result


class BinnedProbability(stats.StatsTracker, nn.Module):
    """
    The Quantiles layer produces tensors that represent probabilistic regression
    outputs.
    """
    def __init__(
            self,
            name: str,
            shape: Union[int, Tuple[int]],
            n_bins: torch.Tensor,
            x_min: float,
            x_max: float,
            scale: str = "linear",
            bins: Optional[List[float]] = None,
            transformation: Optional[nn.Module] = None
    ):
        """
        Create an output layer for a binned probability distribution.

        Args:
            name: The name of the output produced by this output layer.
            shape: The shape of the output tensor,
            n_bins: The number of probability bins.
            x_min: The smallest bin boundary.
            x_max: The largest bin boundary.
            scale: 'linear' or 'log' for linear or logarithmically spaced bins.
            bins: A list containing the bin boundaries.
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
        tau = self.tau.to(device=x.device)
        if self.training:
            result = QuantileTensor(x, tau=tau, transformation=self.transformation)
        else:
            if self.transformation is not None:
                result = QuantileTensor(self.transformation.invert(x), tau=tau, transformation=None)
            else:
                result = QuantileTensor(x, tau=tau, transformation=None)
        return result


class Classification(stats.StatsTracker, nn.Module):
    """
    The Classification layer produces tensors that represent a classficiation.
    """
    def __init__(
            self,
            name: str,
            shape: Union[int, Tuple[int]],
    ):
        """
        Create an output layer for a classficiation task.

        Args:
            name: The name of the output produced by this output layer.
            shape: The shape of the target quantity for a single input pixel.
            n_classes: The number of classes to distinguish.
        """
        self.name = name
        nn.Module.__init__(self)
        if isinstance(shape, int):
            n_features = shape
        else:
            n_features = shape[0]
        stats.StatsTracker.__init__(self, n_features)

    def forward(self, x):
        """
        Produces a ClassificationTensor from the model output.
        """
        result = ClassificationTensor(x)
        return result


class Detection(stats.StatsTracker, nn.Module):
    """
    The Detection layer produces tensors that represent a binary
    classficiation.
    """
    def __init__(
            self,
            name: str,
            shape: Union[int, Tuple[int]],
    ):
        """
        Create an output layer for a detection task.

        Args:
            name: The name of the output produced by this output layer.
            shape: The shape of the target quantity for a single input pixel.
        """
        self.name = name
        nn.Module.__init__(self)
        if isinstance(shape, int):
            n_features = shape
        else:
            n_features = shape[0]
        stats.StatsTracker.__init__(self, n_features)

    def forward(self, x):
        """
        Produces a probability of detection tensor from the model output.
        """
        result = DetectionTensor(x)
        return result
