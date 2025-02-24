"""
pytorch_retrieve.tensors.probability_tensor
===========================================

Provides the ProbabilityTensor class, which is used to represent tensors
containing predictions of distributions of scalar quantities represented
 using distributions over a discretized range of values.
"""
from collections.abc import Sequence, Mapping
import functools
from typing import Union, Optional


import torch
from torch import nn


from .masked_tensor import MaskedTensor
from .utils import select
from .base import RegressionTensor


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class ProbabilityTensor(torch.Tensor, RegressionTensor):
    """
    A Probability is a tensor that holds probabilistic estimates of
    scalar quantities represented using a sequence of probabilities
    over a discretized range of values.
    """

    def __new__(cls, tensor, bins, *args, bin_dim=1, transformation=None, **kwargs):
        new_tensor = super().__new__(cls, tensor, *args, **kwargs)

        if transformation is not None:
            new_tensor.__transformation__ = transformation
        if not hasattr(new_tensor, "__transformation__") and hasattr(tensor, "__transformation__"):
            new_tensor.__transformation__ = tensor.__transformation__

        new_tensor.base = tensor

        new_tensor.bins = torch.as_tensor(bins)

        if bin_dim < 0:
            bin_dim = new_tensor.ndim - bin_dim
        new_tensor.bin_dim = bin_dim
        return new_tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """ """
        if kwargs is None:
            kwargs = {}

        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

        base_args = [get_base(arg) for arg in args]
        base_kwargs = {key: get_base(val) for key, val in kwargs.items()}
        result = func(*base_args, **base_kwargs)

        if func == torch.as_tensor:
            return result

        if func == torch.Tensor.unbind or func == torch.unbind:
            return tuple(
                [ProbabilityTensor(tensor, bins=args[0].bins) for tensor in result]
            )

        if isinstance(result, torch.Tensor):
            p_args = get_prob_attrs(args)
            if p_args is None:
                p_args = get_prob_attrs(kwargs)
            return ProbabilityTensor(result, **p_args)

        return result

    def loss(
        self,
        y_true: Union[torch.Tensor, MaskedTensor],
        weights: Optional[torch.Tensor] = None,
    ) -> torch.tensor:
        """
        Args:
            y_true: Tensor containing the true values.
            weights: An optional tensor containing weights to weigh
                the predictions. Should have the same shape as y_true.
        Return:
            The cross-entropy loss with respect to the true values.
        """
        y_binned = torch.bucketize(y_true, self.bins.type_as(y_true)) - 1
        y_binned = torch.clamp(y_binned, 0, len(self.bins) - 2)

        if weights is None:
            return nn.functional.cross_entropy(self.base, y_binned)

        if weights.shape != y_true.shape:
            raise ValueError(
                "If provided, 'weights' must match the reference tensor 'y_true'."
            )

        loss = nn.functional.cross_entropy(self.base, y_binned, reduction="none")
        return (weights * loss).sum() / weights.sum()

    def __repr__(self):
        tensor_repr = self.base.__repr__()
        return "ProbabilityTensor" + tensor_repr[6:]

    def pdf(self):
        """
        Calculates the probability density functions (PDF) of the
        distributions represented by this probability tensor.

        Returns:
            Tuple ``(x_pdf, y_pdf)`` of target values ``x_pdf`` and
            corresponding probability densities ``y_pdf``.
        """
        probs = nn.functional.softmax(self.base, dim=self.bin_dim)
        bins = self.bins.to(dtype=self.base.dtype, device=self.base.device)
        d_x = torch.diff(bins)
        dim_pad = (...,) + (None,) * (self.base.ndim - self.bin_dim - 1)
        y_pdf = probs / d_x[dim_pad]
        x_pdf = torch.broadcast_to((bins[:-1] + 0.5 * d_x)[dim_pad], y_pdf.shape)
        return x_pdf, y_pdf

    def cdf(self):
        """
        Calculates the cumulative distribution function (CDF) of the
        distributions represented by this probability tensor.

        Returns:
            Tuple ``(x_cdf, y_cdf)`` of target values ``x_cdf`` and
            corresponding probability densities ``y_cdf``.
        """
        probs = nn.functional.softmax(self.base, dim=self.bin_dim)
        bins = self.bins.to(dtype=self.base.dtype, device=self.base.device)
        d_x = torch.diff(bins)
        dim_pad = (...,) + (None,) * (self.base.ndim - self.bin_dim - 1)

        y_cdf = torch.cumsum(probs, self.bin_dim)
        pad = (0, 0) * self.bin_dim + (1, 0)
        y_cdf = nn.functional.pad(y_cdf, pad, mode="constant", value=0.0)
        x_cdf = torch.broadcast_to(bins[dim_pad], y_cdf.shape)
        return x_cdf, y_cdf

    def expected_value(self):
        r"""
        Computes the mean of the posterior distribution defined by an array
        of predicted quantiles.

        Args:
            y_pred: A tensor of predicted quantiles with the quantiles located
                along the axis given by ``quantile_axis``.

        Returns:

            Array containing the posterior means for the provided inputs.
        """
        probs = nn.functional.softmax(self.base, dim=self.bin_dim)
        bins = self.bins.to(dtype=self.base.dtype, device=self.base.device)
        x_c = 0.5 * (bins[1:] + bins[:-1])
        dim_pad = (...,) + (None,) * (self.base.ndim - self.bin_dim - 1)
        exp = torch.sum(probs * x_c[dim_pad], dim=self.bin_dim)
        return exp

    def maximum_probability(self):
        """
        Calculate the value corresponding to the maximum of the propability density represented
        by this tensor.

        Return:
            A tensor containing the values that maximize the probability density of the distribution
            represented by this tensor.
        """
        x_pdf, y_pdf = self.pdf()
        inds = y_pdf.max(dim=self.bin_dim, keepdim=True)[1]
        pdf_max = torch.gather(x_pdf, self.bin_dim, inds)
        return select(pdf_max, dim=self.bin_dim, ind=0)

    def probability_less_than(self, thresh):
        """
        A mean tensor alone is not a probabilistic estimate.
        """
        probs = nn.functional.softmax(self.base, dim=self.bin_dim)

        w_b = torch.diff(self.bins)

        weights = torch.clamp((thresh - self.bins[1:]) / w_b + 1.0, min=0.0, max=1.0)
        weights = weights.to(device=self.base.device, dtype=self.base.dtype)

        dim_pad = (...,) + (None,) * (self.base.ndim - self.bin_dim - 1)
        probs = (probs * weights[dim_pad]).sum(self.bin_dim)
        return probs

    def probability_greater_than(self, thresh):
        """
        A mean tensor alone is not a probabilistic estimate.
        """
        return 1.0 - self.probability_less_than(thresh)


def get_base(arg) -> torch.Tensor:
    """
    Get 'base' tensor of an argument.

    Return the raw data tensors from an argument that is a ProbabilityTensor
    or an iterable containing probability tensors.
    """
    if isinstance(arg, tuple):
        return tuple([get_base(elem) for elem in arg])
    elif isinstance(arg, list):
        return [get_base(elem) for elem in arg]
    elif isinstance(arg, dict):
        return {key: get_base(value) for key, value in arg.items()}
    elif isinstance(arg, ProbabilityTensor):
        return arg.base
    return arg


def get_prob_attrs(arg):
    """
    Extract 'bins', 'bin_dim', and 'transformation' attributes from object.

    Return the 'bins', 'bin_dim', and 'transformation' attributes from the first value in arg
    that is a ProbabilityTensor.

    Args:
        arg: A torch.Tensor, list, or dict containing function arguments
            from which to extract the ProbabilityTensor attributes.

    Return:
        A dict  containing the ``bins``, ``bin_dim``, and ``transformation``
        attributes to construct a new ProbabilityTensor

    """
    if isinstance(arg, Sequence):
        for elem in arg:
            result = get_prob_attrs(elem)
            if result is not None:
                return result
    elif isinstance(arg, Mapping):
        for elem in arg.values():
            result = get_prob_attrs(elem)
            if result is not None:
                return result
    elif isinstance(arg, ProbabilityTensor):
        return {
            "bins": arg.bins,
            "bin_dim": arg.bin_dim,
            "transformation": getattr(arg, "__transformation__", None)
        }
    return None


@implements(torch.Tensor.to)
def to(inpt, *args, **kwargs):
    """
    Implementation of .to method.
    """
    other = inpt.base.to(*args, **kwargs)
    bins = inpt.bins.to(*args, **kwargs)
    return ProbabilityTensor(other, bins=bins, bin_dim=inpt.bin_dim)


@implements(torch.Tensor.cpu)
def cpu(inpt, *args, **kwargs):
    """
    Implementation of .cpu method.
    """
    other = inpt.base.cpu(*args, **kwargs)
    bins = inpt.bins.cpu(*args, **kwargs)
    return ProbabilityTensor(other, bins=bins, bin_dim=inpt.bin_dim)
