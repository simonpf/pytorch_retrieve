"""
pytorch_retrieve.tensors.probability_tensor
===========================================

Provides the ProbabilityTensor class, which is used to represent tensors
containing predictions of distributions of scalar quantities represented
 using distributions over a discretized range of values.
"""
from collections.abc import Sequence, Mapping
from typing import Union


import torch
from torch import nn


from .masked_tensor import MaskedTensor
from .utils import select


HANDLED_FUNCTIONS = {}


class ProbabilityTensor(torch.Tensor):
    """
    A Probability is a tensor that holds probabilistic estimates of
    scalar quantities represented using a sequence of probabilities
    over a discretized range of values.
    """
    def __new__(cls, tensor, bins, *args, bin_dim=1, **kwargs):
        new_tensor = super().__new__(cls, tensor, *args, **kwargs)

        ## Keep reference to original tensor.
        # if isinstance(tensor, ProbabilityTensor):
        #    new_tensor.base = tensor.base
        # else:
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
        base_args = [get_base(arg) for arg in args]
        base_kwargs = {key: get_base(val) for key, val in kwargs.items()}
        result = func(*base_args, **base_kwargs)

        if func == torch.as_tensor:
            return result

        if func == torch.Tensor.unbind or func == torch.unbind:
            return tuple([ProbabilityTensor(tensor, bins=args[0].bins) for tensor in result])

        if isinstance(result, torch.Tensor):
            p_args = get_prob_attrs(args)
            if p_args is None:
                p_args = get_prob_attrs(kwargs)
            bins, bin_dim = p_args
            return ProbabilityTensor(result, bins, bin_dim=bin_dim)

        return result

    def loss(self, y_true: Union[torch.Tensor, MaskedTensor]) -> torch.tensor:
        """
        Args:
            y_true: Tensor containing the true values.

        Return:
            The cross-entropy loss with respect to the true values.
        """
        y_binned = torch.bucketize(y_true, self.bins.type_as(y_true)) - 1
        y_binned = torch.clamp(y_binned, 0, len(self.bins) - 2)
        return nn.functional.cross_entropy(self.base, y_binned)

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
    Extract 'bins' and 'bin_dim' attributes from object.

    Return the 'bins' and 'bin_dim' values of the first value in arg
    that is a ProbabilityTensor.

    Args:
        arg: A torch.Tensor, list, or dict containing function arguments
            from which to extract the ProbabilityTensor attributes.

    Return:
        A tuple ``(bins, bin_dim)`` containing the ``bins`` and ``bin_dim``
        attributes of the first encountered ProbabilityTensor.

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
        return arg.bins, arg.bin_dim
    return None
