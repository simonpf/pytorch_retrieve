"""
pytorch_retrieve.tensors.quantile_tensor
========================================

Provides the QuantileTensor class, which is used to represent tensors
containing predictions of distributions represented by a sequence of quantiles.
"""
from collections.abc import Sequence, Mapping
import functools
from typing import Union, List, Optional


import torch
from torch import nn

from .base import RegressionTensor
from .utils import interpolate


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


def select(tensor: torch.Tensor, dim: int, ind: Union[int, slice]):
    """
    Generic selec method that allows slicing.

    Args:
        tensor: The tensor from which to select a slice
        dim: The dimension along with to slice.
        ind: An index or slice defining the extent of the slice(s) to extract.

    Return:
        A view on the slice of the given tensor.
    """
    return tensor.__getitem__((slice(None),) * dim + (ind,))


class QuantileTensor(torch.Tensor, RegressionTensor):
    """
    A QuantileTensor is a tensor that holds probabilitis estimates
    of scalar quantities represented using a squence of quantiles.
    """

    def __new__(cls, tensor, tau, *args, quantile_dim=1, transformation=None, **kwargs):
        new_tensor = super().__new__(cls, tensor, *args, **kwargs)

        ## Keep reference to original tensor.
        # if isinstance(tensor, QuantileTensor):
        #    new_tensor.base = tensor.base
        # else:
        new_tensor.base = tensor
        if transformation is not None:
            new_tensor.__transformation__ = transformation
        if not hasattr(new_tensor, "__transformation__") and hasattr(tensor, "__transformation__"):
            new_tensor.__transformation__ = tensor.__transformation__

        new_tensor.tau = torch.as_tensor(tau)

        if quantile_dim < 0:
            quantile_dim = new_tensor.ndim - quantile_dim
        new_tensor.quantile_dim = quantile_dim
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
            q_args = get_quantile_attrs(args)
            return tuple([
                QuantileTensor(tensor, **q_args) for tensor in result
            ])

        if isinstance(result, torch.Tensor):
            q_args = get_quantile_attrs(args)
            if q_args is None:
                q_args = get_quantile_attrs(kwargs)
            return QuantileTensor(result, **q_args)

        return result

    def loss(self, y_true, weights: Optional[torch.Tensor] = None):
        """
        Args:
            y_true: Tensor containing the true values.
            weights: An optional tensor containing weights to weigh
                the predictions. Should have the same shape as y_true.

        Return:
            The quantile loss of this tensor with respect to 'y_true'.
        """
        if hasattr(self, "__transformation__"):
            y_true = self.__transformation__(y_true)

        tau = self.tau.to(self.device, self.dtype)
        # Pad dummy dimensions to make broadcasting work.
        tau = tau.__getitem__((...,) + (None,) * (self.ndim - self.quantile_dim - 1))

        if y_true.ndim < self.ndim:
            new_shape = y_true.shape[:1] + (1,) * (self.ndim - y_true.ndim) + y_true.shape[1:]
            y_true = y_true.reshape(new_shape)
            if weights is not None:
                weights = weights.reshape(new_shape)

        delta = y_true - self.base

        if weights is None:
            loss = torch.where(delta > 0, tau * delta, (tau - 1.0) * delta)
            return loss.mean()

        if weights.shape != y_true.shape:
            raise ValueError(
                "If provided, 'weights' must match the reference tensor 'y_true'."
            )

        if weights.ndim < self.ndim:
            weights = weights.unsqueeze(self.quantile_dim)

        loss = torch.where(delta > 0, tau * delta, (tau - 1.0) * delta)
        return (weights * loss).sum() / weights.sum()

    def __repr__(self):
        tensor_repr = self.base.__repr__()
        return "QuantileTensor" + tensor_repr[6:]

    def cdf(self):
        """
        Calculates the cumulative distribution functions (PDF) of the
        distributions represented by this quantile tensor.

        This method  extends the quantiles in 'y_pred' to  0 and 1 by
        extending the first and last segments with a 50% reduction in the
        slope.

        Args:
            y_pred: Array containing a range of predicted quantiles. The array
                is expected to contain the quantiles along the axis given by
                ``quantile_dim.``
            quantiles: Array containing quantile fraction corresponding to the
                the predicted quantiles.
            quantile_dim: The index of the axis f the ``y_pred`` array, along
                which the quantiles are found.

        Returns:
            Tuple ``(x_pdf, y_pdf)`` of x and corresponding y-values of the PDF
            corresponding to quantiles given by ``y_pred``.

        Raises:

            InvalidArrayTypeException: When the data is provided neither as
                numpy array nor as torch tensor.

            InvalidDimensionException: When the provided predicted quantiles do
                not match the provided number of quantiles.
        """
        if hasattr(self, "__transformation__"):
            base = self.__transformation__.invert(self.base)
        else:
            base = self.base

        qdim = self.quantile_dim
        d_x = torch.diff(base, dim=qdim)
        pad_dims = self.dim() - qdim - 1
        y_cdf = nn.functional.pad(self.tau, (1, 1))
        y_cdf[0] = 0.0
        y_cdf[-1] = 1.0

        x_cdf = base

        d_y = self.tau[1] - self.tau[0]
        x_0 = select(x_cdf, qdim, slice(0, 1))
        x_1 = select(x_cdf, qdim, slice(1, 2))
        d_x = x_1 - x_0
        x_l = x_0 - 3.0 * self.tau[0] * d_x / d_y

        d_y = self.tau[-1] - self.tau[-2]
        x_0 = select(x_cdf, qdim, slice(-2, -1))
        x_1 = select(x_cdf, qdim, slice(-1, None))
        d_x = x_1 - x_0
        x_r = x_1 + 3.0 * (1.0 - self.tau[-1]) * d_x / d_y

        slices = [x_l, x_cdf, x_r]
        x_cdf = torch.cat(slices, qdim)

        return x_cdf, y_cdf

    def pdf(self):
        """
        Calculate the probability distribution function (PDF) of the
        distributions represented by this quantile tensor.

        This method  extends the quantiles in 'y_pred' to  0 and 1 by
        extending the first and last segments with a 50% reduction in the
        slope.

        Returns:
            Tuple ``(x_pdf, y_pdf)`` of x and corresponding y-values of the PDF
            corresponding to quantiles given by ``y_pred``.
        """

        qdim = self.quantile_dim
        pad_dims = self.dim() - qdim - 1

        x_cdf, y_cdf = self.cdf()
        d_y = torch.diff(y_cdf).__getitem__((...,) + (None,) * pad_dims)
        d_x = torch.diff(x_cdf, dim=qdim)

        y_pdf = d_y / d_x
        x_pdf = 0.5 * (
            select(x_cdf, qdim, slice(1, None))
            + select(x_cdf, qdim, slice(0, -1))
        )
        return x_pdf, y_pdf


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
        x_cdf, y_cdf = self.cdf()
        return torch.trapz(x_cdf, y_cdf, dim=self.quantile_dim)

    def probability_less_than(self, thresh: float) -> torch.Tensor:
        """
        Calculate the probabilities that predicted values are less than a given
        threshold.

        Args:
             thresh: The threshold value.

        Return:
             A tensor with one less dimension that this probability tensor
             containing the probabilities that the predicted values are less
             than 'thresh'.
        """
        x_cdf, y_cdf = self.cdf()

        qdim = self.quantile_dim
        x_r = select(x_cdf, qdim, slice(1, None))
        d_x = torch.diff(x_cdf, dim=qdim)

        weight_l = (x_r - thresh) / d_x
        mask = (weight_l < 0) + (weight_l > 1)
        weight_l = torch.where(mask, 0.0, weight_l)
        weight_r = torch.where(mask, 0.0, 1.0 - weight_l)

        pad_dims = self.dim() - qdim - 1
        y_l = y_cdf[:-1].__getitem__((...,) + (None,) * pad_dims)
        y_r = y_cdf[1:].__getitem__((...,) + (None,) * pad_dims)

        prob = (y_l * weight_l).sum(qdim) + (y_r * weight_r).sum(qdim)
        prob /= weight_l.sum(qdim) + weight_r.sum(qdim)

        out_of_bounds_l = thresh < torch.min(x_cdf, dim=qdim)[0]
        prob[out_of_bounds_l] = 0.0

        out_of_bounds_r = thresh > torch.max(x_cdf, dim=qdim)[0]
        prob[out_of_bounds_r] = 1.0

        return prob

    def probability_greater_than(self, thresh):
        """
        Calculate the probabilities that predicted values are greater than a given
        threshold.

        Args:
             thresh: The threshold value.

        Return:
             A tensor with one less dimension that this probability tensor
             containing the probabilities that the predicted values are greater
             than 'thresh'.
        """
        return 1.0 - self.probability_less_than(thresh)


    def maximum_probability(self):
        """
        Calculate the value corresponding to the maximum of the propability density represented
        by this quantile tensor.

        Return:
            A tensor containing the values that maximize the probability density of the distribution
            represented by this tensor.
        """
        x_pdf, y_pdf = self.pdf()
        inds = y_pdf.max(dim=self.quantile_dim, keepdim=True)[1]
        pdf_max = torch.gather(x_pdf, self.quantile_dim, inds)
        return select(pdf_max, dim=self.quantile_dim, ind=0)


    def quantiles(self, tau: Union[List[float], torch.Tensor]) -> "QuantileTensor":
        """
        Calculate arbitrary quantiles by linear interpolation of existing
        quantiles.

        Args:
            tau: A list or tensor containing the quantile fraction to which to
                 interpolate the quantile tensor.

        Return:
            A new quantile tensor containing the quantile corresponding to the
            desired quantile fractions.
        """
        new = []
        for tau_i in tau:
            ind_r = min(torch.bucketize(tau_i, self.tau), self.tau.numel() - 1)
            tau_r = self.tau[ind_r]
            if ind_r > 1:
                tau_l = self.tau[ind_r - 1]
                d_tau = tau_r - tau_l
                w_r = (tau_i - tau_l) / d_tau
                w_l = max((tau_r - tau_i) / d_tau, 0.0)
                w_r = 1.0 - w_l
                res = w_l * torch.select(self.base, self.quantile_dim, ind_r - 1)
                res += w_r * torch.select(self.base, self.quantile_dim, ind_r)
            else:
                res = torch.select(self.base, self.quantile_dim, 0)
            new.append(res)

        new = torch.stack(new, self.quantile_dim)
        return QuantileTensor(new, torch.tensor(tau))


    def random_sample(self, n_samples: int = 1) -> "QuantileTensor":
        """
        Generate random sample from posterior distribution.

        Args:
            n_samples: The number of samples to generate per output pixel.

        Return:
            A tensor containing random samples from the distributions represented by
            by this quantile tensor.
        """
        samples = []

        x_cdf, y_cdf = self.cdf()
        y_cdf = y_cdf.__getitem__((...,) + (None,) * (self.ndim - self.quantile_dim - 1))
        y_cdf = torch.broadcast_to(y_cdf, x_cdf.shape)

        for ind in range(n_samples):
            sample_shape = list(self.shape)
            del sample_shape[self.quantile_dim]
            rand = torch.rand(sample_shape, device=self.device, dtype=self.dtype)
            samples.append(interpolate(rand, y_cdf, x_cdf, dim=self.quantile_dim))

        if n_samples > 1:
            return torch.stack(samples)

        return samples[0]


def get_base(arg):
    """
    Recursively strip QuantileTensor types of arguments.

    Args:
        An arbitray Python object.

    Return:
        The same Python object is not a tensor. If a container containing quantile tensors,
        the same container containining regular tensors is returned.
    """
    if isinstance(arg, tuple):
        return tuple([get_base(elem) for elem in arg])
    elif isinstance(arg, list):
        return [get_base(elem) for elem in arg]
    elif isinstance(arg, dict):
        return {key: get_base(value) for key, value in arg.items()}
    elif isinstance(arg, QuantileTensor):
        return arg.base
    return arg


def get_quantile_attrs(arg):
    """
    Extract 'tau' and 'quantile_dim', and 'transformation' attributes from objects.

    Return the 'tau', 'quantile_dim', and 'transformation' attributes of the first value in arg
    that is a QuantileTensor.

    Args:
        arg: A torch.Tensor, list, or dict containing function arguments
            from which to extract the QuantileTensor attributes.

    Return:
        A tuple ``(tau, quantile_dim)`` containing the ``tau`` and ``quantile_dim``
        attributes of the first encountered QuantileTensor.

    """
    if isinstance(arg, Sequence):
        for elem in arg:
            result = get_quantile_attrs(elem)
            if result is not None:
                return result
    elif isinstance(arg, Mapping):
        for elem in arg.values():
            result = get_quantile_attrs(elem)
            if result is not None:
                return result
    elif isinstance(arg, QuantileTensor):
        return {
            "tau": arg.tau,
            "quantile_dim": arg.quantile_dim,
            "transformation": getattr(arg, "__transformation__", None)
        }
    return None


@implements(torch.Tensor.to)
def to(inpt, *args, **kwargs):
    """
    Implementation of .to method.
    """
    other = inpt.base.to(*args, **kwargs)
    tau = inpt.tau.to(*args, **kwargs)
    return QuantileTensor(other, tau=tau)


@implements(torch.Tensor.cpu)
def cpu(inpt, *args, **kwargs):
    """
    Implementation of .cpu method.
    """
    other = inpt.base.cpu(*args, **kwargs)
    tau = inpt.tau.cpu(*args, **kwargs)
    return QuantileTensor(other, tau=tau)
