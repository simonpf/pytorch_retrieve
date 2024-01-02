"""
pytorch_retrieve.tensors.quantile_tensor
========================================

Provides the QuantileTensor class, which is used to represent tensors
containing predictions of distributions represented by a sequence of quantiles.
"""
from collections.abc import Sequence, Mapping
from typing import Union


import torch
from torch import nn


HANDLED_FUNCTIONS = {}


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


class QuantileTensor(torch.Tensor):
    """
    A QuantileTensor is a tensor that holds predictions that represent a tensor
    of distributions using a squence of quantiles.
    """

    def __new__(cls, tensor, tau, *args, quantile_dim=1, **kwargs):
        new_tensor = super().__new__(cls, tensor, *args, **kwargs)

        print(type(tensor))
        ## Keep reference to original tensor.
        # if isinstance(tensor, QuantileTensor):
        #    new_tensor.base = tensor.base
        # else:
        new_tensor.base = tensor

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
        base_args = [get_base(arg) for arg in args]
        base_kwargs = {key: get_base(val) for key, val in kwargs.items()}
        result = func(*base_args, **base_kwargs)

        print(func, types)

        if func == torch.as_tensor:
            return result

        if isinstance(result, torch.Tensor):
            q_args = get_quantile_attrs(args)
            if q_args is None:
                q_args = get_quantile_attrs(kwargs)
            tau, quantile_dim = q_args
            return QuantileTensor(result, tau, quantile_dim=quantile_dim)

        return result

    def loss(self, y_true):
        """
        Args:
            y_true: Tensor containing the true values.

        Return:
            The quantile loss of this tensor with respect to 'y_true'.
        """
        tau = self.tau.to(self.device, self.dtype)
        # Pad dummy dimensions to make broadcasting work.
        tau = tau.__getitem__((...,) + (None,) * (self.ndim - self.quantile_dim - 1))
        if y_true.ndim < self.ndim:
            y_true = y_true.unsqueeze(self.quantile_dim)

        delta = self.base - y_true
        loss = torch.where(delta > 0, tau * delta, (tau - 1.0) * delta)
        return loss.mean()

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
        qdim = self.quantile_dim
        d_x = torch.diff(self.base, dim=qdim)
        pad_dims = self.dim() - qdim - 1
        y_cdf = nn.functional.pad(self.tau, (1, 1))
        y_cdf[0] = 0.0
        y_cdf[-1] = 1.0

        x_cdf = self.base

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

        qdim = self.quantile_dim
        pad_dims = self.dim() - qdim - 1

        x_cdf, y_cdf = self.cdf()
        d_y = torch.diff(y_cdf).__getitem__((...,) + (None,) * pad_dims)
        d_x = torch.diff(x_cdf, dim=qdim)

        y_pdf = d_y / d_x
        x_pdf = 0.5 * (
            select(x_cdf, qdim, slice(1, None)) + select(x_cdf, qdim, slice(0, -1))
        )
        return x_pdf, y_pdf

        y_pdf = d_y / d_x
        slices = [
            0.5 * select(y_pdf, qdim, 0).unsqueeze(qdim),
            y_pdf,
            0.5 * select(y_pdf, qdim, -1).unsqueeze(qdim),
        ]
        y_pdf = torch.cat(slices, qdim)

        x_pdf = 0.5 * (
            select(self.base, qdim, slice(1, None))
            + select(self.base, qdim, slice(0, -1))
        )
        slices = [
            2.0 * select(x_pdf, qdim, slice(0, 1)) - select(x_pdf, qdim, slice(1, 2)),
            x_pdf,
            2.0 * select(x_pdf, qdim, slice(-1, None))
            - select(x_pdf, qdim, slice(-2, -1)),
        ]
        x_pdf = torch.cat(slices, qdim)

        return x_pdf, y_pdf

    def expected_value(y_pred, quantiles, quantile_axis=1):
        r"""
        Computes the mean of the posterior distribution defined by an array
        of predicted quantiles.

        Args:
            y_pred: A tensor of predicted quantiles with the quantiles located
                along the axis given by ``quantile_axis``.
            quantiles: The quantile fractions corresponding to the quantiles
                located along the quantile axis.
            quantile_axis: The axis along which the quantiles are located.

        Returns:

            Array containing the posterior means for the provided inputs.
        """
        x_cdf, y_cdf = self.cdf()
        return torch.trapz(x_cdf, y_cdf, self.quantile_dim)

    def probability_less_than(self, thresh):
        """
        A mean tensor alone is not a probabilistic estimate.
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

    def probability_larger_than(self, thresh):
        """
        A mean tensor alone is not a probabilistic estimate.
        """
        return 1.0 - self.probability_less_than(thresh)


def get_base(arg):
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
    Extract 'tau' and 'quantile_dim' attributes from object.

    Return the 'tau' and 'quantile_dim' values of the first value in arg
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
        return arg.tau, arg.quantile_dim
    return None