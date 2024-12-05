"""
pytorch_retrieve.tensors.utils
==============================

Utility functions used within the 'pytorch_retrieve.tensors' module.
"""
from typing import Union

import torch
from torch import nn


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


def crps_cdf(
        x_cdf: torch.Tensor,
        y_cdf: torch.Tensor,
        y_true: Union[torch.Tensor, float],
        cdf_dim: int
) -> torch.Tensor:
    """
    Calculate the continuously ranked probability score (CRPS) from tensor
    describing the CDF.

    Args:
        x_cdf: A rank-k tensor containing the abscissa values of a CDF of
            probabilistic predictions.
        y_cdf: A rank-k tensor containing the ordinate values of a CDF of
            a probabilistic prediction.
        y_true: A single floating point value or a rank-(k-1) tensor containing
            the true values.
        cdf_dim: An dimension index identifying the axis along which the
            CDF values are oriented.

    Return:
        A rank-(k-1) tensor containing the CRPS values for the predictions
        represented by 'x_cdf' and 'y_cdf'.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.unsqueeze(cdf_dim)

    x_cdf_l = select(x_cdf, cdf_dim, slice(0, -1))
    x_cdf_r = select(x_cdf, cdf_dim, slice(1, None))
    d_x = torch.diff(x_cdf, dim=cdf_dim)

    weights_r = torch.clamp((x_cdf_r - y_true)/ d_x, 0.0, 1.0)
    weights_l = 1.0 - weights_r

    pad_l = (0, 0) * cdf_dim + (1, 0)
    weights_l = nn.functional.pad(weights_l, pad_l, mode="constant", value=1.0)
    pad_r = (0, 0) * cdf_dim + (0, 1)
    weights_r = nn.functional.pad(weights_r, pad_r, mode="constant", value=1.0)

    y_l = y_cdf ** 2
    y_r = (1.0 - y_cdf) ** 2

    crps = 0.5 * (
        torch.trapz(y_l * weights_l, x=x_cdf, dim=cdf_dim) +
        torch.trapz(y_r * weights_r, x=x_cdf, dim=cdf_dim)
    )
    return crps


def interpolate(
        x_new: torch.Tensor,
        x_f: torch.Tensor,
        y_f: torch.Tensor,
        dim: int
) -> torch.Tensor:
    """
    Linearly interpolate tensors along given dimension.

    Args:
        x_new: A rank-(k-1) tensor containing the values to which to
            interpolate the function represented by 'x_f' and  'y_f'.
        x_f: A rank-k tensor containing the abscissa values corresponding to
            'y_f'
        y_f: A rank-k tensor containing the ordinate values to interpolate.
        dim: An integer identifying the dimenions along which to interpolate.

    Return:
        A tensor containing the values 'y_i' containing the tensor 'y_f' interpolated
        to the values in 'x_new'.

    """
    x_new = x_new.unsqueeze(dim)
    x_f_l = select(x_f, dim, slice(0, -1))
    x_f_r = select(x_f, dim, slice(1, None))
    d_x = torch.diff(x_f, dim=dim)

    weights_l = torch.clamp((x_f_r - x_new)/ d_x, 0.0, 1.0)
    weights_r = 1.0 - weights_l

    mask = x_f_r < x_new
    weights_r[mask] = 0.0
    select(weights_r, dim, -1)[mask.all(dim)] = 1.0

    mask = x_f_l > x_new
    weights_l[mask] = 0.0
    select(weights_l, dim, 0)[mask.all(dim)] = 1.0

    norm = (weights_r + weights_l).sum(dim=dim, keepdims=True)
    weights_l = weights_l / norm
    weights_r = weights_r / norm

    y_f_l = select(y_f, dim, slice(0, -1))
    y_f_r = select(y_f, dim, slice(1, None))

    y_new = (y_f_l * weights_l) + (y_f_r * weights_r)

    return y_new.sum(dim)
