"""
pytorch_retrieve.modules.normalization
======================================

Custom normalization layers and interface to pytorch normalization factories.
"""
from typing import Callable

import torch
from torch import nn


class LayerNormFirst(nn.Module):
    """
    Layer norm that normalizes the first dimension.
    """

    def __init__(self, n_channels, eps=1e-6):
        """
        Args:
            n_channels: The number of channels in the input.
            eps: Epsilon added to variance to avoid numerical issues.
        """
        super().__init__()
        self.scaling = nn.Parameter(torch.ones(n_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_channels), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        """
        Apply normalization to x.
        """
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x_n = (x - mu) / torch.sqrt(var + self.eps)
        x = self.scaling[..., None, None] * x_n + self.bias[..., None, None]
        return x


def get_normalization_factory(name: str) -> Callable:
    """
    Retrieve a normalization factory from its name.

    Args:
        name: String specifying the name of a normalization factory.

    Rerturn:
        A normalization factory, i.e. a callable that can be used to
        produce normalization layers.
    """
    if name is None or name == "none":
        return None

    if hasattr(nn, name):
        return getattr(nn, name)

    if name in globals():
        return globals()[name]

    raise RuntimeError(
        f"The normalization factory {name} does not match any of the "
        " supported normalization factories. 'normalization_factory' must "
        " either match the name of a module provided by 'torch.nn' or a "
        " normalization layer module defined in "
        " pytorch_retrieve.modules.normalization."
    )
