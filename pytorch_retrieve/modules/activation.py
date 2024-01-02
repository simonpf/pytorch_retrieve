"""
pytorch_retrieve.modules.activation
===================================

Provides an interface to load torch activations functions from config
files.
"""
from typing import Callable

from torch import nn


def get_activation_factory(name: str) -> Callable:
    """
    Retrieve an activation function from torch.nn.
    """
    if hasattr(nn, name):
        return getattr(nn, name)

    raise RuntimeError(
        f"The activation factory {name} does not match any of the "
        " supported activation factories. 'activation_factory' must "
        "  match the name of a module provided by 'torch.nn'."
    )
