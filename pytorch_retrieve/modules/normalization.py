"""
pytorch_retrieve.modules.normalization
======================================

Custom normalization layers and interface to pytorch normalization factories.
"""
from typing import Callable

from torch import nn


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

    if name in dir():
        return dir()[name]

    raise RuntimeError(
        f"The normalization factory {name} does not match any of the "
        " supported normalization factories. 'normalization_factory' must "
        " either match the name of a module provided by 'torch.nn' or a "
        " normalization layer module defined in "
        " pytorch_retrieve.modules.normalization."
    )
