"""
pytorch_retrieve.tensors.mean_tensor
====================================

Provides the MeanTensor class, which is used to represent tensors containing
predictions of the posterior mean.
"""
from collections.abc import Sequence, Iterable, Mapping
from typing import Optional

import torch

from .base import RegressionTensor


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """
    Register a torch function override for MeanTensor.
    """

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class MeanTensor(torch.Tensor, RegressionTensor):
    """
    A MeanTensor is a tensor that holds predictions corresponding to the
    mean of the posterior distribution.
    """

    def __new__(cls, *args, transformation=None, **kwargs):

        tensor = super().__new__(cls, *args, **kwargs)

        if transformation is not None:
            tensor.__transformation__ = transformation

        # Keep reference to original tensor.
        if isinstance(args[0], MeanTensor):
            tensor.base = args[0].base
            if not hasattr(tensor, "__transformation__") and hasattr(args[0], "__transformation__"):
                self.__transformation__ = None
        else:
            tensor.base = args[0]

        return tensor


    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """ """
        if kwargs is None:
            kwargs = {}
        attrs = get_tensor_attrs(args)
        args = [get_base(arg) for arg in args]
        kwargs = {key: get_base(val) for key, val in kwargs.items()}
        result = func(*args, **kwargs)

        if func == torch.as_tensor:
            return result

        if func == torch.Tensor.unbind or func == torch.unbind:
            return tuple([MeanTensor(tensor, **attrs) for tensor in result])

        if isinstance(result, torch.Tensor):
            return MeanTensor(result, **attrs)
        return result

    def __repr__(self):
        tensor_repr = self.base.__repr__()
        return "MeanTensor" + tensor_repr[6:]

    def expected_value(self) -> torch.Tensor:
        if hasattr(self, "__transformation__"):
            return self.__transformation__.invert(self.base)
        return self.base

    def loss(self, y_true, weights: Optional[torch.Tensor] = None):
        """
        Args:
            y_true: Tensor containing the true values.
            weights: An optional tensor containing weights to weigh
                the predictions. Must have the same shape as y_true.

        Return:
            The means-squared error of this tensor and 'y_true.
        """
        if hasattr(self, "__transformation__"):
            y_true = self.__transformation__(y_true)

        if y_true.dim() < self.dim():
            new_shape = y_true.shape[:1] + (1,) * (self.ndim - y_true.ndim) + y_true.shape[1:]
            y_true = y_true.reshape(new_shape)
            if weights is not None:
                weights = weights.reshape(new_shape)

        if weights is None:
            return ((self.base - y_true) ** 2).mean()

        if weights.shape != y_true.shape:
            raise ValueError(
                f"If provided, the shape of 'weights' {weights.shape} must "
                f"match the reference tensor 'y_true' {y_true.shape}."
            )

        if weights.dim() < self.dim():
            weights = weights.unsqueeze(1)

        return (weights * (self.base - y_true) ** 2).sum() / weights.sum()

    def probability_less_than(self, y):
        """
        A mean tensor alone is not a probabilistic estimate.
        """
        return NotImplemented


def get_base(arg):
    """
    Recursively strip MeanTensor types of arguments.

    Args:
        An arbitray Python object.

    Return:
        The same Python object is not a tensor. If a container containing mean tensors,
        the same container containining regular tensors is returned.
    """
    if isinstance(arg, tuple):
        return tuple([get_base(elem) for elem in arg])
    elif isinstance(arg, list):
        return [get_base(elem) for elem in arg]
    elif isinstance(arg, dict):
        return {key: get_base(value) for key, value in arg.items()}
    elif isinstance(arg, MeanTensor):
        return arg.base
    return arg


def get_tensor_attrs(arg):
    """
    Extract and 'transformation' attributes from objects.

    Return the 'transformation' attribute of the first value in arg
    that is a MeanTensor.

    Args:
        arg: A torch.Tensor, list, or dict containing function arguments
            from which to extract the tensor attributes.

    Return:
        A dictionary  containing the ``transformation`` attribute of the
        first encountered MeanTensor.
    """
    if isinstance(arg, Sequence):
        for elem in arg:
            result = get_tensor_attrs(elem)
            if result is not None:
                return result
    elif isinstance(arg, Mapping):
        for elem in arg.values():
            result = get_mean_attrs(elem)
            if result is not None:
                return result
    elif isinstance(arg, MeanTensor):
        return {
            "transformation": getattr(arg, "__transformation__", None)
        }
    return None
