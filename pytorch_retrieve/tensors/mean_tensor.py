"""
pytorch_retrieve.tensors.mean_tensor
====================================

Provides the MeanTensor class, which is used to represent tensor containing
predictions of the posterior mean.
"""
from collections.abc import Sequence, Iterable, Mapping

import torch


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """
    Register a torch function override for ScalarTensor

    """

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class MeanTensor(torch.Tensor):
    """
    A MeanTensor is a tensor that holds predictions corresponding to the
    mean of the posterior distribution.
    """

    def __new__(cls, *args, **kwargs):
        tensor = super().__new__(cls, *args, **kwargs)

        # Keep reference to original tensor.
        if isinstance(args[0], MeanTensor):
            tensor.base = args[0].base
        else:
            tensor.base = args[0]

        return tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """ """
        if kwargs is None:
            kwargs = {}
        args = [get_base(arg) for arg in args]
        kwargs = {key: get_base(val) for key, val in kwargs.items()}
        result = func(*args, **kwargs)

        if func == torch.as_tensor:
            return result

        if isinstance(result, torch.Tensor):
            return MeanTensor(result)
        return result

    def __repr__(self):
        tensor_repr = self.base.__repr__()
        return "MeanTensor" + tensor_repr[6:]

    def expected_value(self) -> torch.Tensor:
        return self.base

    def loss(self, y_true):
        """
        Args:
            y_true: Tensor containing the true values.

        Return:
            The means-squared error of this tensor and 'y_true.
        """
        if y_true.dim() < self.dim():
            y_true = y_true.unsqueeze(1)
        return ((self.base - y_true) ** 2).mean()

    def probability_less_than(self, y):
        """
        A mean tensor alone is not a probabilistic estimate.
        """
        return NotImplemented


def get_base(arg):
    if isinstance(arg, tuple):
        return tuple([get_base(elem) for elem in arg])
    elif isinstance(arg, list):
        return [get_base(elem) for elem in arg]
    elif isinstance(arg, dict):
        return {key: get_base(value) for key, value in arg.items()}
    elif isinstance(arg, MeanTensor):
        return arg.base
    return arg
