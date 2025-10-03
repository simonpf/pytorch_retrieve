"""
pytorch_retrieve.tensors.classification
=======================================

Provides the 'DetectionTensor' and 'ClassificationTensor' classes representing
the ouput from detection (binary classification) and mulit-class
 classification.
"""
from typing import Optional

import torch
from torch import nn

from .masked_tensor import MaskedTensor


class DetectionTensor(torch.Tensor):
    """
    A DetectionTensor is a tensor that holds predictions of a detection
    of two-class classification tasks.
    """

    def __new__(cls, *args, **kwargs):
        tensor = super().__new__(cls, *args, **kwargs)

        # Keep reference to original tensor.
        if isinstance(args[0], DetectionTensor):
            tensor.base = args[0].base
        else:
            tensor.base = args[0]

        return tensor

    @classmethod
    def get_base(cls, arg) -> torch.Tensor:
        """
        Get 'base' tensor of an argument.

        Return the raw data tensors from an argument that is a ProbabilityTensor
        or an iterable containing probability tensors.
        """
        if isinstance(arg, tuple):
            return tuple([cls.get_base(elem) for elem in arg])
        elif isinstance(arg, list):
            return [cls.get_base(elem) for elem in arg]
        elif isinstance(arg, dict):
            return {key: cls.get_base(value) for key, value in arg.items()}
        elif isinstance(arg, cls):
            return arg.base
        return arg

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """ """
        if kwargs is None:
            kwargs = {}
        args = [cls.get_base(arg) for arg in args]
        kwargs = {key: cls.get_base(val) for key, val in kwargs.items()}
        result = func(*args, **kwargs)

        if func == torch.as_tensor:
            return result

        if func == torch.Tensor.unbind or func == torch.unbind:
            return tuple([DetectionTensor(tensor) for tensor in result])

        if isinstance(result, torch.Tensor):
            return DetectionTensor(result)
        return result

    def __repr__(self):
        tensor_repr = self.base.__repr__()
        return "DetectionTensor" + tensor_repr[6:]

    def loss(
        self, y_true: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            y_true: Tensor containing the true values.
            weights: An optional tensor containing weights to weigh
                the predictions. Must have the same shape as y_true.

        Return:
            The binary cross-entropy loss of this tensor and the given
            true label.

        """
        if weights is None:
            return nn.functional.binary_cross_entropy_with_logits(
                self.base.squeeze(),
                y_true.squeeze(),
            )

        if weights.shape != y_true.shape:
            raise ValueError(
                "If provided, 'weights' must match the reference tensor 'y_true'."
            )

        loss = nn.functional.binary_cross_entropy_with_logits(
            self.base.squeeze(), y_true.squeeze(), reduction="none"
        )
        return (loss * weights).sum() / weights.sum()

    def probability(self):
        """
        Return detection probabilities.
        """
        return torch.sigmoid(self.base)

    def most_likely_class(self) -> torch.Tensor:
        """
        Calculate most likely class.
        """
        return (0.5 < self.probability()).astype(self.dtype)


class ClassificationTensor(torch.Tensor):
    """
    A classification tensor is a tensor that holds predictions of a classification
    task.
    """

    @classmethod
    def get_base(cls, arg) -> torch.Tensor:
        """
        Get 'base' tensor of an argument.

        Return the raw data tensors from an argument that is a ProbabilityTensor
        or an iterable containing probability tensors.
        """
        if isinstance(arg, tuple):
            return tuple([cls.get_base(elem) for elem in arg])
        elif isinstance(arg, list):
            return [cls.get_base(elem) for elem in arg]
        elif isinstance(arg, dict):
            return {key: cls.get_base(value) for key, value in arg.items()}
        elif isinstance(arg, cls):
            return arg.base
        return arg

    def __new__(cls, *args, **kwargs):
        tensor = super().__new__(cls, *args, **kwargs)

        # Keep reference to original tensor.
        if isinstance(args[0], ClassificationTensor):
            tensor.base = args[0].base
        else:
            tensor.base = args[0]

        return tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """ """
        if kwargs is None:
            kwargs = {}
        args = [cls.get_base(arg) for arg in args]
        kwargs = {key: cls.get_base(val) for key, val in kwargs.items()}
        result = func(*args, **kwargs)

        if func == torch.as_tensor:
            return result

        if func == torch.Tensor.unbind or func == torch.unbind:
            return tuple([ClassificationTensor(tensor) for tensor in result])

        if isinstance(result, torch.Tensor):
            return ClassificationTensor(result)
        return result

    def __repr__(self):
        tensor_repr = self.base.__repr__()
        return "ClassificationTensor" + tensor_repr[6:]

    def loss(
        self, y_true: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            y_true: Tensor containing the true values.
            weights: An optional tensor containing weights to weigh
                the predictions. Must have the same shape as y_true.

        Return:
            The binary cross-entropy loss of this tensor and the given
            true label.

        """
        if weights is None:
            return nn.functional.cross_entropy(
                self.base,
                y_true,
            )

        if weights.shape != y_true.shape:
            raise ValueError(
                "If provided, 'weights' must match the reference tensor 'y_true'."
            )

        loss = nn.functional.cross_entropy(self.base, y_true, reduction="none")
        return (loss * weights).sum() / weights.sum()

    def probability(self) -> torch.Tensor:
        """
        Return class probabilities.
        """
        return torch.softmax(self.base, 1)

    def most_likely_class(self) -> torch.Tensor:
        """
        Calculate most likely class.
        """
        probs = self.probability()
        inds = torch.argmax(probs, 1).to(dtype=self.dtype)
        return inds
