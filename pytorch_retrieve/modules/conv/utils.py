"""
pytorch_retrieve.modules.conv.utils
===================================

Defines the scale class used to keep track of the scales of tensors
in a NN model.
"""
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np


@dataclass
class Scale:
    """
    Special data type used to represent scales of tensors.

    A scale is represented using a length-three sequence of integers representing
    the relative size of a single tensor pixel alsong the time, height, and width
    dimensions respectively.

    Downsampling and upsampling of tensors are represented by multiplying and,
    respectively, dividing the scale by the corresponding down- or
    upsampling factors.
    """
    scale: Tuple[int]

    def __init__(self, scl: Union[int, Tuple[int], List[int]]):
        if isinstance(scl, Scale):
            self.scale = scl.scale
        elif isinstance(scl, int):
            self.scale = (1,) + 2 * (scl,)
        elif isinstance(scl, (list, tuple)):
            self.scale = (1,) * (3 - len(scl)) + tuple(scl)
        else:
            raise RuntimeError(
                f"Cannot construct scale from {type(scl)}."
            )

    def __eq__(self, other):
        if not isinstance(other, Scale):
            other = Scale(other)
        return self.scale == other.scale

    def __lt__(self, other):
        if not isinstance(other, Scale):
            other = Scale(other)
        return np.prod(self.scale) < np.prod(other.scale)

    def __le__(self, other):
        if not isinstance(other, Scale):
            other = Scale(other)
        return np.prod(self.scale) <= np.prod(other.scale)

    def __mul__(self, other):
        other = Scale(other)
        return Scale([scl_l * scl_r for scl_l, scl_r in zip(self.scale, other.scale)])

    def __rmul__(self, other):
        other = Scale(other)
        return Scale([scl_l * scl_r for scl_l, scl_r in zip(self.scale, other.scale)])

    def __floordiv__(self, other):
        other = Scale(other)
        return Scale([scl_l // scl_r for scl_l, scl_r in zip(self.scale, other.scale)])

    def __ifloordiv__(self, other):
        other = Scale(other)
        return Scale(tuple(
            [scl_l // scl_r for scl_l, scl_r in zip(self.scale, other.scale)]
        ))

    def __getitem__(self, ind) -> int:
        return self.scale[ind]

    def __hash__(self):
        return hash(self.scale)

    def __str__(self):
        return str(self.scale)

    def __repr__(self):
        return str(self.scale)
