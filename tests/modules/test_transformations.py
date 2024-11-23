"""
Tests for the pytorch_retrieve.modules.transformations module
"""
import pytest
import torch
from torch import nn

from pytorch_retrieve.modules.transformations import (
    SquareRoot,
    Log,
    LogLinear,
    MinMax
)

@pytest.mark.parametrize("transformation", [SquareRoot(), Log(), LogLinear(), MinMax(100.0, 200.0)])
def test_transformations(transformation):

    x_ref = 1e3 * torch.rand(10, 10, 10) + 1e-6
    y = transformation(x_ref)
    x = transformation.invert(y)

    assert torch.isclose(x, x_ref).all()
