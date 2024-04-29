
import numpy as np
import torch

from pytorch_retrieve.modules.transformations import SquareRoot
from pytorch_retrieve.modules.output import Mean, Quantiles
from pytorch_retrieve.tensors import MeanTensor, QuantileTensor


def test_mean_output_layer():
    """
    Test that the mean output layer produces a mean tensor.
    """
    output_layer = Mean("x", 1)
    x = torch.rand(10, 10, 10)

    x_out = output_layer(x)

    assert (x == x_out).all()
    assert (x == x_out.expected_value()).all()
    assert isinstance(x_out, MeanTensor)

    output_layer = Mean("x", 1, transformation=SquareRoot())
    x_out = output_layer(x)
    assert hasattr(x_out, "__transformation__")
    assert x_out.__transformation__ is output_layer.transformation


def test_quantiles_output_layer():
    """
    Test that the mean output layer produces a mean tensor.
    """
    output_layer = Quantiles("x", 1, np.linspace(0, 1, 12)[1:-1])
    x = torch.rand(10, 10, 10)

    x_out = output_layer(x)

    assert isinstance(x_out, QuantileTensor)
    assert (x == x_out).all()
    assert x_out.expected_value().shape == (10, 10)

    output_layer = Quantiles("x", 1, np.linspace(0, 1, 12)[1:-1], transformation=SquareRoot())
    x_out = output_layer(x)
    assert hasattr(x_out, "__transformation__")
    assert x_out.__transformation__ is output_layer.transformation
