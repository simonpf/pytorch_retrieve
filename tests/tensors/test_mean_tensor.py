"""
Tests for the 'pytorch_retrieve.tensors' module.
"""
import numpy as np
import torch
from torch import nn

import pytest

from pytorch_retrieve.tensors import MeanTensor, MaskedTensor
from pytorch_retrieve.modules.transformations import SquareRoot


def test_cat():
    """
    Test concatenation of mean tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mean_tensor = MeanTensor(tensor)

    mean_tensor_2 = torch.cat([mean_tensor, mean_tensor], 1)
    assert mean_tensor_2.shape[1] == 20

    mean_tensor_2 = torch.cat([mean_tensor, tensor], 1)
    assert mean_tensor_2.shape[1] == 20

    mean_tensor_2 = torch.cat([tensor, mean_tensor, tensor], 1)
    assert mean_tensor_2.shape[1] == 30

    torch.cat([tensor, mean_tensor], 1, out=mean_tensor_2)
    assert mean_tensor_2.shape[1] == 20


def test_stack():
    """
    Test stacking of mean tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mean_tensor = MeanTensor(tensor)

    mean_tensor_2 = torch.stack([mean_tensor, mean_tensor], 1)
    assert mean_tensor_2.shape[1] == 2

    mean_tensor_2 = torch.stack([mean_tensor, tensor], 1)
    assert mean_tensor_2.shape[1] == 2

    mean_tensor_2 = torch.stack([tensor, mean_tensor, tensor], 1)
    assert mean_tensor_2.shape[1] == 3

    torch.stack([tensor, mean_tensor], 1, out=mean_tensor_2)
    assert mean_tensor_2.shape[1] == 2


def test_add():
    """
    Test addition of mean tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mean_tensor_1 = MeanTensor(tensor)
    mean_tensor_2 = MeanTensor(tensor)

    mean_tensor_2 = mean_tensor_1 + mean_tensor_2
    assert torch.isclose(mean_tensor_2, 2.0 * tensor).all()


def test_sub():
    """
    Test difference of mean tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mean = torch.rand(10, 10, 10) - 0.5 > 0
    mean = torch.rand(10, 10, 10) - 0.5 > 0
    mean_tensor_1 = MeanTensor(tensor)
    mean_tensor_2 = MeanTensor(tensor)

    mean_tensor_2 = torch.sub(mean_tensor_1, mean_tensor_2)
    assert torch.isclose(mean_tensor_2, torch.zeros_like(mean_tensor_2)).all()


def test_mul():
    """
    Test multiplication of mean tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mean_tensor_1 = MeanTensor(tensor)
    mean_tensor_2 = MeanTensor(tensor)

    mean_tensor_2 = mean_tensor_1 * mean_tensor_2
    assert torch.isclose(mean_tensor_2, tensor**2).all()


def test_pow():
    """
    Test exponentiation of mean tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask_1 = torch.rand(10, 10, 10) - 0.5 > 0
    mask_2 = torch.rand(10, 10, 10) - 0.5 > 0
    mean_tensor_1 = MeanTensor(tensor)

    mean_tensor_2 = torch.pow(mean_tensor_1, 2)
    assert torch.isclose(mean_tensor_2, tensor**2).all()

    # mean_tensor_2 = torch._C._TensorBase.pow(mean_tensor_2, 2)
    mean_tensor_2 = mean_tensor_2**2
    assert torch.isclose(mean_tensor_2, tensor**4).all()


def test_permute():
    """
    Test permutation of mean tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    mean_tensor_1 = MeanTensor(tensor)
    mean_tensor_2 = torch.permute(mean_tensor_1, (2, 1, 0))
    assert mean_tensor_2.shape == (3, 2, 1)


def test_reshape():
    """
    Test reshaping of mean tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mean_tensor_1 = MeanTensor(tensor)
    mean_tensor_2 = torch.reshape(mean_tensor_1, (3, 2, 1))
    assert mean_tensor_2.shape == (3, 2, 1)


def test_view():
    """
    Test view applied to mean tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mean_tensor_1 = MeanTensor(tensor)
    mean_tensor_2 = mean_tensor_1.view((3, 2, 1))
    assert mean_tensor_2.shape == (3, 2, 1)


def test_squeeze():
    """
    Test squeezing of mean tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mean_tensor_1 = MeanTensor(tensor)
    mean_tensor_2 = mean_tensor_1.squeeze()
    assert mean_tensor_2.shape == (2, 3)

    mean_tensor_2 = torch.squeeze(mean_tensor_1)
    assert mean_tensor_2.shape == (2, 3)


def test_unsqueeze():
    """
    Test unsqueezing of mean tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mean_tensor_1 = MeanTensor(tensor)
    mean_tensor_2 = mean_tensor_1.unsqueeze(0)
    assert mean_tensor_2.shape == (1, 1, 2, 3)

    mean_tensor_2 = torch.unsqueeze(mean_tensor_1, 0)
    assert mean_tensor_2.shape == (1, 1, 2, 3)


def test_sum():
    """
    Test summing of tensor elements.
    """
    tensor = torch.rand(1, 2, 3)
    mean_tensor_1 = MeanTensor(tensor)
    mean_sum = mean_tensor_1.sum()


def test_mean():
    """
    Test calculating the mean of a mean tensor.
    """
    tensor = torch.rand(1, 2, 3)
    mean_tensor_1 = MeanTensor(tensor)

    mean_mean = mean_tensor_1.mean()


def test_gt():
    """
    Test calculating the mean of a mean tensor.
    """
    tensor = torch.rand(1, 2, 3)
    mean_tensor_1 = MeanTensor(tensor)

    mean_tensor_1 > 1


def test_dim():
    """
    Test for the dim method.
    """
    tensor = torch.rand(10, 10, 10)
    mean_tensor = MeanTensor(tensor)
    assert mean_tensor.dim() == 3


def test_tensor_ops():
    """
    Test basic operations with tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mean_tensor = MeanTensor(tensor)

    mean_tensor_2 = MeanTensor(mean_tensor)

    tensor_2 = torch.rand(10, 10, 10)
    mean_tensor_3 = MeanTensor(tensor_2)

    mean_tensor_4 = mean_tensor_2 + mean_tensor_3

    mean_tensor_5 = mean_tensor_2 * mean_tensor_3

    mean_tensor_5 = mean_tensor[:5, :5]
    assert mean_tensor_5.shape == (5, 5, 10)

    mean_tensor_6 = torch.permute(mean_tensor_5, (2, 0, 1))

    mean_tensor_8 = torch.stack([mean_tensor_6, mean_tensor_6])
    assert mean_tensor_8.shape == (2, 10, 5, 5)


def test_mse_loss():
    """
    Tests for the MSE loss function.
    """
    y_pred = MeanTensor(torch.ones(10, 1))
    y_true = torch.zeros(10)
    mse_loss = y_pred.loss(y_true)
    assert torch.isclose(mse_loss, torch.tensor(1.0))

    y_pred = 0.5 * MeanTensor(torch.ones(10, 1))
    y_true = torch.zeros(10)
    mse_loss = y_pred.loss(y_true)
    assert torch.isclose(mse_loss, torch.tensor(0.25))


def test_masked_mse_loss():
    """
    Tests for the MSE loss function with masked tensors.
    """
    y_pred = MeanTensor(torch.ones(10, 1))
    y_true = torch.zeros(10)
    mask = torch.rand(10) > 0.5
    y_true[mask] = torch.nan
    y_true = MaskedTensor(y_true, mask=mask)
    mse_loss = y_pred.loss(y_true)
    assert torch.isclose(mse_loss, torch.tensor(1.0))


def test_unbind():
    """
    Ensure that unbinding yields a mean tensor.
    """
    y_pred = MeanTensor(torch.ones(10, 1))
    tensors = y_pred.unbind(1)
    assert isinstance(tensors[0], MeanTensor)
    tensors = torch.unbind(y_pred, 1)
    assert isinstance(tensors[0], MeanTensor)


def test_any_all():
    """
    Test any and all operations.
    """
    tensor_1 = torch.rand(4, 4, 4)
    tensor_1 = MeanTensor(tensor_1)

    assert (tensor_1 <= 1.0).all()
    assert not (tensor_1 > 1.0).any()

    assert torch.all(tensor_1 <= 1.0)
    assert not torch.any(tensor_1 > 1.0)


def test_loss():
    """
    Test calculation of MSE.
    """
    tensor_1 = MeanTensor(torch.zeros((4, 8, 8)))
    tensor_2 = torch.ones((4, 8, 8))
    loss = tensor_1.loss(tensor_2)
    assert torch.isclose(loss, torch.tensor(1.0))

    weights = torch.zeros_like(tensor_2)
    weights[2:] = 1.0
    loss = tensor_1.loss(tensor_2, weights=weights)
    assert torch.isclose(loss, torch.tensor(1.0))

    tensor_2[2:] = 0.0
    loss = tensor_1.loss(tensor_2, weights=weights)
    assert torch.isclose(loss, torch.tensor(0.0))

    with pytest.raises(ValueError):
        weights = torch.zeros((1,))
        loss = tensor_1.loss(tensor_2, weights=weights)


def test_transformation():
    """
    Ensur that transformations are passed on when new tensors are created.
    """
    tnsr = MeanTensor(torch.rand(4, 32, 64), transformation=SquareRoot)
    assert tnsr.__transformation__ is not None

    tnsr_new = tnsr.reshape(4, 32, 8, 8)
    assert tnsr_new.__transformation__ is not None

    tnsr_new = tnsr_new + tnsr_new
    assert tnsr_new.__transformation__ is not None
