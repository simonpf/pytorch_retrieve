"""
Tests for the 'pytorch_retrieve.tensors.classification' module.
"""
import pytest
import torch

from pytorch_retrieve.tensors.masked_tensor import MaskedTensor
from pytorch_retrieve.tensors.classification import (
    DetectionTensor,
    ClassificationTensor,
)


def test_detection_tensor():
    """
    Ensure that masked values are ignored in calculation of masked binary
    cross entropy.
    """
    tensor = DetectionTensor(100 * torch.rand(10, 10, 10))
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    target = torch.ones_like(tensor.base)
    target[mask] = 0
    target = MaskedTensor(target, mask=mask)
    loss = tensor.loss(target)
    loss_masked = tensor[~mask].loss(target[~mask])
    assert torch.isclose(loss, loss_masked)

    probs = tensor.probability()
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_classification_tensor():
    """
    Ensure that masked values are ignored in calculation of masked binary
    cross entropy.
    """
    tensor = ClassificationTensor(100 * torch.rand(10, 20, 30))

    target = torch.ones(10, 30).to(dtype=torch.int64)
    mask = torch.rand(10, 30) - 0.5 > 0
    target[mask] = 0
    target = MaskedTensor(target, mask=mask)

    loss = tensor.loss(target)
    loss_masked = tensor.transpose(1, -1)[~mask].loss(target[~mask])
    assert torch.isclose(loss, loss_masked)

    probs = tensor.probability()
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_any_all():
    """
    Test any and all operations.
    """
    tensor_1 = torch.rand(4, 32, 4)
    tensor_1 = ClassificationTensor(tensor_1)

    assert (tensor_1 <= 1.0).all()
    assert not (tensor_1 > 1.0).any()

    assert torch.all(tensor_1 <= 1.0)
    assert not torch.any(tensor_1 > 1.0)

    tensor_1 = DetectionTensor(tensor_1.base)

    assert (tensor_1 <= 1.0).all()
    assert not (tensor_1 > 1.0).any()

    assert torch.all(tensor_1 <= 1.0)
    assert not torch.any(tensor_1 > 1.0)


def test_detection_loss():
    """
    Test any and all operations.
    """
    tensor_1 = torch.rand(4, 32)
    tensor_1 = DetectionTensor(tensor_1)

    tensor_2 = torch.ones_like(tensor_1)

    loss = tensor_1.loss(tensor_2)

    loss_ref = -torch.log(torch.exp(tensor_1) / (1.0 + torch.exp(tensor_1))).mean()
    assert torch.isclose(loss, loss_ref)

    weights = torch.ones_like(tensor_2)
    loss = tensor_1.loss(tensor_2, weights=weights)
    assert torch.isclose(loss, loss_ref)

    weights[2:] = 0.0
    loss = tensor_1.loss(tensor_2, weights=weights)
    assert not torch.isclose(loss, loss_ref)

    loss_ref = -torch.log(
        torch.exp(tensor_1[:2]) / (1.0 + torch.exp(tensor_1[:2]))
    ).mean()
    assert torch.isclose(loss, loss_ref)

    with pytest.raises(ValueError):
        weights = torch.zeros((1,))
        loss = tensor_1.loss(tensor_2, weights=weights)


def test_classfication_loss():
    """
    Test any and all operations.
    """
    tensor_1 = torch.rand(4, 4, 32)
    tensor_1 = ClassificationTensor(tensor_1)

    tensor_2 = torch.zeros((4, 32), dtype=torch.int64)

    loss = tensor_1.loss(tensor_2)

    loss_ref = -torch.log(
        torch.exp(tensor_1[:, 0]) / (torch.exp(tensor_1).sum(1))
    ).mean()
    assert torch.isclose(loss, loss_ref)

    weights = torch.ones_like(tensor_2)
    loss = tensor_1.loss(tensor_2, weights=weights)
    assert torch.isclose(loss, loss_ref)

    weights[2:] = 0.0
    loss = tensor_1.loss(tensor_2, weights=weights)
    assert not torch.isclose(loss, loss_ref)

    loss_ref = -torch.log(
        torch.exp(tensor_1[:2, 0]) / (torch.exp(tensor_1[:2]).sum(1))
    ).mean()
    assert torch.isclose(loss, loss_ref)

    with pytest.raises(ValueError):
        weights = torch.zeros((1,))
        loss = tensor_1.loss(tensor_2, weights=weights)
