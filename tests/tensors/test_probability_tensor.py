"""
Tests for the 'pytorch_retrieve.tensors' module.
"""
import numpy as np
import pytest
from scipy.stats import norm
import torch
from torch import nn


from pytorch_retrieve.tensors import ProbabilityTensor, MaskedTensor
from pytorch_retrieve.modules.transformations import SquareRoot


def test_cat():
    """
    Test concatenation of probability tensors.
    """
    tensor = torch.rand(10, 10, 10)
    bins = torch.linspace(0, 1, 12)[1:-1]
    prob_tensor = ProbabilityTensor(tensor, bins)

    prob_tensor_2 = torch.cat([prob_tensor, prob_tensor], 1)
    assert prob_tensor_2.shape[1] == 20

    prob_tensor_2 = torch.cat([prob_tensor, tensor], 1)
    assert prob_tensor_2.shape[1] == 20

    prob_tensor_2 = torch.cat([tensor, prob_tensor, tensor], 1)
    assert prob_tensor_2.shape[1] == 30

    torch.cat([tensor, prob_tensor], 1, out=prob_tensor_2)
    assert prob_tensor_2.shape[1] == 20


def test_stack():
    """
    Test stacking of probability tensors.
    """
    tensor = torch.rand(10, 10, 10)
    bins = torch.linspace(0, 1, 12)[1:-1]
    prob_tensor = ProbabilityTensor(tensor, bins)

    prob_tensor_2 = torch.stack([prob_tensor, prob_tensor], 1)
    assert prob_tensor_2.shape[1] == 2

    prob_tensor_2 = torch.stack([prob_tensor, tensor], 1)
    assert prob_tensor_2.shape[1] == 2

    prob_tensor_2 = torch.stack([tensor, prob_tensor, tensor], 1)
    assert prob_tensor_2.shape[1] == 3

    torch.stack([tensor, prob_tensor], 1, out=prob_tensor_2)
    assert prob_tensor_2.shape[1] == 2


def test_add():
    """
    Test addition of probability tensors.
    """
    tensor = torch.rand(10, 10, 10)
    bins = torch.linspace(0, 1, 12)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)
    prob_tensor_2 = ProbabilityTensor(tensor, bins)

    prob_tensor_2 = prob_tensor_1 + prob_tensor_2
    assert torch.isclose(prob_tensor_2, 2.0 * tensor).all()


def test_sub():
    """
    Test difference of probability tensors.
    """
    tensor = torch.rand(10, 10, 10)
    bins = torch.linspace(0, 1, 12)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)
    prob_tensor_2 = ProbabilityTensor(tensor, bins)

    prob_tensor_2 = torch.sub(prob_tensor_1, prob_tensor_2)
    assert torch.isclose(prob_tensor_2, torch.zeros_like(prob_tensor_2)).all()


def test_mul():
    """
    Test multiplication of probability tensors.
    """
    tensor = torch.rand(10, 10, 10)
    bins = torch.linspace(0, 1, 12)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)
    prob_tensor_2 = ProbabilityTensor(tensor, bins)

    prob_tensor_2 = prob_tensor_1 * prob_tensor_2
    assert torch.isclose(prob_tensor_2, tensor**2).all()


def test_pow():
    """
    Test exponentiation of probability tensors.
    """
    tensor = torch.rand(10, 10, 10)
    bins = torch.linspace(0, 1, 12)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)

    prob_tensor_2 = torch.pow(prob_tensor_1, 2)
    assert torch.isclose(prob_tensor_2, tensor**2).all()

    # prob_tensor_2 = torch._C._TensorBase.pow(prob_tensor_2, 2)
    prob_tensor_2 = prob_tensor_2**2
    assert torch.isclose(prob_tensor_2, tensor**4).all()


def test_permute():
    """
    Test permutation of probability tensors.
    """
    tensor = torch.rand(1, 2, 3)
    bins = torch.linspace(0, 1, 4)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)
    prob_tensor_2 = torch.permute(prob_tensor_1, (2, 1, 0))
    assert prob_tensor_2.shape == (3, 2, 1)


def test_reshape():
    """
    Test reshaping of probability tensors.
    """
    tensor = torch.rand(1, 2, 3)
    bins = torch.linspace(0, 1, 4)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)
    prob_tensor_2 = torch.reshape(prob_tensor_1, (3, 2, 1))
    assert prob_tensor_2.shape == (3, 2, 1)


def test_view():
    """
    Test view applied to probability tensors.
    """
    tensor = torch.rand(1, 2, 3)
    bins = torch.linspace(0, 1, 4)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)
    prob_tensor_2 = prob_tensor_1.view((3, 2, 1))
    assert prob_tensor_2.shape == (3, 2, 1)


def test_squeeze():
    """
    Test squeezing of probability tensors.
    """
    tensor = torch.rand(1, 2, 3)
    bins = torch.linspace(0, 1, 4)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)
    prob_tensor_2 = prob_tensor_1.squeeze()
    assert prob_tensor_2.shape == (2, 3)

    prob_tensor_2 = torch.squeeze(prob_tensor_1)
    assert prob_tensor_2.shape == (2, 3)


def test_unsqueeze():
    """
    Test unsqueezing of probability tensors.
    """
    tensor = torch.rand(1, 2, 3)
    bins = torch.linspace(0, 1, 4)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)
    prob_tensor_2 = prob_tensor_1.unsqueeze(0)
    assert prob_tensor_2.shape == (1, 1, 2, 3)

    prob_tensor_2 = torch.unsqueeze(prob_tensor_1, 0)
    assert prob_tensor_2.shape == (1, 1, 2, 3)


def test_sum():
    """
    Test summing of tensor elements.
    """
    tensor = torch.rand(1, 2, 3)
    bins = torch.linspace(0, 1, 4)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)
    probability_sum = prob_tensor_1.sum()


def test_mean():
    """
    Test calculating the mean of a probability tensor.
    """
    tensor = torch.rand(1, 2, 3)
    bins = torch.linspace(0, 1, 4)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)

    mean = prob_tensor_1.mean()


def test_gt():
    """
    Test greater than operation on a probability tensor.
    """
    tensor = torch.rand(1, 2, 3)
    bins = torch.linspace(0, 1, 4)[1:-1]
    prob_tensor_1 = ProbabilityTensor(tensor, bins)

    prob_tensor_1 > 1


def test_tensor_ops():
    """
    Test basic operations with tensors.
    """
    tensor = torch.rand(10, 10, 10)
    bins = torch.linspace(0, 1, 4)[1:-1]
    prob_tensor = ProbabilityTensor(tensor, bins)

    prob_tensor_2 = ProbabilityTensor(prob_tensor, bins)

    tensor_2 = torch.rand(10, 10, 10)
    prob_tensor_3 = ProbabilityTensor(tensor_2, bins)

    prob_tensor_4 = prob_tensor_2 + prob_tensor_3

    prob_tensor_5 = prob_tensor_2 * prob_tensor_3

    prob_tensor_5 = prob_tensor[:5, :5]

    prob_tensor_6 = torch.permute(prob_tensor_5, (2, 0, 1))

    prob_tensor_8 = torch.stack([prob_tensor_6, prob_tensor_6])
    assert prob_tensor_8.shape == (2, 10, 5, 5)


def test_probability_loss():
    """
    Tests for the probability loss function.
    """
    bins = torch.linspace(0, 1, 11)
    y_pred = ProbabilityTensor(torch.zeros(10, 10), bins=bins)
    y_pred[..., -1] = 100.0
    y_true = 0.95 * torch.ones(10)
    p_loss = y_pred.loss(y_true)
    assert torch.isclose(p_loss, torch.tensor(0.0))

    y_true = 0.95 * torch.ones(10)
    mask = torch.linspace(0, 1, 10) > 0.5
    y_true[mask] = torch.nan
    y_true = MaskedTensor(y_true, mask=mask)
    p_loss = y_pred.loss(y_true)
    assert torch.isclose(p_loss, torch.tensor(0.0))


def test_pdf():
    """
    Test that PDF is correctly calculated.
    """
    bins = np.linspace(0, 1, 11)
    probabilities = torch.zeros(10, 10, 10)
    probabilities[:, 0] = 100.0
    prob_tensor = ProbabilityTensor(probabilities, bins=bins, bin_dim=1)
    x_pdf, y_pdf = prob_tensor.pdf()
    assert torch.isclose(x_pdf[0, 0], torch.tensor(0.05)).all()
    assert torch.isclose(y_pdf[:, 0], torch.tensor(10.0)).all()


def test_cdf():
    """
    Test that CDF is correctly calculated.
    """
    bins = np.linspace(0, 1, 11)
    probabilities = torch.zeros(10, 10, 10)
    probabilities[:, 0] = 100.0
    prob_tensor = ProbabilityTensor(probabilities, bins=bins, bin_dim=1)
    x_cdf, y_cdf = prob_tensor.cdf()
    assert torch.isclose(x_cdf[0, 0], torch.tensor(0.0)).all()
    assert torch.isclose(x_cdf[0, 1], torch.tensor(0.1)).all()
    assert torch.isclose(y_cdf[:, 0], torch.tensor(0.0)).all()
    assert torch.isclose(y_cdf[:, 1], torch.tensor(1.0)).all()


def test_expected_value():
    """
    Test that the expected value is calculated correctly.
    """
    bins = np.linspace(0, 1, 11)
    probabilities = torch.zeros(10, 10, 10)
    probabilities[:, 0] = 100.0
    prob_tensor = ProbabilityTensor(probabilities, bins=bins, bin_dim=1)
    exp = prob_tensor.expected_value()
    assert np.isclose(exp, torch.tensor(0.05)).all()


def test_maximum_probability():
    """
    Test that the expected value is calculated correctly.
    """
    bins = np.linspace(0, 1, 11)
    probabilities = torch.zeros(10, 10, 10)
    probabilities[:, 0] = 100.0
    prob_tensor = ProbabilityTensor(probabilities, bins=bins, bin_dim=1)
    exp = prob_tensor.maximum_probability()
    assert np.isclose(exp, torch.tensor(0.05)).all()


def test_probability_less_and_greater_than():
    bins = np.linspace(0, 10, 11)
    # Uniform distribution
    probabilities = torch.zeros(10, 10, 10)
    prob_tensor = ProbabilityTensor(probabilities, bins=bins, bin_dim=1)

    for prob in np.linspace(0, 1, 10).astype(np.float32):
        p_less = prob_tensor.probability_less_than(prob)
        assert np.isclose(p_less.numpy(), prob / 10.0).all()
        p_greater = prob_tensor.probability_greater_than(prob)
        assert np.isclose(p_less.numpy() + p_greater.numpy(), 1.0).all()


def test_any_all():
    """
    Test any and all operations.
    """
    bins = np.linspace(0, 10, 11)
    tensor_1 = torch.rand(4, 32, 4)
    tensor_1 = ProbabilityTensor(tensor_1, bins=bins, bin_dim=1)

    assert (tensor_1 <= 1.0).all()
    assert not (tensor_1 > 1.0).any()

    assert torch.all(tensor_1 <= 1.0)
    assert not torch.any(tensor_1 > 1.0)


pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs GPU.")


def test_to():
    """
    Test .to method.
    """
    tensor = 100 * torch.rand(100, 3, 100)
    bins = np.linspace(0, 10, 4)
    tensor = ProbabilityTensor(tensor, bins=bins)

    tensor = tensor.to(dtype=torch.bfloat16)
    assert tensor.dtype == torch.bfloat16
    assert tensor.bins.dtype == torch.bfloat16

    tensor = tensor.to(dtype=torch.float32)
    assert tensor.dtype == torch.float32
    assert tensor.bins.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs GPU.")
def test_cpu():
    """
    Test .cpu method.
    """
    tensor = 100 * torch.rand(100, 3, 100)
    bins = np.linspace(0, 10, 4)
    tensor = ProbabilityTensor(tensor, bins=bins)

    tensor = tensor.to("cuda:0")
    assert tensor.device == torch.device("cuda:0")
    assert tensor.bins.device == torch.device("cuda:0")

    tensor = tensor.cpu()
    assert tensor.device == torch.device("cpu")
    assert tensor.bins.device == torch.device("cpu")


def test_loss():
    """
    Test loss calculation.
    """
    tensor_1 = torch.zeros(100, 2, 100)
    tensor_1[:, 1] = 1.0
    bins = np.linspace(0, 10, 3)
    tensor_1 = ProbabilityTensor(tensor_1, bins=bins)

    tensor_2 = torch.zeros(100, 100)

    loss = tensor_1.loss(tensor_2)
    assert torch.isclose(
        loss, torch.tensor(-np.log(1 / (1.0 + np.exp(1.0))).astype(np.float32))
    )

    tensor_2[:50] = 10.0
    loss = tensor_1.loss(tensor_2)
    assert not torch.isclose(
        loss, torch.tensor(-np.log(1 / (1.0 + np.exp(1.0))).astype(np.float32))
    )

    weights = torch.ones_like(tensor_2)
    weights[:50] = 0.0
    loss = tensor_1.loss(tensor_2, weights=weights)
    assert torch.isclose(
        loss, torch.tensor(-np.log(1 / (1.0 + np.exp(1.0))).astype(np.float32))
    )

    with pytest.raises(ValueError):
        weights = torch.zeros((1,))
        loss = tensor_1.loss(tensor_2, weights=weights)


def test_transformation():
    """
    Ensur that transformations are passed on when new tensors are created.
    """
    bins = np.linspace(0, 10, 33)
    tnsr = ProbabilityTensor(torch.rand(4, 32, 64), bins=bins, transformation=SquareRoot())
    assert tnsr.__transformation__ is not None

    tnsr_new = tnsr.reshape(4, 32, 8, 8)
    assert tnsr_new.__transformation__ is not None

    tnsr_new = tnsr_new + tnsr_new
    assert tnsr_new.__transformation__ is not None
