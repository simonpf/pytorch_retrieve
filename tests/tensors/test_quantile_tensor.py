"""
Tests for the 'pytorch_retrieve.tensors' module.
"""
import numpy as np
from scipy.stats import norm
import torch
from torch import nn

from pytorch_retrieve.tensors import QuantileTensor, MaskedTensor


def test_cat():
    """
    Test concatenation of quantile tensors.
    """
    tensor = torch.rand(10, 10, 10)
    tau = torch.linspace(0, 1, 12)[1:-1]
    quantile_tensor = QuantileTensor(tensor, tau)

    quantile_tensor_2 = torch.cat([quantile_tensor, quantile_tensor], 1)
    assert quantile_tensor_2.shape[1] == 20

    quantile_tensor_2 = torch.cat([quantile_tensor, tensor], 1)
    assert quantile_tensor_2.shape[1] == 20

    quantile_tensor_2 = torch.cat([tensor, quantile_tensor, tensor], 1)
    assert quantile_tensor_2.shape[1] == 30

    torch.cat([tensor, quantile_tensor], 1, out=quantile_tensor_2)
    assert quantile_tensor_2.shape[1] == 20


def test_stack():
    """
    Test stacking of quantile tensors.
    """
    tensor = torch.rand(10, 10, 10)
    tau = torch.linspace(0, 1, 12)[1:-1]
    quantile_tensor = QuantileTensor(tensor, tau)

    quantile_tensor_2 = torch.stack([quantile_tensor, quantile_tensor], 1)
    assert quantile_tensor_2.shape[1] == 2

    quantile_tensor_2 = torch.stack([quantile_tensor, tensor], 1)
    assert quantile_tensor_2.shape[1] == 2

    quantile_tensor_2 = torch.stack([tensor, quantile_tensor, tensor], 1)
    assert quantile_tensor_2.shape[1] == 3

    torch.stack([tensor, quantile_tensor], 1, out=quantile_tensor_2)
    assert quantile_tensor_2.shape[1] == 2


def test_add():
    """
    Test addition of quantile tensors.
    """
    tensor = torch.rand(10, 10, 10)
    tau = torch.linspace(0, 1, 12)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)
    quantile_tensor_2 = QuantileTensor(tensor, tau)

    quantile_tensor_2 = quantile_tensor_1 + quantile_tensor_2
    assert torch.isclose(quantile_tensor_2, 2.0 * tensor).all()


def test_sub():
    """
    Test difference of quantile tensors.
    """
    tensor = torch.rand(10, 10, 10)
    tau = torch.linspace(0, 1, 12)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)
    quantile_tensor_2 = QuantileTensor(tensor, tau)

    quantile_tensor_2 = torch.sub(quantile_tensor_1, quantile_tensor_2)
    assert torch.isclose(quantile_tensor_2, torch.zeros_like(quantile_tensor_2)).all()


def test_mul():
    """
    Test multiplication of quantile tensors.
    """
    tensor = torch.rand(10, 10, 10)
    tau = torch.linspace(0, 1, 12)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)
    quantile_tensor_2 = QuantileTensor(tensor, tau)

    quantile_tensor_2 = quantile_tensor_1 * quantile_tensor_2
    assert torch.isclose(quantile_tensor_2, tensor**2).all()


def test_pow():
    """
    Test exponentiation of quantile tensors.
    """
    tensor = torch.rand(10, 10, 10)
    tau = torch.linspace(0, 1, 12)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)

    quantile_tensor_2 = torch.pow(quantile_tensor_1, 2)
    assert torch.isclose(quantile_tensor_2, tensor**2).all()

    # quantile_tensor_2 = torch._C._TensorBase.pow(quantile_tensor_2, 2)
    quantile_tensor_2 = quantile_tensor_2**2
    assert torch.isclose(quantile_tensor_2, tensor**4).all()


def test_permute():
    """
    Test permutation of quantile tensors.
    """
    tensor = torch.rand(1, 2, 3)
    tau = torch.linspace(0, 1, 4)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)
    quantile_tensor_2 = torch.permute(quantile_tensor_1, (2, 1, 0))
    assert quantile_tensor_2.shape == (3, 2, 1)


def test_reshape():
    """
    Test reshaping of quantile tensors.
    """
    tensor = torch.rand(1, 2, 3)
    tau = torch.linspace(0, 1, 4)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)
    quantile_tensor_2 = torch.reshape(quantile_tensor_1, (3, 2, 1))
    assert quantile_tensor_2.shape == (3, 2, 1)


def test_view():
    """
    Test view applied to quantile tensors.
    """
    tensor = torch.rand(1, 2, 3)
    tau = torch.linspace(0, 1, 4)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)
    quantile_tensor_2 = quantile_tensor_1.view((3, 2, 1))
    assert quantile_tensor_2.shape == (3, 2, 1)


def test_squeeze():
    """
    Test squeezing of quantile tensors.
    """
    tensor = torch.rand(1, 2, 3)
    tau = torch.linspace(0, 1, 4)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)
    quantile_tensor_2 = quantile_tensor_1.squeeze()
    assert quantile_tensor_2.shape == (2, 3)

    quantile_tensor_2 = torch.squeeze(quantile_tensor_1)
    assert quantile_tensor_2.shape == (2, 3)


def test_unsqueeze():
    """
    Test unsqueezing of quantile tensors.
    """
    tensor = torch.rand(1, 2, 3)
    tau = torch.linspace(0, 1, 4)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)
    quantile_tensor_2 = quantile_tensor_1.unsqueeze(0)
    assert quantile_tensor_2.shape == (1, 1, 2, 3)

    quantile_tensor_2 = torch.unsqueeze(quantile_tensor_1, 0)
    assert quantile_tensor_2.shape == (1, 1, 2, 3)


def test_sum():
    """
    Test summing of tensor elements.
    """
    tensor = torch.rand(1, 2, 3)
    tau = torch.linspace(0, 1, 4)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)
    quantile_sum = quantile_tensor_1.sum()


def test_mean():
    """
    Test calculating the mean of a quantile tensor.
    """
    tensor = torch.rand(1, 2, 3)
    tau = torch.linspace(0, 1, 4)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)

    mean = quantile_tensor_1.mean()


def test_gt():
    """
    Test greater than operation on a quantile tensor.
    """
    tensor = torch.rand(1, 2, 3)
    tau = torch.linspace(0, 1, 4)[1:-1]
    quantile_tensor_1 = QuantileTensor(tensor, tau)

    quantile_tensor_1 > 1


def test_tensor_ops():
    """
    Test basic operations with tensors.
    """
    tensor = torch.rand(10, 10, 10)
    tau = torch.linspace(0, 1, 4)[1:-1]
    quantile_tensor = QuantileTensor(tensor, tau)

    quantile_tensor_2 = QuantileTensor(quantile_tensor, tau)

    tensor_2 = torch.rand(10, 10, 10)
    quantile_tensor_3 = QuantileTensor(tensor_2, tau)

    quantile_tensor_4 = quantile_tensor_2 + quantile_tensor_3

    quantile_tensor_5 = quantile_tensor_2 * quantile_tensor_3

    quantile_tensor_5 = quantile_tensor[:5, :5]

    quantile_tensor_6 = torch.permute(quantile_tensor_5, (2, 0, 1))

    quantile_tensor_8 = torch.stack([quantile_tensor_6, quantile_tensor_6])
    assert quantile_tensor_8.shape == (2, 10, 5, 5)


def test_quantile_loss():
    """
    Tests for the quantile loss function.
    """
    tau = torch.tensor([0.5])
    y_pred = QuantileTensor(torch.ones(10, 1), tau=tau)
    y_true = torch.zeros(10)
    q_loss = y_pred.loss(y_true)
    ref = torch.abs(y_pred.base - q_loss).mean()
    assert torch.isclose(q_loss, ref)

    y_1 = torch.rand(10, 1, 10, 10)
    y_2 = torch.rand(10, 1, 10, 10)
    tau_1 = torch.tensor([0.1])
    y_pred_1 = QuantileTensor(y_1, tau=tau_1)
    y_true_1 = y_2

    tau_2 = torch.tensor([0.9])
    y_pred_2 = QuantileTensor(y_2, tau=tau_2)
    y_true_2 = y_1

    loss_1 = y_pred_1.loss(y_true_1)
    loss_2 = y_pred_2.loss(y_true_2)
    assert torch.isclose(loss_1, loss_2)

    y_1 = torch.rand(10, 10, 10, 1)
    y_2 = torch.rand(10, 10, 10, 1)
    tau_1 = torch.tensor([0.1])
    y_pred_1 = QuantileTensor(y_1, tau=tau_1, quantile_dim=-1)
    y_true_1 = y_2

    tau_2 = torch.tensor([0.9])
    y_pred_2 = QuantileTensor(y_2, tau=tau_2, quantile_dim=-1)
    y_true_2 = y_1

    loss_1 = y_pred_1.loss(y_true_1)
    loss_2 = y_pred_2.loss(y_true_2)
    assert torch.isclose(loss_1, loss_2)


def test_masked_quantile_loss():
    """
    Tests for the quantile loss function with masked tensors.
    """
    tau = torch.tensor([0.5])

    tensor = torch.ones(100, 1)
    mask = torch.rand(100, 1) > 0.5
    tensor[mask] = np.nan
    tensor = MaskedTensor(tensor, mask=mask)

    y_pred = QuantileTensor(tensor, tau=tau)

    y_true = torch.zeros(100)
    q_loss = y_pred.loss(y_true)
    ref = torch.abs(y_pred.base - q_loss).mean()
    assert torch.isclose(q_loss, ref)

    y_true = torch.zeros(100)
    mask = torch.rand(100) > 0.5
    y_true[mask] = np.nan
    y_true = MaskedTensor(y_true, mask=mask)

    q_loss = y_pred.loss(y_true)
    ref = torch.abs(y_pred.base - q_loss).mean()
    assert torch.isclose(q_loss, ref)

    y_1 = torch.rand(10, 1, 10, 10)
    y_2 = torch.rand(10, 1, 10, 10)
    tau_1 = torch.tensor([0.1])
    y_pred_1 = QuantileTensor(y_1, tau=tau_1)
    y_true_1 = y_2

    tau_2 = torch.tensor([0.9])
    y_pred_2 = QuantileTensor(y_2, tau=tau_2)
    y_true_2 = y_1

    loss_1 = y_pred_1.loss(y_true_1)
    loss_2 = y_pred_2.loss(y_true_2)
    assert torch.isclose(loss_1, loss_2)

    y_1 = torch.rand(10, 10, 10, 1)
    y_2 = torch.rand(10, 10, 10, 1)
    tau_1 = torch.tensor([0.1])
    y_pred_1 = QuantileTensor(y_1, tau=tau_1, quantile_dim=-1)
    y_true_1 = y_2

    tau_2 = torch.tensor([0.9])
    y_pred_2 = QuantileTensor(y_2, tau=tau_2, quantile_dim=-1)
    y_true_2 = y_1

    loss_1 = y_pred_1.loss(y_true_1)
    loss_2 = y_pred_2.loss(y_true_2)
    assert torch.isclose(loss_1, loss_2)


def test_pdf():
    tau = np.linspace(0, 1, 34)[1:-1]
    quantiles = torch.tensor(norm.ppf(tau))

    quantiles = torch.broadcast_to(
        quantiles[None, None, ..., None, None], [3, 4, 32, 5, 6]
    )
    quantile_tensor = QuantileTensor(quantiles, tau=tau, quantile_dim=2)

    x_pdf, y_pdf = quantile_tensor.pdf()
    y_pdf = y_pdf[1, 2, :, 3, 4]
    x_pdf = x_pdf[0, 0, :, 0, 0]
    y_pdf_ref = norm.pdf(x_pdf)

    assert np.all(np.isclose(y_pdf[1:-1].numpy(), y_pdf_ref[1:-1], atol=1e-2))


def test_cdf():
    tau = np.linspace(0, 1, 34)[1:-1]
    quantiles = torch.tensor(norm.ppf(tau))

    quantiles = torch.broadcast_to(
        quantiles[None, None, ..., None, None], [3, 4, 32, 5, 6]
    )
    quantile_tensor = QuantileTensor(quantiles, tau=tau, quantile_dim=2)

    x_cdf, y_cdf = quantile_tensor.cdf()
    x_pdf, y_pdf = quantile_tensor.pdf()
    x_pdf = x_pdf[0, 0, :, 0, 0]
    x_cdf = x_cdf[0, 0, :, 0, 0]

    assert np.all(np.isclose(x_pdf, 0.5 * (x_cdf[1:] + x_cdf[:-1])))


def test_probability_less_than():
    tau = np.linspace(0, 1, 34)[1:-1]
    quantiles = torch.tensor(norm.ppf(tau))

    quantiles = torch.broadcast_to(
        quantiles[None, None, ..., None, None], [3, 4, 32, 5, 6]
    )
    quantile_tensor = QuantileTensor(quantiles, tau=tau, quantile_dim=2)

    p = quantile_tensor.probability_less_than(0.0)
    assert np.all(np.isclose(p, 0.5))

    p = quantile_tensor.probability_less_than(-1)
    assert np.all(np.isclose(p, 0.5 - 0.341, rtol=0.01))

    p = quantile_tensor.probability_less_than(+1)
    assert np.all(np.isclose(p, 0.5 + 0.341, rtol=0.01))

    p = quantile_tensor.probability_less_than(3)
    assert np.all(np.isclose(p, 0.5 + 0.498, rtol=0.01))

    p = quantile_tensor.probability_less_than(-3)
    assert np.all(np.isclose(p, 0.5 - 0.498, atol=0.01))


def test_probability_greater_than():
    tau = np.linspace(0, 1, 34)[1:-1]
    quantiles = torch.tensor(norm.ppf(tau))

    quantiles = torch.broadcast_to(
        quantiles[None, None, ..., None, None], [3, 4, 32, 5, 6]
    )
    quantile_tensor = QuantileTensor(quantiles, tau=tau, quantile_dim=2)

    for thresh in [-3, -1, 0, 1, 3]:
        p_less = quantile_tensor.probability_less_than(0.0)
        p_greater = quantile_tensor.probability_greater_than(0.0)
        assert np.all(np.isclose(p_less + p_greater, 1.0, rtol=0.01))
