from conftest import data_loader_1d

import torch

from pytorch_retrieve.metrics import Bias, CorrelationCoef
from pytorch_retrieve.tensors import MaskedTensor


def test_bias():
    """
    Test that calculating the correlation coefficient works with masked tensors.
    """
    bias = Bias()
    data_loader = data_loader_1d(1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        bias.update(y, y)

    result = bias.compute()
    assert torch.isclose(result, torch.tensor(0.0))


def test_correlation_coef():
    """
    Test that calculating the correlation coefficient works with masked tensors.
    """
    corr = CorrelationCoef()
    data_loader = data_loader_1d(1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        corr.update(y, y)

    result = corr.compute()
    assert torch.isclose(result, torch.tensor(1.0))

    corr.reset()
    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        corr.update(y, -y)

    result = corr.compute()
    assert torch.isclose(result, torch.tensor(-1.0))
