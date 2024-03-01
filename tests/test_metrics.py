from conftest import data_loader_1d, data_loader_3d

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import pytest
import torch

from pytorch_retrieve.metrics import Bias, CorrelationCoef, PlotSamples
from pytorch_retrieve.tensors import MaskedTensor, MeanTensor


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


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="needs matplotlib")
def test_plot_samples():
    """
    Test the worst sample metric.
    """

    metric = PlotSamples()
    data_loader = data_loader_3d(1024, 8)

    for x, y in data_loader:
        pred = MeanTensor(y)
        target = y
        metric.update(pred, target)

    images = metric.compute()
    assert "pred" in images
    assert "target" in images
    assert len(images["pred"]) == 8
    assert len(images["target"]) == 8


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="needs matplotlib")
def test_plot_samples_sequence():
    """
    Test the worst sample metric.
    """
    metric = PlotSamples()
    data_loader = data_loader_3d(1024, 8)

    for x, y in data_loader:
        pred = MeanTensor(y)
        target = y
        metric.update([pred] * 4, [target] * 4)

    images = metric.compute()
    assert len(images) == 8
    assert "pred" in images[0]
    assert "target" in images[0]
    assert len(images[0]["pred"]) == 4
    assert len(images[0]["target"]) == 4
