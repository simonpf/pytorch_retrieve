from conftest import data_loader_1d, data_loader_3d

try:
    import matplotlib

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import pytest
import torch

from pytorch_retrieve.metrics import (
    Bias,
    CorrelationCoef,
    MAE,
    MSE,
    SMAPE,
    PlotSamples,
    HeidkeSkillScore,
    BrierScore
)
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

    bias = Bias(conditional={"rand": (0, 1, 4)})
    data_loader = data_loader_1d(1024, 128)

    # Test conditional bias with regularly-spaced bins.
    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        rand = torch.rand(y.shape)
        bias.update(y, y, conditional={"rand": rand})

    result = bias.compute()
    assert result.shape == (4,)
    assert torch.isclose(result, torch.tensor(0.0)).all()

    # Test conditional bias with fixed number of bins
    bias = Bias(conditional={"rand": 4})
    data_loader = data_loader_1d(1024, 128)
    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        rand = torch.rand(y.shape) * 4
        bias.update(y, y, conditional={"rand": rand})
    result = bias.compute()
    assert result.shape == (4,)
    assert torch.isclose(result, torch.tensor(0.0)).all()


def test_weighted_bias():
    """
    Test that weighting of samples is handled correctly.
    """
    bias = Bias()
    data_loader = data_loader_1d(1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y_pred = torch.clone(y)
        y_pred[mask] += 1.0
        weights = 1.0 - mask.to(torch.float32)
        assert torch.isclose((y_pred - y) * weights, torch.tensor(0.0)).all()
        bias.update(y, y_pred, weights=weights)

    result = bias.compute()
    assert torch.isclose(result, torch.tensor(0.0))

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y_pred = torch.clone(y)
        y_pred[mask] += 1.0
        bias.update(y, y_pred)

    result = bias.compute()
    assert not torch.isclose(result, torch.tensor(0.0))

    bias = Bias()


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

    # Test conditional correlation with regularly-spaced bins.
    corr = CorrelationCoef(conditional={"rand": (0, 1, 4)})
    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        rand = torch.rand(y.shape)
        corr.update(y, y, conditional={"rand": rand})

    result = corr.compute()
    assert result.shape == (4,)
    assert torch.isclose(result, torch.tensor(1.0)).all()

    # Test conditional correlation with broadcasted coordinates
    corr = CorrelationCoef(conditional={"rand": (0, 1, 4)})
    for x, y in data_loader:
        y = y[..., None].repeat_interleave(4, -2).repeat_interleave(4, -1)
        mask = torch.rand(y.shape) > 0.5
        #y[mask] = torch.nan
        #y = MaskedTensor(y, mask=mask)
        rand = torch.rand(y.shape[:1])
        corr.update(y, y, conditional={"rand": rand})

    result = corr.compute()
    assert result.shape == (4,)
    assert torch.isclose(result, torch.tensor(1.0)).all()


def test_correlation_coef_weighted():
    """
    Test calculation of the weighted correlation coefficient.
    """
    corr = CorrelationCoef()
    data_loader = data_loader_1d(1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y_pred = torch.clone(y)
        y_pred[mask] += 1.0
        corr.update(y_pred, y)

    result = corr.compute()
    assert not torch.isclose(result, torch.tensor(1.0))

    corr = CorrelationCoef()
    data_loader = data_loader_1d(1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y_pred = torch.clone(y)
        y_pred[mask] += 1.0
        weights = 1.0 - mask.to(torch.float32)
        corr.update(y_pred, y, weights=weights)

    result = corr.compute()
    assert torch.isclose(result, torch.tensor(1.0))


def test_mse():
    """
    Test that calculating the MSE works with masked tensors.
    """
    mse = MSE()
    data_loader = data_loader_1d(10 * 1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y.set_(torch.normal(torch.zeros_like(y), torch.ones_like(y)))
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        mse.update(y, torch.zeros_like(y))

    result = mse.compute()
    assert torch.isclose(result, torch.tensor(1.0), atol=0.2)

    # Test conditional MSE with regularly-spaced bins.
    mse = MSE(conditional={"rand": (0, 1, 4)})
    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y.set_(torch.normal(torch.zeros_like(y), torch.ones_like(y)))
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        rand = torch.rand(y.shape)
        mse.update(y, torch.zeros_like(y), conditional={"rand": rand})

    result = mse.compute()
    assert result.shape == (4,)
    assert torch.isclose(result, torch.tensor(1.0), atol=0.2).all()


def test_weighted_mse():
    """
    Test weighted MSE.
    """
    mse = MSE()
    data_loader = data_loader_1d(10 * 1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y_pred = torch.clone(y)
        y_pred[mask] += 1.0
        mse.update(y_pred, y)

    result = mse.compute()
    assert torch.isclose(result, torch.tensor(0.5), rtol=1e-1)

    mse = MSE()

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        weights = 1.0 - 0.5 * mask.to(torch.float32)
        y_pred = torch.clone(y)
        y_pred[mask] += 1.0
        mse.update(y_pred, y, weights=weights)

    result = mse.compute()
    assert torch.isclose(result, torch.tensor(1 / 3), rtol=1e-1)


def test_mae():
    """
    Test that calculating the MAE works with masked tensors.
    """
    mae = MAE()
    data_loader = data_loader_1d(10 * 1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y.set_(torch.normal(torch.zeros_like(y), torch.ones_like(y)))
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        mae.update(y, torch.zeros_like(y))

    result = mae.compute()
    assert torch.isclose(result, torch.tensor(0.798), atol=0.2)

    # Test conditional MAE with regularly-spaced bins.
    mae = MAE(conditional={"rand": (0, 1, 4)})
    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y.set_(torch.normal(torch.zeros_like(y), torch.ones_like(y)))
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        rand = torch.rand(y.shape)
        mae.update(y, torch.zeros_like(y), conditional={"rand": rand})

    result = mae.compute()
    assert result.shape == (4,)
    assert torch.isclose(result, torch.tensor(0.798), atol=0.2).all()


def test_weighted_mae():
    """
    Test weighted MAE.
    """
    mae = MAE()
    data_loader = data_loader_1d(10 * 1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y_pred = torch.clone(y)
        y_pred[mask] += 1.0
        mae.update(y_pred, y)

    result = mae.compute()
    assert torch.isclose(result, torch.tensor(0.5), rtol=1e-1)

    mae = MAE()

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        weights = 1.0 - 0.5 * mask.to(torch.float32)
        y_pred = torch.clone(y)
        y_pred[mask] += 1.0
        mae.update(y_pred, y, weights=weights)

    result = mae.compute()
    assert torch.isclose(result, torch.tensor(1 / 3), rtol=1e-1)


def test_smape():
    """
    Test that calculating the SMAPE works with masked tensors.
    """
    smape = SMAPE()
    data_loader = data_loader_1d(10 * 1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y.set_(torch.normal(torch.zeros_like(y), torch.ones_like(y)))
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        smape.update(torch.zeros_like(y), y)

    result_ref = smape.compute()
    assert torch.isfinite(result_ref)

    # Test conditional SMAPE with regularly-spaced bins.
    smape = SMAPE(conditional={"rand": (0, 1, 4)})
    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y.set_(torch.normal(torch.zeros_like(y), torch.ones_like(y)))
        y[mask] = torch.nan
        y = MaskedTensor(y, mask=mask)
        rand = torch.rand(y.shape)
        smape.update(torch.zeros_like(y), y, conditional={"rand": rand})

    result = smape.compute()
    assert result.shape == (4,)
    assert torch.isclose(result, result_ref, atol=0.2).all()


def test_weighted_smape():
    """
    Test weighted SMAPE.
    """
    smape = SMAPE()
    data_loader = data_loader_1d(1024, 128)

    for x, y in data_loader:
        mask = torch.rand(y.shape) > 0.5
        y_pred = torch.clone(y)
        y_pred[mask] += 1.0
        weights = 1.0 - mask.to(torch.float32)
        smape.update(y_pred, y, weights=weights)

    result = smape.compute()
    assert torch.isclose(result, torch.tensor(0.0))


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="needs matplotlib")
def test_plot_samples():
    """
    Test the plot sample metric.
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
    Test the plot sample metric.
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


def test_heidke_skill_score():
    """
    Test the Heidke Skill Score metric.
    """
    metric = HeidkeSkillScore()

    target = torch.rand(1_000) > 0.5
    pred = target
    metric.update(pred, target)
    val = metric.compute()
    assert torch.isclose(val, torch.tensor(1.0))

    metric = HeidkeSkillScore()
    target = torch.rand(100_000) > 0.5
    pred = torch.rand(100_000) > 0.5
    metric.update(pred, target)
    val = metric.compute()
    assert torch.isclose(val, torch.tensor(0.0), atol=0.01)


def test_brier_score():
    """
    Test the Brier Score metric.
    """
    metric = BrierScore()

    target = torch.rand(1_000) > 0.5
    pred = target.to(torch.float32)
    metric.update(pred, target)
    val = metric.compute()
    assert torch.isclose(val, torch.tensor(0.0))

    metric = BrierScore()
    target = torch.rand(100_000) > 0.5
    pred = 0.5 * torch.ones(100_000)
    metric.update(pred, target)
    val = metric.compute()
    assert torch.isclose(val, torch.tensor(0.25), atol=0.01)
