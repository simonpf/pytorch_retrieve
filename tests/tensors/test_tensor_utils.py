"""
Tests for the pytorch_retrieve.tensors.utils module.
"""
import torch

from pytorch_retrieve.tensors.utils import (
    crps_cdf,
    interpolate
)

def test_crps_cdf():
    """
    Test calculation of CRPS for step-function CDF the resulting value
    should be half of the CDF grid spacing.
    """
    x_cdf = torch.linspace(0, 2, 21)[None, :, None].repeat(10, 1, 10)
    y_cdf = (x_cdf > 1.0).to(dtype=torch.float32)
    crps = crps_cdf(x_cdf, y_cdf, 1.0, 1)
    assert torch.isclose(crps, torch.tensor(0.05)).all()

    # Test with y_true as tensor.
    y_true = torch.ones(10, 10)
    crps = crps_cdf(x_cdf, y_cdf, y_true, 1)
    assert torch.isclose(crps, torch.tensor(0.05)).all()


def test_interpolate():
    """
    Interpolate identity function and assert that:
        - Values are correctly interpolated within range.
        - Out-of-range interpolation is closest y value.
    """
    x_f = torch.linspace(1, 9, 9)[None, :, None].repeat(10, 1, 10)
    y_f = x_f
    x_new = 10.0 * torch.rand(10, 10)

    y_new = interpolate(x_new, x_f, y_f, 1)

    assert (y_new >= 1.0).all()
    assert (y_new <= 9.0).all()

    within = (x_new >= 1.0) * (x_new <= 9.0)
    assert torch.isclose(y_new[within], x_new[within]).all()
