"""
Tests for the retrieval output classes.
"""
import numpy as np
import pytest
from scipy.stats import norm
import torch

from pytorch_retrieve.config import OutputConfig
from pytorch_retrieve.tensors import (
    QuantileTensor,
    ProbabilityTensor
)
from pytorch_retrieve.tensors import (
    MeanTensor,
    QuantileTensor,
    ProbabilityTensor,
)
from pytorch_retrieve.retrieval_output import (
    Full,
    ExpectedValue,
    ExceedanceProbability
)

def get_normal_mean_tensor() -> MeanTensor:
    """
    Create quantile tensor describing a Gaussian distribution centered on 0 with std. dev. 1.
    """
    return MeanTensor(torch.zeros(1))


def get_normal_quantile_tensor() -> QuantileTensor:
    """
    Create quantile tensor describing a Gaussian distribution centered on 0 with std. dev. 1.
    """
    quantiles = np.linspace(0, 1, 33)[1:-1]
    tensor = norm.ppf(quantiles)
    return QuantileTensor(
        torch.tensor(tensor)[None].to(torch.float32),
        tau=torch.tensor(quantiles).to(torch.float32)
    )

def get_normal_probability_tensor() -> ProbabilityTensor:
    """
    Create quantile tensor describing a Gaussian distribution centered on 0 with std. dev. 1.
    """
    bins = np.linspace(-6, 6, 101)
    pdf = np.log(norm.pdf(0.5 * (bins[1:] + bins[:-1])))
    return ProbabilityTensor(
        torch.tensor(pdf)[None].to(torch.float32),
        bins=torch.tensor(bins).to(torch.float32)
    )


TEST_TENSORS = [
    get_normal_mean_tensor(),
    get_normal_quantile_tensor(),
    get_normal_probability_tensor()
]

TEST_TENSORS_PROB = [
    get_normal_quantile_tensor(),
    get_normal_probability_tensor()
]


@pytest.mark.parametrize("pred", TEST_TENSORS)
def test_full(pred: torch.Tensor):
    """
    Ensure that calculation of full retrieval output leaves tensor unchanged.
    """
    output_config = OutputConfig("y", "Mean", (1,))
    output = Full(output_config)
    full = output.compute(pred)
    assert torch.isclose(full, pred).all()


@pytest.mark.parametrize("pred", TEST_TENSORS)
def test_expected_value(pred: torch.Tensor):
    """
    Ensure that calculation of expected yields 0.
    """
    output_config = OutputConfig("y", "Mean", (1,))
    output = ExpectedValue(output_config)
    expv = output.compute(pred)
    assert torch.isclose(expv, torch.tensor(0.0), atol=1e-6).all()


@pytest.mark.parametrize("pred", TEST_TENSORS_PROB)
def test_exceedance_probability(pred: torch.Tensor):
    """
    Ensure that calculation of exceedance probability for 0.0 yields 0.5.
    """
    output_config = OutputConfig("y", "Mean", (1,))
    output = ExceedanceProbability(output_config, 0.0)
    expv = output.compute(pred)
    assert torch.isclose(expv, torch.tensor(0.5)).all()
