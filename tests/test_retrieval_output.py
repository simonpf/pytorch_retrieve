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
    ProbabilityTensor,
    DetectionTensor,
    ClassificationTensor
)
from pytorch_retrieve.tensors import (
    MeanTensor,
    QuantileTensor,
    ProbabilityTensor,
)
from pytorch_retrieve.retrieval_output import (
    Full,
    ExpectedValue,
    ExceedanceProbability,
    ClassProbability,
    RandomSample,
    MaximumProbability
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


def get_detection_tensor() -> ProbabilityTensor:
    """
    Create detection tensor containing logits for binary classification output.
    """
    p = torch.zeros((32, 1))
    return DetectionTensor(p)


def get_classficiation_tensor() -> ProbabilityTensor:
    """
    Create detection tensor containing logits for binary classification output.
    """
    p = torch.zeros((32, 4))
    return ClassificationTensor(p)


TEST_TENSORS = [
    get_normal_mean_tensor(),
    get_normal_quantile_tensor(),
    get_normal_probability_tensor(),
]

TEST_TENSORS_PROB = [
    get_normal_quantile_tensor(),
    get_normal_probability_tensor()
]

TEST_TENSORS_CLASS = [
    get_detection_tensor(),
    get_classficiation_tensor()
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


@pytest.mark.parametrize("pred", TEST_TENSORS_CLASS)
def test_class_probability(pred: torch.Tensor):
    """
    Ensure that calculation of class probabilities yields uniform values
    across classes.
    """
    n_classes = pred.shape[1]
    if n_classes == 1:
        n_classes = 2

    output_config = OutputConfig("y", "Classification", shape=1, n_classes=n_classes)
    output = ClassProbability(output_config)
    probs = output.compute(pred)
    assert torch.isclose(probs, torch.tensor(1.0 / n_classes)).all()


def test_random_sample():
    """
    Ensure that random samples have a mean of 0.0.
    """
    quantiles = np.linspace(0, 1, 1026)[1:-1]
    tensor = norm.ppf(quantiles)
    tensor = QuantileTensor(
        torch.tensor(tensor)[None].to(torch.float32),
        tau=torch.tensor(quantiles).to(torch.float32)
    )
    output_config = OutputConfig("y", "RandomSample", shape=(1,))
    output = RandomSample(output_config, n_samples=2500)
    rand = output.compute(tensor)
    assert torch.isclose(rand.mean(), torch.tensor(0.0), atol=0.08)


def test_maximum_probability():
    """
    Ensure that maximum probability of a Gaussian distribution.
    """
    quantiles = np.linspace(0, 1, 1026)[1:-1]
    tensor = norm.ppf(quantiles)
    tensor = QuantileTensor(
        torch.tensor(tensor)[None].to(torch.float32),
        tau=torch.tensor(quantiles).to(torch.float32)
    )
    output_config = OutputConfig("y", "RandomSample", shape=(1,))
    output = MaximumProbability(output_config)
    rand = output.compute(tensor)
    assert torch.isclose(rand.mean(), torch.tensor(0.0), atol=0.08)
