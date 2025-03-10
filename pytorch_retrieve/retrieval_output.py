"""
pytorch_retrieve.retrieval_output
=================================

This module defines retrieval output classes defining calculations to turn the raw model
output to retrieval quantities of interes.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch

from pytorch_retrieve.config import OutputConfig


class RetrievalOutput(ABC):
    """
    Abstract base class for retrieval outputs.
    """

    def __init__(self, dimensions: List[str], coordinates: Dict[str, np.ndarray]):
        """
        Args:
            model_output: The OutputConfig describing the original model output.
            dimensions: A list containing the names of the dimensions of the output.
            coordinates: A dict mapping dimension names to corresponding coordinates.
        """
        self.dimensions = dimensions
        self.coordinates = coordinates

    @abstractmethod
    def compute(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval output from model outputs.

        Args:
            preds: A torch.Tensor containing the raw model output.

        Return:
            A tensor containing the retrieval output computed from 'preds'.
        """


class Full(RetrievalOutput):
    """
    The full model output.
    """

    def __init__(
        self,
        model_output: OutputConfig,
    ):
        """
        Args:
            model_output: The output config describing the model output.
        """
        dimensions = model_output.get_output_dimensions()
        coordinates = model_output.get_output_coordinates()
        super().__init__(dimensions, coordinates)

    def compute(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval output from model outputs.

        Args:
            preds: A torch.Tensor containing the raw model output.

        Return:
            A tensor containing the retrieval output computed from 'preds'.
        """
        return preds


class ExpectedValue(RetrievalOutput):
    """
    The expected value of probabilistic model output.
    """

    def __init__(
        self,
        model_output: OutputConfig,
    ):
        """
        Args:
            model_output: The output config describing the model output.
        """
        dimensions = model_output.get_output_dimensions()
        coordinates = model_output.get_output_coordinates()
        extra_dims = model_output.extra_dimensions
        coodinates = {
            name: coords
            for name, coords in coordinates.items()
            if name not in extra_dims
        }
        super().__init__(dimensions[1:], coordinates)

    def compute(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval output from model outputs.

        Args:
            preds: A torch.Tensor containing the raw model output.

        Return:
            A tensor containing the retrieval output computed from 'preds'.
        """
        if isinstance(preds, list):
            return [self.compute(pred) for pred in preds]
        return preds.expected_value()


class ExceedanceProbability(RetrievalOutput):
    """
    Calculates the probability of the output exceeding a given threshold.
    """

    def __init__(self, model_output: OutputConfig, threshold: float):
        """
        Args:
            model_output: The output config describing the model output.
            threshold: The threshold for which to compute the exceedance
                probability.
        """
        dimensions = model_output.get_output_dimensions()
        coordinates = model_output.get_output_coordinates()
        extra_dims = model_output.extra_dimensions
        coodinates = {
            name: coords
            for name, coords in coordinates.items()
            if name not in extra_dims
        }
        super().__init__(dimensions[1:], coordinates)
        self.threshold = threshold

    def compute(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval output from model outputs.

        Args:
            preds: A torch.Tensor containing the raw model output.

        Return:
            A tensor containing the retrieval output computed from 'preds'.
        """
        if isinstance(preds, list):
            return [self.compute(pred) for pred in preds]
        return preds.probability_greater_than(self.threshold)


class Quantiles(RetrievalOutput):
    """
    Output class representing quantiles of the posterior distribution
    as retrieval output.
    """

    def __init__(
        self,
        model_output: OutputConfig,
        tau: List[float],
        dim_name: Optional[str] = None,
    ):
        """
        Args:
            model_output: The output config describing the model output.
            tau: A list containing the quantile fractions defining the quantiles
                to compute.
            dim_name: Optional name to use for the new quantile dimension.
        """
        dimensions = model_output.get_output_dimensions()
        coordinates = model_output.get_output_coordinates()
        extra_dims = model_output.extra_dimensions
        coodinates = {
            name: coords
            for name, coords in coordinates.items()
            if name not in extra_dims
        }
        if dim_name is None:
            dim_name = f"tau_{model_output.target}"
        coordinates[dim_name] = tau
        super().__init__([dim_name] + dimensions[1:], coordinates)
        self.tau = tau

    def compute(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval output from model outputs.

        Args:
            preds: A torch.Tensor containing the raw model output.

        Return:
            A tensor containing the retrieval output computed from 'preds'.
        """
        if isinstance(preds, list):
            return [self.compute(pred) for pred in preds]
        return preds.quantiles(tau=self.tau)


class ClassProbability(RetrievalOutput):
    """
    Output class representing class probabilities of classfication output.
    """
    def __init__(
        self,
        model_output: OutputConfig,
        dim_name: Optional[str] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Args:
            model_output: The output config describing the model output.
            tau: A list containing the quantile fractions defining the quantiles
                to compute.
            dim_name: Optional name to use for the new quantile dimension.
        """
        dimensions = model_output.get_output_dimensions()
        coordinates = model_output.get_output_coordinates()
        extra_dims = model_output.extra_dimensions
        coodinates = {
            name: coords
            for name, coords in coordinates.items()
            if name not in extra_dims
        }
        if dim_name is None:
            dim_name = f"{model_output.target}_probability"

        if class_names is None:
            class_names = [f"class_{i}" for i in range(model_output.n_classes)]
        coordinates[dim_name] = class_names
        self.n_classes = model_output.n_classes
        super().__init__([dim_name] + dimensions[1:], coordinates)


    def compute(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval output from model outputs.

        Args:
            preds: A torch.Tensor containing the raw model output.

        Return:
            A tensor containing the retrieval output computed from 'preds'.
        """
        if isinstance(preds, list):
            return [self.compute(pred) for pred in preds]

        return preds.probability()


class RandomSample(RetrievalOutput):
    """
    Random sample from probabilistic regression results.
    """
    def __init__(
        self,
        model_output: OutputConfig,
        n_samples: int,
        dim_name: Optional[str] = None,
    ):
        """
        Args:
            model_output: The output config describing the model output.
        """
        dimensions = model_output.get_output_dimensions()
        coordinates = model_output.get_output_coordinates()
        extra_dims = model_output.extra_dimensions
        coodinates = {
            name: coords
            for name, coords in coordinates.items()
            if name not in extra_dims
        }
        self.n_samples = n_samples
        if n_samples > 1:
            if dim_name is None:
                dim_name = f"{model_output.target}_samples"
            super().__init__([dim_name] + dimensions[1:], coordinates)
        else:
            super().__init__(dimensions[1:], coordinates)


    def compute(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval output from model outputs.

        Args:
            preds: A torch.Tensor containing the raw model output.

        Return:
            A tensor containing the retrieval output computed from 'preds'.
        """
        if isinstance(preds, list):
            return [self.compute(pred) for pred in preds]
        return preds.random_sample(n_samples=self.n_samples)


class MaximumProbability(RetrievalOutput):
    """
    Maximum probability estimator from probabilistic regression results.
    """
    def __init__(
        self,
        model_output: OutputConfig,
        dim_name: Optional[str] = None,
    ):
        """
        Args:
            model_output: The output config describing the model output.
        """
        dimensions = model_output.get_output_dimensions()
        coordinates = model_output.get_output_coordinates()
        extra_dims = model_output.extra_dimensions
        coodinates = {
            name: coords
            for name, coords in coordinates.items()
            if name not in extra_dims
        }
        super().__init__(dimensions[1:], coordinates)


    def compute(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Compute retrieval output from model outputs.

        Args:
            preds: A torch.Tensor containing the raw model output.

        Return:
            A tensor containing the retrieval output computed from 'preds'.
        """
        if isinstance(preds, list):
            return [self.compute(pred) for pred in preds]
        return preds.maximum_probability()
