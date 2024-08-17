"""
pytorch_retrieve.tensors
========================

The 'pytorch_retrieve.tensors' module implements Tensor classes used by
'pytorch_retrieve'.
"""
import functools

import torch


"""
pytorch_retrieve
================

Provides specialized tensor classes used by ``pytorch_retrieve``.
"""
from .base import RegressionTensor
from .mean_tensor import MeanTensor
from .quantile_tensor import QuantileTensor
from .probability_tensor import ProbabilityTensor
from .masked_tensor import MaskedTensor
from .classification import ClassificationTensor, DetectionTensor
