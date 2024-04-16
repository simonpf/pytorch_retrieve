"""
Tests for the pytorch_retrieve.modules.normalization module.
"""
import torch
from torch import nn

from pytorch_retrieve.modules import normalization


def test_layer_norm_first():
    """
    Retrieve LayerNormFirst module by name and ensure that:
    - Tensors can be propagated through it
    - The dtype is conserved
    - Normalization works for 5D inputs.
    """
    normalization_factory = normalization.get_normalization_factory("LayerNormFirst")
    norm = normalization_factory(32)

    x = torch.rand(1, 32, 32, 32).to(dtype=torch.float32)
    y = norm(x)
    assert y.dtype == torch.float32

    norm = norm.to(dtype=torch.bfloat16)
    x = torch.rand(1, 32, 32, 32).to(dtype=torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16

    x = torch.rand(1, 32, 16, 32, 32).to(dtype=torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16


def test_rms_norm():
    """
    Retrieve RMSNorm module by name and ensure that:
    - Tensors can be propagated through it
    - The dtype is conserved
    - Normalization works for 5D inputs.
    """
    normalization_factory = normalization.get_normalization_factory("RMSNorm")
    norm = normalization_factory(32)

    x = torch.rand(1, 32, 32, 32).to(dtype=torch.float32)
    y = norm(x)
    assert y.dtype == torch.float32

    norm = norm.to(dtype=torch.bfloat16)
    x = torch.rand(1, 32, 32, 32).to(dtype=torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16

    x = torch.rand(1, 32, 16, 32, 32).to(dtype=torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16
