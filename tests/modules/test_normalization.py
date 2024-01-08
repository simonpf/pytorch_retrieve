"""
Tests for the pytorch_retrieve.modules.normalization module.
"""
import torch
from torch import nn

from pytorch_retrieve.modules import normalization


def test_layer_norm_first():
    """
    Retrieve LayerNormFirst module by name and ensure that tensor can be
    forwarded through it.
    """
    normalization_factory = normalization.get_normalization_factory("LayerNormFirst")
    norm = normalization_factory(32)
    x = torch.rand(1, 32, 32, 32)
    y = norm(x)
