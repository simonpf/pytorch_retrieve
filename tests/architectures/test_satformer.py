#!/usr/bin/env python3
import pytest
import toml
import torch

import numpy as np
import xarray as xr

from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.config import InputConfig, OutputConfig
from pytorch_retrieve.tensors import MeanTensor
from pytorch_retrieve.architectures.satformer import Satformer


SATFORMER_CFG = """
[architecture]
name = "Satformer"
output_embed_dim = 16
n_heads_perceiver = 2

[architecture.encoder]
channels = [16, 32, 64, 128, 256]
stage_depths = [1, 2, 2, 3, 5]
block_factory_args = [
    {expansion_factor=1, excitation_ratio=0.0, fused=true, attention=false, n_heads=2},
    {expansion_factor=4, excitation_ratio=0.0, fused=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.0, fused=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true, attention=true},
]


[architecture.decoder]
channels = [128, 64, 32, 16]
stage_depths = [3, 2, 2, 1]
block_factory_args = [
    {expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.0, fused=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.0, fused=true, attention=true, n_heads=2},
    {expansion_factor=1, excitation_ratio=0.0, fused=true, attention=false, n_heads=2},
]

[architecture.encoding.observations]
channels_in = 32
channels_out = 16
hidden_channels = 32
depth = 3
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.observations]
n_features = 1
meta_data = "input_observation_props"
encoding = "observations"
mask = "input_observation_mask"

[input.t2m]
n_features = 1

[output.output_observations]
kind = "Mean"
shape = 1
conditional = "output_observation_props"
encoding = "observations"
mask = "output_observation_mask"

[output.surface_precip]
kind = "Mean"
shape = 1
"""


@pytest.fixture
def satformer_cfg():
    """
    Fixture providing a config dict for a Satformer architecture.

    """
    return toml.loads(SATFORMER_CFG)


def test_satformer(satformer_cfg, tmp_path):

    sfr = Satformer.from_config_dict(satformer_cfg)
    x = {
        "observations": torch.rand(1, 1, 12, 128, 128),
        "output_observations": torch.rand(1, 1, 4, 128, 128),
        "input_observation_props": torch.rand(1, 32, 12, 128, 128),
        "output_observation_props": torch.rand(1, 32, 4, 128, 128),
        "input_observation_mask": torch.rand(1, 12) > 0.5,
        "output_observation_mask": torch.rand(1, 4) > 0.5,
    }
    y = sfr(x)

    assert len(y["output_observations"]) == 4
    assert len(y["surface_precip"]) == 1


SATFORMER_UNCOND_CFG = """
[architecture]
name = "Satformer"
output_embed_dim = 16
n_heads_perceiver = 2

[architecture.encoder]
channels = [16, 32, 64, 128, 256]
stage_depths = [1, 2, 2, 3, 5]
block_factory_args = [
    {expansion_factor=1, excitation_ratio=0.0, fused=true, attention=false},
    {expansion_factor=4, excitation_ratio=0.0, fused=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.0, fused=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true, attention=true},
]

[architecture.decoder]
channels = [128, 64, 32, 16]
stage_depths = [3, 2, 2, 1]
block_factory_args = [
    {expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.25, anti_aliasing=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.0, fused=true, attention=true},
    {expansion_factor=4, excitation_ratio=0.0, fused=true, attention=true, n_heads=2},
    {expansion_factor=1, excitation_ratio=0.0, fused=true, attention=false, n_heads=2},
]

[architecture.encoding.observations]
channels_in = 32
channels_out = 16
hidden_channels = 32
depth = 3
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.observations]
n_features = 1
meta_data = "input_observation_props"
encoding = "observations"
mask = "input_observation_mask"

[input.t2m]
n_features = 1

[output.surface_precip]
kind = "Mean"
shape = 1
"""


@pytest.fixture
def satformer_uncond_cfg():
    """
    Fixture providing a config dict for a Satformer architecture with only
    unconditional outputs.
    """
    return toml.loads(SATFORMER_UNCOND_CFG)


def test_satformer_unconditional_outputs(satformer_uncond_cfg, tmp_path):

    sfr = Satformer.from_config_dict(satformer_uncond_cfg)
    x = {
        "observations": torch.rand(1, 1, 12, 128, 128),
        "output_observations": torch.rand(1, 1, 4, 128, 128),
        "input_observation_props": torch.rand(1, 32, 12, 128, 128),
        "output_observation_props": torch.rand(1, 32, 4, 128, 128),
        "input_observation_mask": torch.rand(1, 12) > 0.5,
    }
    y = sfr(x)
    assert len(y["surface_precip"]) == 1
