import pickle

import pytest
import toml

import torch

from pytorch_retrieve.architectures.multi_scale_autoregressor import (
    MultiScaleAutoregressorConfig,
    MultiScaleAutoregressor
)
from pytorch_retrieve.config import InputConfig, OutputConfig

MODEL_CONFIG = """
[architecture]
name = "MultiScaleAutoregressor"
time_step = 30

[architecture.stem]
kind = "BasicConv3d"

[architecture.encoder]
block_factory = ["BasicConv3d", "BasicConv3d", "BasicConv3d", "BasicConv3d"]
channels = [16, 32, 64, 128]
stage_depths = [2, 2, 2, 2]
downsampling_factors = [[2, 2, 2], [2, 2, 2], [1, 2, 2]]

[architecture.decoder]
block_factory = ["BasicConv3d", "BasicConv3d", "BasicConv3d", "BasicConv3d"]
channels = [64, 32, 16]
stage_depths = [2, 2, 2]
upsampling_factors = [[1, 2, 2], [2, 2, 2], [2, 2, 2]]
upsampling_factory = "Trilinear"

[architecture.propagator]
block_factory = ["BasicConv", "BasicConv", "BasicConv", "BasicConv"]
block_factory_args = [{activation_factory='ReLU'}, {activation_factory='ReLU'}, {activation_factory='ReLU'}, {activation_factory='ReLU'}]
channels = [128, 64, 32, 16]
stage_depths = [2, 2, 2, 2]

[input.x]
n_features = 16

[output.precip]
kind = "Mean"
shape = 1
"""


def test_multi_scale_autoregressor_config(tmp_path):
    """
    Test parsing of the MSA config and conversion to configuration dictionary.
    """
    model_cfg = toml.loads(MODEL_CONFIG)
    arch_config = model_cfg["architecture"]
    input_configs = model_cfg["input"]
    output_configs = model_cfg["output"]
    input_configs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_configs.items()
    }
    output_configs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_configs.items()
    }
    msa_cfg = MultiScaleAutoregressorConfig.parse(
        input_configs,
        output_configs,
        arch_config
    )

    cfg_dct = msa_cfg.to_config_dict()
    assert "time_step" in cfg_dct
    assert "retrieval" in cfg_dct

    dmpd = pickle.dump(
        cfg_dct,
        open((tmp_path / "config_dict.pckl"), "wb")
    )


def test_multi_scale_autoregressor():
    """
    Test performing forecasts with the multi-scale autoregressor architecture.
    """
    model_cfg = toml.loads(MODEL_CONFIG)
    arch_config = model_cfg["architecture"]
    input_configs = model_cfg["input"]
    output_configs = model_cfg["output"]
    input_configs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_configs.items()
    }
    output_configs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_configs.items()
    }

    msa = MultiScaleAutoregressor.from_config_dict(model_cfg)
    inpts = {
        "x": [torch.rand(1, 16, 64, 64) for _ in range(8)],
        "lead_time": (30 * (torch.arange(8) + 1))[None]
    }

    preds = msa(inpts)
    assert "precip" in preds
    assert len(preds["precip"]) == 16

    err = sum(preds["precip"]).mean()
    err.backward()

    for param in msa.parameters():
        assert param.grad is not None


MODEL_CONFIG_ONLY_FORECAST = """
[architecture]
name = "MultiScaleAutoregressor"
time_step = 30
retrieval = false

[architecture.stem]
kind = "BasicConv3d"

[architecture.encoder]
block_factory = "BasicConv3d"
channels = [16, 32, 64, 128]
stage_depths = [2, 2, 2, 2]
downsampling_factors = [[2, 2, 2], [2, 2, 2], [1, 2, 2]]

[architecture.decoder]
block_factory = "BasicConv3d"
channels = [64, 32, 16]
stage_depths = [2, 2, 2]
upsampling_factors = [[1, 2, 2], [2, 2, 2], [2, 2, 2]]
upsampling_factory = "Trilinear"

[architecture.propagator]
block_factory = "BasicConv"
channels = [128, 64, 32, 16]
stage_depths = [2, 2, 2, 2]

[input.x]
n_features = 16

[output.precip]
kind = "Mean"
shape = 1
"""

def test_multi_scale_autoregressor_only_forecast():
    """
    Test performing forecasts with the multi-scale autoregressor architecture.
    """
    model_cfg = toml.loads(MODEL_CONFIG_ONLY_FORECAST)
    arch_config = model_cfg["architecture"]
    input_configs = model_cfg["input"]
    output_configs = model_cfg["output"]
    input_configs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_configs.items()
    }
    output_configs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_configs.items()
    }

    msa = MultiScaleAutoregressor.from_config_dict(model_cfg)
    inpts = {
        "x": [torch.rand(1, 16, 64, 64) for _ in range(8)],
        "lead_time": (30 * (torch.arange(8) + 1))[None]
    }

    preds = msa(inpts)
    assert "precip" in preds
    assert len(preds["precip"]) == 8

    err = sum(preds["precip"]).mean()
    err.backward()

    for param in msa.parameters():
        assert param.grad is not None
