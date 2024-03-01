"""
Tests for the pytorch_retrieve.architectures.metnet architecture.
"""
import toml
import numpy as np
import torch

from pytorch_retrieve.architectures import compile_architecture
from pytorch_retrieve.architectures.metnet import MetNetConfig
from pytorch_retrieve.config import InputConfig, OutputConfig

METNET_CONFIG = """
[input.goes]
n_features = 16

[input.mrms]
n_features = 1

[output.precip_rate]
kind = "Mean"
shape = 1

[architecture]
name = "MetNet"
input_size = 512
time_step = 15
forecast_range = 240

[architecture.stem]
kind = "avgpool"

[architecture.temporal_encoder]
hidden_channels = 128
n_inputs = 4
n_layers = 2

[architecture.spatial_aggregator]
depth = 4
n_heads = 4
"""


def test_parse_metnet_config():
    """
    Ensure that parsing of MetNet config works.
    """
    config_dict = toml.loads(METNET_CONFIG)

    input_configs = {
        name: InputConfig.parse(name, cfg) for name, cfg in config_dict["input"].items()
    }
    output_configs = {
        name: OutputConfig.parse(name, cfg)
        for name, cfg in config_dict["output"].items()
    }
    arch_config = config_dict["architecture"]
    metnet_config = MetNetConfig.parse(input_configs, output_configs, arch_config)


def test_compile_metnet_config():
    """
    Ensure that compilation of metnet config works.
    """
    metnet = compile_architecture(toml.loads(METNET_CONFIG))

    x = {
        "goes": [torch.rand(1, 16, 512, 512) for _ in range(4)],
        "mrms": [torch.rand(1, 1, 512, 512) for _ in range(4)],
    }
    x["lead_times"] = torch.tensor([[15, 30]])
    y = metnet(x)
    assert len(y["precip_rate"]) == 2
