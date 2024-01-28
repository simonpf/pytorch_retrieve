"""
Tests for the RecurrentEncoderDecoder architecture defined in
 pytorch_retrieve.architectures.recurrent_encoder_decoder
"""
import pytest
import toml
import torch

from pytorch_retrieve.config import InputConfig, OutputConfig
from pytorch_retrieve.architectures.recurrent_encoder_decoder import (
    EncoderDecoderConfig,
    RecurrentEncoderDecoder
)


MULTI_INPUT_CFG = """
[architecture]
name = "RecurrentEncoderDecoder"

[architecture.stem]
depth = 2
out_channels = 64

[architecture.encoder]
channels = [64, 128, 128, 256]
stage_depths = [2, 2, 2, 2]

[architecture.decoder]
channels = [128, 128, 64]
stage_depths = [2, 2, 2]

[input.x_1]
n_features = 16

[input.x_2]
n_features = 32

[output.precip]
kind = "Mean"
shape = 1
"""


@pytest.fixture
def multi_input_cfg():
    return toml.loads(MULTI_INPUT_CFG)


def test_encoder_config_multi_input(multi_input_cfg):
    arch_config = multi_input_cfg["architecture"]
    input_config = multi_input_cfg["input"]
    output_config = multi_input_cfg["output"]

    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }

    config = EncoderDecoderConfig.parse(input_cfgs, output_cfgs, arch_config)

    assert config.encoder_config.channels == [64, 128, 128, 256]
    assert config.decoder_config.channels == [256, 128, 128, 64]
    assert isinstance(config.stem_configs, dict)
    assert isinstance(config.head_configs, dict)

    encdec = RecurrentEncoderDecoder.from_config_dict(multi_input_cfg)
    assert encdec.stems["x_1"][-1].n_params > 0

    x = {
        "x_1": [torch.rand(1, 16, 128, 128) for _ in range(4)],
        "x_2": [torch.rand(1, 32, 128, 128) for _ in range(4)],
    }

    y = encdec(x)

    assert len(y["precip"]) == 4
    assert y["precip"][0].shape == (1, 1, 128, 128)
    assert isinstance(y["precip"][0], torch.Tensor)
