"""
Tests for the pytorch_retrieve.architectures.autoregressive module.
"""
import toml
import torch

from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.config import InputConfig, OutputConfig
from pytorch_retrieve.architectures.autoregressive import (
    EncoderConfig,
    DecoderConfig,
    TemporalEncoderConfig,
    PropagatorConfig,
    AutoregressiveConfig,
    Autoregressive
)

PROPAGATOR_CONFIG = (
    """
    [encoder]
    channels = [32, 32, 64]
    stage_depths = [2, 2, 2]

    [decoder]
    channels = [32, 16]
    stage_depths= [2, 2]
    """
)

def test_propagator_config():
    """
    Parse propagator config, compile and forward tensor through propagator.

    """
    propagator_config = toml.loads(PROPAGATOR_CONFIG)
    propagator_config = PropagatorConfig.parse(2, 16, propagator_config)
    propagator = propagator_config.compile()

    x = torch.rand(1, 32, 64, 64)
    y = propagator(x)
    assert y.shape == (1, 16, 64, 64)


TEMPORAL_ENCODER_CONFIG = (
    """
    latent_dim = 16
    kind = "recurrent"
    [encoder]
    channels = [16, 32, 64]
    stage_depths = [2, 2, 2]

    [decoder]
    channels = [32, 16]
    stage_depths= [2, 2]
    """
)

def test_temporal_encoder_config():
    """
    Parse temporal encoder config, compile and list of tensors through encoder.
    """
    temporal_encoder_config = toml.loads(TEMPORAL_ENCODER_CONFIG)
    temporal_encoder_config = TemporalEncoderConfig.parse(16, 2, temporal_encoder_config)
    temporal_encoder = temporal_encoder_config.compile()

    x = [torch.rand(1, 16, 64, 64) for x in range(4)]
    y = temporal_encoder(x)
    assert len(y) == 4
    assert y[0].shape == (1, 16, 64, 64)

DIRECT_TEMPORAL_ENCODER_CONFIG = (
    """
    latent_dim = 16
    n_inputs = 4
    kind = "direct"

    [encoder]
    channels = [64, 32, 16]
    stage_depths = [2, 2, 2]

    [decoder]
    channels = [32, 16]
    stage_depths= [2, 2]
    """
)

def test_temporal_encoder_config():
    """
    Parse temporal encoder config, compile and list of tensors through encoder.
    """
    temporal_encoder_config = toml.loads(DIRECT_TEMPORAL_ENCODER_CONFIG)
    temporal_encoder_config = TemporalEncoderConfig.parse(16, 2, temporal_encoder_config)
    temporal_encoder = temporal_encoder_config.compile()

    x = [torch.rand(1, 16, 64, 64) for x in range(4)]
    y = temporal_encoder(x)
    assert len(y) == 2
    assert y[0].shape == (1, 16, 64, 64)

INPUT_CONFIG = (
    """
    [x]
    n_features = 12
    """
)

ENCODER_CONFIG = (
    """
    latent_dim = 16
    [encoder]
    channels = [8, 32, 64]
    stage_depths = [2, 2, 2]

    [decoder]
    channels = [32, 16]
    stage_depths= [2, 2]

    [stem]
    kind = "BasicConv"
    depth = 1
    out_channels = 8
    """
)

def test_encoder_config():
    """
    Parse temporal encoder config, compile and list of tensors through encoder.
    """
    encoder_config = toml.loads(ENCODER_CONFIG)
    input_configs = toml.loads(INPUT_CONFIG)
    input_configs = {
        name: InputConfig.parse(name, config)
        for name, config in input_configs.items()
    }

    encoder_config = EncoderConfig.parse(16, input_configs, encoder_config)
    encoder = encoder_config.compile()

    x = [torch.rand(1, 12, 64, 64) for x in range(4)]
    y = encoder(x)
    assert len(y) == 4
    assert y[0].shape == (1, 16, 64, 64)


DECODER_CONFIG = (
    """
    channels = [32, 64]
    stage_depths = [1, 1]
    upsampling_factors = [2, 2]
    """
)

OUTPUT_CONFIG = (
    """
    [surface_precip]
    shape = 1
    kind = "Mean"
    """
)

def test_decoder_config():
    """
    Parse decoder config, compile and list of tensors through encoder.
    """
    decoder_config = toml.loads(DECODER_CONFIG)
    output_configs = toml.loads(OUTPUT_CONFIG)
    output_configs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_configs.items()
    }
    decoder_config = DecoderConfig.parse(16, decoder_config, output_configs)
    decoder = decoder_config.compile()

    x = torch.rand(1, 16, 32, 32)
    y = decoder(x)
    assert "surface_precip" in y
    assert y["surface_precip"].shape == (1, 1, 128, 128)


AUTOREGRESSIVE_CONFIG = (
    """
[architecture]
name = "Autoregressive"
order = 2
time_step = 30
latent_dim = 128

[architecture.encoder]

[architecture.encoder.encoder]
channels = [64, 128, 128]
stage_depths = [1, 1, 1]
block_factory = "ResNeXt"
block_factory_args = {padding_factory="Global", normalization_factory="LayerNormFirst", activation_factory="ReLU"}
downsampling_factors = [2, 2]

[architecture.encoder.decoder]
channels = []
stage_depths = []
block_factory = "ResNeXt"
block_factory_args = {padding_factory="Global", normalization_factory="LayerNormFirst", activation_factory="ReLU"}

[architecture.temporal_encoder]
kind = "Direct"
n_inputs = 2

[architecture.temporal_encoder.encoder]
stage_depths = [1]
channels = [256, 128, 128]
recurrence_factory = "Assimilator"
block_factory = "ResNeXt"
block_factory_args = {padding_factory="Global", normalization_factory="LayerNormFirst", activation_factory="ReLU"}

[architecture.temporal_encoder.decoder]
stage_depths = []
channels = []

[architecture.propagator]
[architecture.propagator.encoder]
channels = [128, 128, 128]
stage_depths = [1, 1, 1]
downsampling_factors = [[2, 2], [1, 2]]
block_factory = "ResNeXt"
block_factory_args = {padding_factory="Global", normalization_factory="LayerNormFirst", activation_factory="ReLU"}

[architecture.propagator.decoder]
channels = [128, 128]
stage_depths = [1, 1]
block_factory = "ResNeXt"
upsampling_factors = [[1, 2], [2, 2]]
block_factory_args = {padding_factory="Global", normalization_factory="LayerNormFirst", activation_factory="ReLU"}

[architecture.decoder]
channels = [128, 64]
stage_depths = [1, 1]
block_factory = "ResNeXt"
upsampling_factors = [2, 2]
block_factory_args = {padding_factory="Global", normalization_factory="LayerNormFirst", activation_factory="ReLU"}

[architecture.encoder.stem]
depth = 1
out_channels = 64

[architecture.decoder.head]
individual = false
depth = 3

[input.seviri]
n_features = 12

[output.surface_precip]
kind = "Mean"

    """
)


def test_autoregressive():
    """
    Ensure that an AutoregressiveConfig can be parsed and inputs propagated through
    it.
    """

    config = toml.loads(AUTOREGRESSIVE_CONFIG)

    output_configs = config["output"]
    output_configs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_configs.items()
    }
    input_configs = config["input"]
    input_configs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_configs.items()
    }
    arch_config = config["architecture"]

    autoregressive_config = AutoregressiveConfig.parse(
        input_configs, output_configs, arch_config
    )
    autoregressive = Autoregressive(
        input_configs,
        output_configs,
        autoregressive_config
    )

    x = {"seviri": [torch.rand(2, 12, 128, 128) for _ in range(2)]}
    x["lead_time"] = torch.tensor([[30, 60]])
    y = autoregressive(x)

    assert "surface_precip" in y
    assert len(y["surface_precip"]) == 2
    assert y["surface_precip"][0].shape == (2, 1, 128, 128)


def test_autoregressive_save_and_load(tmp_path):
    """
    Ensure that Autoregressive models can be saved and loaded.
    """
    config = toml.loads(AUTOREGRESSIVE_CONFIG)

    output_configs = config["output"]
    output_configs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_configs.items()
    }
    input_configs = config["input"]
    input_configs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_configs.items()
    }
    arch_config = config["architecture"]

    autoregressive_config = AutoregressiveConfig.parse(
        input_configs, output_configs, arch_config
    )
    autoregressive = Autoregressive(
        input_configs,
        output_configs,
        autoregressive_config
    )

    autoregressive.save(tmp_path / "model.pt")
    loaded = load_model(tmp_path / "model.pt")

    assert loaded.n_params == autoregressive.n_params
