import pytest
import toml
import torch

from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.config import InputConfig, OutputConfig
from pytorch_retrieve.tensors import MeanTensor
from pytorch_retrieve.architectures.encoder_decoder import (
    EncoderDecoderConfig,
    EncoderDecoder,
    EncoderConfig,
    DecoderConfig,
    StemConfig,
    HeadConfig,
)

from conftest import data_loader_3d


def test_stem_config():
    """
    Test that up- and down-sampling in stem are handled correctly.
    """
    stem_config = StemConfig(
        "x", 4, (2, 2), "BasicConv",
        out_channels=12,
        depth=2,
        downsampling=(4, 4),
        upsampling=(2, 2),
        normalize="none"
    )
    assert stem_config.out_scale == (1, 4, 4)

    stem = stem_config.compile()

    x = torch.rand(1, 4, 32, 32)
    y = stem(x)
    assert y.shape == (1, 12, 16, 16)



def test_encoder_decoder_multiple_block_factories():
    """
    Ensure that passing multiple block factories to encoder results in an encoder with
    different block in different stages.
    """
    stem_cfgs = {
        "x": StemConfig("x", 12, 4, "BasicConv", 16)
    }
    encoder_cfg = {
        "channels": [16, 32, 64],
        "stage_depths": [1, 2, 3],
        "block_factory": ["BasicConv", "ResNet", "ResNet"],
    }
    encoder_cfg = EncoderConfig.parse(stem_cfgs, encoder_cfg)
    encoder = encoder_cfg.compile()
    assert type(encoder.stages[0]) != type(encoder.stages[1][0])


    decoder_cfg = {
        "channels": [64, 32],
        "stage_depths": [2, 1],
        "block_factory": ["ResNet", "BasicConv"],
    }
    decoder_cfg = DecoderConfig.parse(encoder_cfg, decoder_cfg)
    decoder = decoder_cfg.compile()
    assert type(decoder.stages[0][0])  != type(decoder.stages[1])


def test_decoder_multiple_block_factories():
    """
    Ensure that passing multiple block factories to encoder results in an encoder with
    different block in different stages.
    """
    stem_cfgs = {
        "x": StemConfig("x", 12, 4, "BasicConv", 16)
    }
    encoder_cfg = {
        "channels": [16, 32, 64],
        "stage_depths": [1, 2, 3],
        "block_factory": ["BasicConv", "ResNet", "ResNet"],
    }
    encoder_cfg = EncoderConfig.parse(stem_cfgs, encoder_cfg)
    encoder = encoder_cfg.compile()
    assert type(encoder.stages[0]) != type(encoder.stages[1][0])





SINGLE_INPUT_NO_STEM_CFG = """
[architecture]
name = "EncoderDecoder"

[architecture.encoder]
channels = [16, 32, 64, 128]
stage_depths = [2, 2, 2, 2]

[architecture.decoder]
channels = [64, 32, 16]
stage_depths = [2, 2, 2]

[input.x]
n_features = 16

[output.precip]
kind = "Mean"
shape = 1
"""


@pytest.fixture
def single_input_no_stem_cfg():
    return toml.loads(SINGLE_INPUT_NO_STEM_CFG)


def test_encoder_config_single_input_no_stem(single_input_no_stem_cfg):
    arch_config = single_input_no_stem_cfg["architecture"]
    input_config = single_input_no_stem_cfg["input"]
    output_config = single_input_no_stem_cfg["output"]

    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }

    config = EncoderDecoderConfig.parse(input_cfgs, output_cfgs, arch_config)

    assert config.encoder_config.channels == [16, 32, 64, 128]
    assert config.decoder_config.channels == [128, 64, 32, 16]
    assert isinstance(config.stem_configs, dict)
    assert isinstance(config.head_configs, dict)

    encdec = EncoderDecoder.from_config_dict(single_input_no_stem_cfg)

    x = torch.rand(1, 16, 128, 128)
    y = encdec(x)

    assert y.shape == (1, 1, 128, 128)
    assert isinstance(y, torch.Tensor)


SINGLE_INPUT_CFG = """
[architecture]
name = "EncoderDecoder"

[architecture.stem]
depth = 2


[architecture.encoder]
channels = [16, 32, 64, 128]
stage_depths = [2, 2, 2, 2]

[architecture.decoder]
channels = [64, 32, 16]
stage_depths = [2, 2, 2]

[input.x]
n_features = 16

[output.precip]
kind = "Mean"
shape = 1
"""


@pytest.fixture
def single_input_cfg():
    return toml.loads(SINGLE_INPUT_CFG)


def test_stem_config_to_dict(single_input_cfg):
    """
    Test converting stem config to dict and parsing it back.
    """
    arch_config = single_input_cfg["architecture"]
    input_config = single_input_cfg["input"]
    output_config = single_input_cfg["output"]
    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }

    stem_config = StemConfig.parse(
        "x", "x", input_cfgs["x"], arch_config.get("stem", {})
    )
    stem_config_dict = stem_config.to_config_dict()
    stem_config_p = StemConfig.parse("x", "x", input_cfgs["x"], stem_config_dict)

    assert stem_config == stem_config_p


def test_encoder_config_single_input(single_input_cfg):
    arch_config = single_input_cfg["architecture"]
    input_config = single_input_cfg["input"]
    output_config = single_input_cfg["output"]

    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }

    config = EncoderDecoderConfig.parse(input_cfgs, output_cfgs, arch_config)

    assert config.encoder_config.channels == [16, 32, 64, 128]
    assert config.decoder_config.channels == [128, 64, 32, 16]
    assert isinstance(config.stem_configs, dict)
    assert isinstance(config.head_configs, dict)

    encdec = EncoderDecoder.from_config_dict(single_input_cfg)
    assert encdec.stems["x"][-1].n_params > 0

    x = torch.rand(1, 16, 128, 128)
    y = encdec(x)

    assert y.shape == (1, 1, 128, 128)
    assert isinstance(y, torch.Tensor)


MULTI_INPUT_CFG = """
[architecture]
name = "EncoderDecoder"

[architecture.stem]
depth = 2
out_channels = 64

[architecture.encoder]
channels = [64, 128, 256, 512]
stage_depths = [2, 2, 2, 2]
downsampling_factors = [2, 1, 2]

[architecture.decoder]
channels = [256, 128, 64]
stage_depths = [2, 2, 2]
upsampling_factors = [2, 1, 2]

[input.x_1]
n_features = 16
scale = 1

[input.x_2]
n_features = 32
scale = 1

[input.x_3]
n_features = 32
scale = 4

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

    assert config.encoder_config.channels == [64, 128, 256, 512]
    assert config.decoder_config.channels == [512, 256, 128, 64]
    assert isinstance(config.stem_configs, dict)
    assert isinstance(config.head_configs, dict)

    encdec = EncoderDecoder.from_config_dict(multi_input_cfg)
    assert encdec.stems["x_1"][-1].n_params > 0

    x = {
        "x_1": torch.rand(1, 16, 128, 128),
        "x_2": torch.rand(1, 32, 128, 128),
        "x_3": torch.rand(1, 32, 32, 32),
    }

    y = encdec(x)

    assert y["precip"].shape == (1, 1, 128, 128)
    assert isinstance(y["precip"], torch.Tensor)


MULTI_OUTPUT_CFG = """
[architecture]
name = "EncoderDecoder"

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

[output.snow]
kind = "Mean"
shape = [4, 4]
"""


@pytest.fixture
def multi_output_cfg():
    return toml.loads(MULTI_OUTPUT_CFG)


def test_encoder_config_multi_output(multi_output_cfg):
    arch_config = multi_output_cfg["architecture"]
    input_config = multi_output_cfg["input"]
    output_config = multi_output_cfg["output"]

    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }

    config = EncoderDecoderConfig.parse(input_cfgs, output_cfgs, arch_config)

    assert config.encoder_config.channels == [16, 32, 64, 128]
    assert config.decoder_config.channels == [128, 64, 32, 16]
    assert isinstance(config.stem_configs, dict)
    assert isinstance(config.head_configs, dict)

    encdec = EncoderDecoder.from_config_dict(multi_output_cfg)
    assert encdec.stems["x_1"][-1].n_params > 0

    x = {
        "x_1": torch.rand(1, 16, 128, 128),
        "x_2": torch.rand(1, 32, 128, 128),
    }

    y = encdec(x)

    assert "precip" in y
    assert "snow" in y
    assert y["snow"].shape == (1, 4, 4, 128, 128)
    assert isinstance(y["snow"], MeanTensor)


def test_encoder_config_multi_output(multi_output_cfg):
    """
    Instantiate an encoder-decoder architecture and ensure that it produces
    the expected outputs.
    """
    arch_config = multi_output_cfg["architecture"]
    input_config = multi_output_cfg["input"]
    output_config = multi_output_cfg["output"]

    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }

    config = EncoderDecoderConfig.parse(input_cfgs, output_cfgs, arch_config)

    assert config.encoder_config.channels == [16, 32, 64, 128]
    assert config.decoder_config.channels == [128, 64, 32, 16]
    assert isinstance(config.stem_configs, dict)
    assert isinstance(config.head_configs, dict)

    encdec = EncoderDecoder.from_config_dict(multi_output_cfg)
    assert encdec.stems["x_1"][-1].n_params > 0

    x = {
        "x_1": torch.rand(1, 16, 128, 128),
        "x_2": torch.rand(1, 32, 128, 128),
    }

    y = encdec(x)

    assert "precip" in y
    assert "snow" in y
    assert y["snow"].shape == (1, 4, 4, 128, 128)
    assert isinstance(y["snow"], MeanTensor)


def test_encoder_config_3d(multi_output_cfg):
    """
    Instantiate an encoder-decoder architecture and ensure that it produces
    the expected outputs.
    """
    arch_config = multi_output_cfg["architecture"]
    arch_config["stem"]["kind"] = "BasicConv3d"
    arch_config["head"] = {"default": {"kind": "BasicConv3d"}}
    arch_config["encoder"]["block_factory"] = "ResNeXt2Plus1"
    arch_config["encoder"]["aggregation_factory"] = "Linear3d"
    arch_config["encoder"]["downsampling_factors"] = [[1, 2, 2]] * 3
    arch_config["decoder"]["block_factory"] = "ResNeXt2Plus1"
    arch_config["decoder"]["upsampling_factors"] = [[1, 2, 2]] * 3
    arch_config["decoder"]["upsampling_factory"] = "Trilinear"
    input_config = multi_output_cfg["input"]
    output_config = multi_output_cfg["output"]

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

    encdec = EncoderDecoder.from_config_dict(multi_output_cfg)
    assert encdec.stems["x_1"][-1].n_params > 0

    x = {
        "x_1": [
            torch.rand(1, 16, 128, 128),
            torch.rand(1, 16, 128, 128),
            torch.rand(1, 16, 128, 128),
        ],
        "x_2": [
            torch.rand(1, 32, 128, 128),
            torch.rand(1, 32, 128, 128),
            torch.rand(1, 32, 128, 128),
        ]
    }

    y = encdec(x)

    assert "precip" in y
    assert "snow" in y
    assert isinstance(y["snow"], list)
    assert y["snow"][0].shape == (1, 4, 4, 128, 128)


def test_encoder_config_multi_output(multi_output_cfg):
    arch_config = multi_output_cfg["architecture"]
    input_config = multi_output_cfg["input"]
    output_config = multi_output_cfg["output"]

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

    encdec = EncoderDecoder.from_config_dict(multi_output_cfg)
    assert encdec.stems["x_1"][-1].n_params > 0

    x = {
        "x_1": torch.rand(1, 16, 128, 128),
        "x_2": torch.rand(1, 32, 128, 128),
    }

    y = encdec(x)

    assert "precip" in y
    assert "snow" in y
    assert y["snow"].shape == (1, 4, 4, 128, 128)
    assert isinstance(y["snow"], MeanTensor)


def test_save_and_load(multi_output_cfg, tmp_path):
    """
    Test saving and loading of an encoder-decoder model and ensure that
    original and loaded models have the same number of parameters.
    """
    arch_config = multi_output_cfg["architecture"]
    input_config = multi_output_cfg["input"]
    output_config = multi_output_cfg["output"]

    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }

    config = EncoderDecoderConfig.parse(input_cfgs, output_cfgs, arch_config)

    encdec = EncoderDecoder.from_config_dict(multi_output_cfg)
    encdec.save("encdec.pt")

    loaded = load_model("encdec.pt")

    assert encdec.n_params == loaded.n_params


PRESET_CFG = """
[architecture]
name = "EncoderDecoder"
preset = "{name}"

[input.x_1]
n_features = 4

[output.y]
kind = "Mean"
shape = 1
"""


PRESETS = ["UNet", "resnet_s", "resnext_s"]


@pytest.mark.parametrize("name", PRESETS)
def test_preset(name):
    """
    Ensure that preset models work with synthetic data.
    """
    cfg = toml.loads(PRESET_CFG.format(name=name))
    encdec = EncoderDecoder.from_config_dict(cfg)
    x, y = next(iter(data_loader_3d(8, 8)))
    pred = encdec(x)
    assert pred.shape[:2] == (8, 1)


MULTI_INPUT_MULTI_SCALE_CFG = """
[architecture]
name = "EncoderDecoder"

[architecture.encoder]
channels = [64, 128, 256, 512]
stage_depths = [2, 2, 2, 2]
downsampling_factors = [[2, 1], [1, 2], [2, 2]]

[architecture.decoder]
channels = [256, 128, 64, 32]
stage_depths = [2, 2, 2, 1]
upsampling_factors = [[2, 2], [1, 2], [2, 1], [2, 2]]

[architecture.stem.x_1]
depth = 1
downsampling = [2, 1]
out_channels = 27

[architecture.stem.x_2]
depth = 1
out_channels = 31
downsampling = 1

[input.x_1]
n_features = 16
scale = 1

[input.x_2]
n_features = 32
scale = 1

[input.x_3]
n_features = 32
scale = 2

[output.precip]
kind = "Mean"
shape = 1
"""


@pytest.fixture
def multi_input_multi_scale_cfg():
    return toml.loads(MULTI_INPUT_MULTI_SCALE_CFG)


def test_encoder_config_multi_input_multi_scale(multi_input_multi_scale_cfg):
    arch_config = multi_input_multi_scale_cfg["architecture"]
    input_config = multi_input_multi_scale_cfg["input"]
    output_config = multi_input_multi_scale_cfg["output"]

    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }

    config = EncoderDecoderConfig.parse(input_cfgs, output_cfgs, arch_config)

    assert config.encoder_config.channels == [64, 128, 256, 512]
    assert config.decoder_config.channels == [512, 256, 128, 64, 32]
    assert isinstance(config.stem_configs, dict)
    assert isinstance(config.head_configs, dict)

    encdec = EncoderDecoder.from_config_dict(multi_input_multi_scale_cfg)
    assert encdec.stems["x_1"][-1].n_params > 0

    x = {
        "x_1": torch.rand(1, 16, 128, 128),
        "x_2": torch.rand(1, 32, 128, 128),
        "x_3": torch.rand(1, 32, 64, 64),
    }

    y = encdec(x)

    assert y["precip"].shape == (1, 1, 256, 256)
    assert isinstance(y["precip"], torch.Tensor)


MULTI_STEP_MULTI_INPUT_MULTI_SCALE_CFG = """
[architecture]
name = "EncoderDecoder"

[architecture.encoder]
channels = [32, 64, 128]
stage_depths = [2, 2, 2]
downsampling_factors = [[1, 2, 1], [2, 1, 2]]
block_factory = "InvertedBottleneck2Plus1"
aggregation_factory = "Linear3d"

[architecture.decoder]
channels = [128, 64]
stage_depths = [2, 2]
upsampling_factors = [[2, 1, 2], [1, 2, 1]]
block_factory = "InvertedBottleneck2Plus1"
aggregation_factory = "Linear3d"
upsampling_factory = "Trilinear"

[architecture.stem]
kind = "BasicConv3d"

[architecture.stem.x_1]
kind = "BasicConv3d"
depth = 1
downsampling = [1, 1, 1]
out_channels = 27

[architecture.stem.x_2]
kind = "BasicConv3d"
depth = 1
out_channels = 31
downsampling = 1

[architecture.stem.x_3]
kind = "BasicConv3d"
depth = 1
downsampling = [2, 1, 1]
out_channels = 31

[architecture.head.default]
kind = "BasicConv3d"

[input.x_1]
n_features = 16
scale = 1

[input.x_2]
n_features = 32
scale = 1

[input.x_3]
n_features = 32
scale = 2

[output.precip]
kind = "Mean"
shape = 1
"""


@pytest.fixture
def multi_step_multi_input_multi_scale_cfg():
    return toml.loads(MULTI_STEP_MULTI_INPUT_MULTI_SCALE_CFG)


def test_encoder_config_multi_step_multi_input_multi_scale(
        multi_step_multi_input_multi_scale_cfg
):

    arch_config = multi_step_multi_input_multi_scale_cfg["architecture"]
    input_config = multi_step_multi_input_multi_scale_cfg["input"]
    output_config = multi_step_multi_input_multi_scale_cfg["output"]

    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }

    config = EncoderDecoderConfig.parse(input_cfgs, output_cfgs, arch_config)

    encdec = EncoderDecoder.from_config_dict(multi_step_multi_input_multi_scale_cfg)
    assert encdec.stems["x_1"][-1].n_params > 0

    x = {
        "x_1": [torch.rand(1, 16, 128, 128) for _ in range(4)],
        "x_2": [torch.rand(1, 32, 128, 128) for _ in range(4)],
        "x_3": [torch.rand(1, 32, 64, 64) for _ in range(4)],
    }

    y = encdec(x)

    assert y["precip"][0].shape == (1, 1, 128, 128)
    assert isinstance(y["precip"], list)
    assert isinstance(y["precip"][0], torch.Tensor)


MULTI_STEP_MULTI_INPUT_UPSAMPLING_CFG = """
[architecture]
name = "EncoderDecoder"
preset = "EfficientNetV2-2p1-Seq"

[architecture.stem]
kind = "BasicConv3d"
depth = 1
kernel_size = [1, 3, 3]
out_channels = 24
downsampling = [1, 2, 2]

[architecture.stem.atms]
kind = "BasicConv3d"
depth = 1
kernel_size = [1, 3, 3]
out_channels = 24
upsampling = [1, 2, 2]
upsampling_factory = "Trilinear"

[architecture.head.default]
kind = "BasicConv3d"

[input.cpcir]
n_features=1
scale = 4

[input.atms]
n_features=9
scale = 16

[output.surface_precip]
kind = "Quantiles"
shape = 1
quantiles = 32
"""


@pytest.fixture
def multi_step_multi_input_upsampling_cfg():
    return toml.loads(MULTI_STEP_MULTI_INPUT_UPSAMPLING_CFG)


def test_encoder_config_multi_step_multi_input_upsampling(
        multi_step_multi_input_upsampling_cfg
):

    arch_config = multi_step_multi_input_upsampling_cfg["architecture"]
    input_config = multi_step_multi_input_upsampling_cfg["input"]
    output_config = multi_step_multi_input_upsampling_cfg["output"]

    input_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
    }
    output_cfgs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
    }


    encdec = EncoderDecoder.from_config_dict(multi_step_multi_input_upsampling_cfg)

    x = {
        "cpcir": [torch.rand(1, 1, 256, 256) for _ in range(8)],
        "atms": [torch.rand(1, 9, 64, 64) for _ in range(8)],
    }

    y = encdec(x)

    assert y["surface_precip"][0].shape == (1, 32, 1, 256, 256)
    assert isinstance(y["surface_precip"], list)
    assert isinstance(y["surface_precip"][0], torch.Tensor)
