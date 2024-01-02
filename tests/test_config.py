"""
Tests for the pytorch_retrieve.config module.
=============================================
"""


import yaml
import toml


from pytorch_retrieve.config import read_config_file, InputConfig, OutputConfig


def test_read_config_file(tmp_path):
    """
    Test reading a config dict in .toml and .yaml formats.
    """

    config = {
        "architecture": {"name": "EncoderDecoder"},
        "input": {"x": {"n_channels": 10}},
        "output": {"y": {"shape": [10, 10]}},
    }

    # Test reading .yaml file.
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as output:
        output.write(yaml.dump(config))

    config_loaded = read_config_file(config_file)
    assert all([key in config for key in config_loaded])

    # Test reading .toml file.
    config_file = tmp_path / "config.toml"
    with open(config_file, "w") as output:
        output.write(toml.dumps(config))

    config_loaded = read_config_file(config_file)
    assert all([key in config for key in config_loaded])


INPUT_CONFIGS = """
[input.x_1]
n_features = 16
scale = 4

[input.x_2]
n_features = 32
scale = 8
"""


def test_input_config():
    """
    Parse input configs and assert that the parsed attributes match the settings
    set in 'INPUT_CONFIGS'.
    """
    inpt_cfgs = toml.loads(INPUT_CONFIGS)
    inpt_cfgs = {
        name: InputConfig.parse(name, cfg) for name, cfg in inpt_cfgs["input"].items()
    }

    assert "x_1" in inpt_cfgs
    inpt_cfgs["x_1"].n_features == 16

    assert "x_2" in inpt_cfgs
    inpt_cfgs["x_2"].n_features == 8


OUTPUT_CONFIGS = """
[output.y_1]
kind = "mean"
shape = [1]

[output.y_2]
kind = "quantiles"
shape = [32]
"""


def test_output_config():
    """
    Parse output configs and assert that the parsed attributes match the settings
    set in 'OUTPUT_CONFIGS'.
    """
    output_cfgs = toml.loads(OUTPUT_CONFIGS)
    output_cfgs = {
        name: OutputConfig.parse(name, cfg)
        for name, cfg in output_cfgs["output"].items()
    }

    assert "y_1" in output_cfgs
    output_cfgs["y_1"].shape == [1]

    assert "y_2" in output_cfgs
    output_cfgs["y_2"].shape == [32]
