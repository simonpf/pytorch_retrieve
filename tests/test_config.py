"""
Tests for the pytorch_retrieve.config module.
=============================================
"""

import os

import yaml
import toml

from conftest import MODEL_CONFIG_MLP
import pytest

from pytorch_retrieve.config import (
    replace_environment_variables,
    get_config_attr,
    read_config_file,
    InputConfig,
    OutputConfig,
    InferenceConfig,
)
from pytorch_retrieve.modules.transformations import MinMax
from pytorch_retrieve.modules.output import Mean, Quantiles



def test_replace_environment_variables():
    """
    Tests replacement of environment variables in string starting with
    "ENV::".
    """
    os.environ["PYTORCH_RETRIEVE"] = "TEST"
    string = replace_environment_variables("{PYTORCH_RETRIEVE}")
    assert string == "{PYTORCH_RETRIEVE}"
    string = replace_environment_variables("ENV::{PYTORCH_RETRIEVE}")
    assert string == "TEST"
    string = replace_environment_variables(1)
    assert string == 1


def test_get_config_attr():
    """
    Test extraction of attributes from configuration dicts.
    """
    config = {
        "a": "a",
        "b": 1,
        "c": [1, 2, 3],
    }

    a = get_config_attr("a", str, config, "test")
    assert a == "a"

    b = get_config_attr("b", int, config, "test")
    assert b == 1

    c = get_config_attr("c", list, config, "test")
    assert isinstance(c, list)

    d = get_config_attr("d", str, config, "test", "d")
    assert d == "d"


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
meta_data = "x_2_meta"
encoding = "x_2"
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
    assert inpt_cfgs["x_1"].n_features == 16

    assert "x_2" in inpt_cfgs
    assert inpt_cfgs["x_2"].n_features == 32
    assert inpt_cfgs["x_2"].n_features == 32
    assert inpt_cfgs["x_2"].meta_data == "x_2_meta"
    assert inpt_cfgs["x_2"].encoding == "x_2"


OUTPUT_CONFIGS = """
[output.y_1]
kind = "Mean"
shape = [1]

[output.y_1_scalar]
kind = "Mean"
shape = 1
conditional = "y_1"
encoding = "position"

[output.y_2]
kind = "Quantiles"
shape = 32
quantiles = 32
transformation = "MinMax"
transformation_args = {"x_min"=10.0, "x_max"=100}

[output.y_3]
kind = "Detection"
shape = 1
quantiles = 32

[output.y_4]
kind = "Classification"
shape = 1
n_classes = 8
transformation = "HistEqual"
transformation_args = {"n_steps"=128}
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
    assert output_cfgs["y_1"].shape == (1,)
    assert output_cfgs["y_1"].get_output_shape() == (1,)
    layer = output_cfgs["y_1"].get_output_layer()
    assert isinstance(layer, Mean)

    assert "y_2" in output_cfgs
    assert output_cfgs["y_2"].shape == (32,)
    assert output_cfgs["y_2"].get_output_shape() == (32, 32,)
    layer = output_cfgs["y_2"].get_output_layer()
    assert isinstance(layer, Quantiles)
    assert isinstance(layer.transformation, MinMax)

    assert len(output_cfgs["y_1"].get_output_dimensions()) == 1
    assert len(output_cfgs["y_1"].get_output_coordinates()) == 0
    assert len(output_cfgs["y_1"].extra_dimensions) == 0
    assert len(output_cfgs["y_1_scalar"].get_output_dimensions()) == 0
    assert len(output_cfgs["y_1_scalar"].get_output_coordinates()) == 0
    assert output_cfgs["y_1_scalar"].conditional == "y_1"
    assert output_cfgs["y_1_scalar"].encoding == "position"

    assert len(output_cfgs["y_2"].extra_dimensions) == 1
    assert len(output_cfgs["y_2"].get_output_dimensions()) == 1
    assert len(output_cfgs["y_2"].get_output_coordinates()) == 1

    assert len(output_cfgs["y_3"].extra_dimensions) == 0
    assert len(output_cfgs["y_3"].get_output_dimensions()) == 0
    assert len(output_cfgs["y_3"].get_output_coordinates()) == 0

    assert len(output_cfgs["y_4"].extra_dimensions) == 1
    assert len(output_cfgs["y_4"].get_output_dimensions()) == 1
    assert len(output_cfgs["y_4"].get_output_coordinates()) == 1
    assert output_cfgs["y_4"].get_output_shape() == (8,)

    assert isinstance(output_cfgs["y_4"].transformation_args, dict)
    assert isinstance(output_cfgs["y_4"].to_config_dict()["transformation_args"], dict)


MISSING_QUANTILES = """
[output.y_2]
kind = "Quantiles"
shape = 32
transformation = "SquareRoot"
"""

def test_output_config_missing_quantiles():
    """
    Ensures that an error is raised when the quantiles attribute is missing in a Quantiles
    output.
    """
    output_cfgs = toml.loads(MISSING_QUANTILES)
    with pytest.raises(ValueError):
        output_cfgs = {
            name: OutputConfig.parse(name, cfg)
            for name, cfg in output_cfgs["output"].items()
        }


MISSING_N_CLASSES = """
[output.y_2]
kind = "Classification"
shape = 32
transformation = "SquareRoot"
"""

def test_output_config_missing_n_classes():
    """
    Ensures that an error is raised when the n_classes attribute is missing in a Classficiation
    output.
    """
    output_cfgs = toml.loads(MISSING_N_CLASSES)
    with pytest.raises(ValueError):
        output_cfgs = {
            name: OutputConfig.parse(name, cfg)
            for name, cfg in output_cfgs["output"].items()
        }


INFERENCE_CONFIG = """
batch_size = 12
tile_size = [32, 32]
spatial_overlap = 8
temporal_overlap = 0

[retrieval_output.surface_precip]
surface_precip_quantiles = {retrieval_output="Full"}
surface_precip_mean = "ExpectedValue"
surface_precip_median = {retrieval_output="Quantiles", tau=[0.5]}
surface_precip_heavy = {retrieval_output="ExceedanceProbability", threshold=10.0}
"""


def test_inference_config():
    """
    Test parsing of inference configs.
    """
    model_config = toml.loads(MODEL_CONFIG_MLP)
    model_config["output"]["surface_precip"] = {"kind": "Mean"}
    output_config = {
        name: OutputConfig.parse(name, value)
        for name, value in model_config["output"].items()
    }
    inference_config = toml.loads(INFERENCE_CONFIG)
    inference_config = InferenceConfig.parse(output_config, inference_config)

    assert inference_config.batch_size == 12
    assert inference_config.tile_size == (32, 32)
    assert inference_config.spatial_overlap == 8
    assert inference_config.temporal_overlap == 0

    qty = inference_config.retrieval_output["surface_precip"]["surface_precip_mean"]
    assert qty.retrieval_output == "ExpectedValue"

    ic_saved = toml.dumps(inference_config.to_dict())
    inference_config_loaded = toml.loads(ic_saved)
    inference_config_loaded = InferenceConfig.parse(
        output_config, inference_config_loaded
    )

    assert inference_config_loaded.batch_size == 12
    assert inference_config_loaded.tile_size == (32, 32)
    assert inference_config_loaded.spatial_overlap == 8
    assert inference_config_loaded.temporal_overlap == 0

    qty = inference_config_loaded.retrieval_output["surface_precip"][
        "surface_precip_mean"
    ]
    assert qty.retrieval_output == "ExpectedValue"
