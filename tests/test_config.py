"""
Tests for the probreg.config module.
====================================
"""


import yaml
import toml


from probreg.config import read_config_file


def test_read_config_file(tmp_path):
    """
    Test reading a config dict in .toml and .yaml formats.
    """

    config = {
        "architecture": {
            "name": "EncoderDecoder"
        },
        "input": {
            "x": {
                "n_channels": 10
            }
        },
        "output" : {
            "y": {
                "shape": [10, 10]
            }
        }
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
