"""
Tests for the pytorch_retrieve.architectures.model module.
"""
from pytorch_retrieve.architectures.model import RetrievalModel
from pytorch_retrieve.architectures.mlp import MLP


def test_load_save(tmp_path):
    """
    Test loading and saving of retrieval models.
    """
    model = MLP.from_config_dict(config_dict={
        "test": "test",
        "architecture": {
            "name": "MLP",
            "body": {
                "hidden_channels": 128,
                "n_layers": 4
            }
        },
        "input": {
            "x": {"n_features": 4}
        },
        "output": {
            "y": {"kind": "Mean"}
        }
    })
    assert model.inference_config == None


def test_inputs_and_outputs(tmp_path):
    """
    Test the 'inputs' and 'outputs' properties.
    """
    model = MLP.from_config_dict(config_dict={
        "test": "test",
        "architecture": {
            "name": "MLP",
            "body": {
                "hidden_channels": 128,
                "n_layers": 4
            }
        },
        "input": {
            "x": {"n_features": 4}
        },
        "output": {
            "y": {"kind": "Mean"}
        }
    })

    inputs = model.input_config
    assert "x" in inputs
    assert inputs["x"].n_features == 4

    outputs = model.output_config
    assert "y" in outputs
    assert outputs["y"].kind == "Mean"
