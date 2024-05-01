"""
Tests for the pytorch_retrieve.architectures.model module.
"""
import toml

from pytorch_retrieve.inference import InferenceConfig
from pytorch_retrieve.architectures import load_model
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


INFERENCE_CONFIG = """
tile_size = [128, 64]
spatial_overlap = 16
input_loader = "gprof_nn.retrieval.InputLoader"
input_loader_args = {config="3d"}

[retrieval_output.y]
surface_precip = "ExpectedValue"
surface_precip_terciles = {retrieval_output="Quantiles", tau=[0.33, 0.66]}
probability_of_precipitation = {retrieval_output="ExceedanceProbability", threshold=1e-3}
"""

def test_save_inference_config(tmp_path):
    """
    Ensure that saving a model with inference config works.
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

    inference_config = InferenceConfig.parse(model.output_config, toml.loads(INFERENCE_CONFIG))
    model.inference_config = inference_config

    model.save(tmp_path / "model.pt")
    model_loaded = load_model(tmp_path / "model.pt")

    assert model_loaded.inference_config is not None
    assert len(model_loaded.inference_config.retrieval_output["y"]) == 3
