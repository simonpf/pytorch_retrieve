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
