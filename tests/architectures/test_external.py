"""
Tests compiling external models.
"""
import pytest

from pytorch_retrieve.architectures import compile_architecture


DUMMY_MODULE = """

import torch
from torch import nn

class DummyModule(nn.Module):
    def __init__(self, **kwargs):
        for name, obj in kwargs.items():
            setattr(self, name, obj)
"""

@pytest.fixture
def dummy_module(tmp_path):
    """
    Creates a Python module with a dummy PyTorch module.
    """
    with open(tmp_path / "dummy_module.py", "w") as output:
        output.write(DUMMY_MODULE)
    return tmp_path


def test_external_model(dummy_module, monkeypatch):
    """
    Test compiling an external model from an independent Python package.
    """
    monkeypatch.syspath_prepend(dummy_module)

    model_cfg = {
        "architecture": {
            "model_class": "dummy_module.DummyModule",
            "arg_1": 1,
            "arg_2": "something_else"
        }
    }
    model = compile_architecture(model_cfg)

    assert model.arg_1 == 1
    assert model.arg_2 == "something_else"
