"""
Tests for the pytorch_retrieve.architectures.mlp architecture.
"""
from pathlib import Path
from tempfile import TemporaryDirectory


import lightning as L
import pytest
import toml
from torch.utils.data import DataLoader
from torch import nn

from pytorch_retrieve.config import read_config_file
from pytorch_retrieve.architectures import compile_architecture
from pytorch_retrieve.data.synthetic import Synthetic1d, Synthetic1dMultiOutput
from pytorch_retrieve.lightning import LightningRetrieval


def data_loader_1d(dataset_class, n_samples: int, batch_size: int) -> DataLoader:
    """
    A DataLoader providing batch of Synthetic1D data.
    """
    data = dataset_class(n_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def run_training(model: nn.Module, data_loader: DataLoader) -> None:
    """
    Runs one training epoch on the given neural network model.
    """
    with TemporaryDirectory() as tmp:
        mod = LightningRetrieval(model, model_dir=Path(tmp))
        trainer = L.Trainer(max_epochs=1, accelerator="cpu")
        trainer.fit(mod, train_dataloaders=data_loader)


SINGLE_INPUT_CFG = """
[architecture]
name = "MLP"

[architecture.body]
hidden_channels = 128
n_layers = 4
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.x]
n_features = 1

[output.y]
shape = [1,]
kind = "Mean"
"""


@pytest.fixture
def single_input_config(tmp_path):
    """
    A config file defining a single-input MLP retrieval architecture.
    """
    cfg_path = tmp_path / "model.toml"

    with open(cfg_path, "w") as output:
        output.write(SINGLE_INPUT_CFG)
    return cfg_path


def test_mlp_single_input(single_input_config):
    """
    Instantiate a single-input, single-output MLP retrieval and
     - propagate synthetic test data through the module
     - perform a 1-epoch training run
    """
    cfg_dict = read_config_file(single_input_config)
    mod = compile_architecture(cfg_dict)
    data_loader = data_loader_1d(Synthetic1d, 128, 8)
    x, y = next(iter(data_loader))
    y_pred = mod(x)
    run_training(mod, data_loader)


def test_save_mlp_single_input(single_input_config):
    """
    Instantiate a single-input, single-output MLP retrieval and
     - propagate synthetic test data through the module
     - perform a 1-epoch training run
    """
    cfg_dict = read_config_file(single_input_config)
    mlp = compile_architecture(cfg_dict)

    toml_str = toml.dumps(mlp.to_config_dict())
    cfg_dict = toml.loads(toml_str)
    mlp_2 = compile_architecture(cfg_dict)

    assert mlp.n_params == mlp_2.n_params


def test_mlp_single_input(single_input_config):
    """
    Instantiate a single-input, single-output MLP retrieval and
     - propagate synthetic test data through the module
     - perform a 1-epoch training run
    """
    cfg_dict = read_config_file(single_input_config)
    mod = compile_architecture(cfg_dict)
    data_loader = data_loader_1d(Synthetic1d, 128, 8)
    x, y = next(iter(data_loader))
    y_pred = mod(x)
    run_training(mod, data_loader)


MULTI_OUTPUT_CFG = """
[architecture]
name = "MLP"
[architecture.body]
hidden_channels = 128
n_layers = 4
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.x]
n_features = 1

[output.y]
shape = [1,]

[output.-y]
shape = [1,]
kind = "Mean"
"""


@pytest.fixture
def multi_output_config(tmp_path):
    """
    A config file defining a multi-output mlp retrieval architecture.
    """
    cfg_path = tmp_path / "model.toml"

    with open(cfg_path, "w") as output:
        output.write(SINGLE_INPUT_CFG)
    return cfg_path


def test_mlp_multi_input(multi_output_config):
    """
    Instantiate a single-input, single-output retrieval and propagate synthetic
    test data through the resulting module.
    """
    cfg_dict = read_config_file(multi_output_config)
    mod = compile_architecture(cfg_dict)
    data_loader = data_loader_1d(Synthetic1dMultiOutput, 128, 8)
    x, y = next(iter(data_loader))
    y_pred = mod(x)
    run_training(mod, data_loader)
