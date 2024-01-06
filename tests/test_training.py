"""
Tests for the pytorch_retrieve.training module.
"""
import os

import pytest
import torch
import toml
from typing import List

from pytorch_retrieve.training import (
    TrainingConfig,
    parse_training_config,
    run_training,
)
from pytorch_retrieve.architectures import load_and_compile_model
from pytorch_retrieve.lightning import LightningRetrieval
from pytorch_retrieve.config import read_config_file

MODEL_CONFIG = """
[architecture]
name = "MLP"

[architecture.body]
hidden_channels = 128
n_layers = 4
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.x]
in_channels = 1

[output.y]
shape = [1,]
"""


@pytest.fixture
def model_config_file(tmp_path):
    """
    Provides a path to a trainign config file in a temporary directory.
    """
    output_path = tmp_path / "model.toml"
    with open(output_path, "w") as output:
        output.write(MODEL_CONFIG)
    return output_path


TRAINING_CONFIG = """
[warm_up]
dataset_module = "pytorch_retrieve.data.synthetic"
training_dataset = "Synthetic1d"
training_dataset_args = {"n_samples"=256}
validation_dataset_args = {"n_samples"=128}
n_epochs = 2
batch_size = 8
optimizer = "SGD"
optimizer_kwargs = {"lr"= 1e-3}
metrics = ["Bias", "CorrelationCoef"]
"""


@pytest.fixture
def training_config_file(tmp_path):
    """
    Provides a path to a trainign config file in a temporary directory.
    """
    output_path = tmp_path / "training.toml"
    with open(output_path, "w") as output:
        output.write(TRAINING_CONFIG)
    return output_path


def test_variable_replacement(monkeypatch):
    """
    Ensure that environment variables in 'training_data_args' and
    'validation_data_args' are replace correctly.
    """
    monkeypatch.setattr(os, "environ", {"PATH": "TEST"})
    config_dict = toml.loads(
        """
        [warm_up]
        dataset_module = "pytorch_retrieve.data.synthetic"
        training_dataset = "Synthetic1D"
        training_dataset_args = {"path" = "ENV::{PATH}"}
        validation_dataset_args = {"path" = "ENV::{PATH}"}
        n_epochs = 10
        """
    )
    training_config = TrainingConfig.parse("warm_up", config_dict["warm_up"])
    assert training_config.training_dataset_args["path"] == "TEST"
    assert training_config.validation_dataset_args["path"] == "TEST"


def test_get_training_dataset(training_config_file):
    """
    Test instantiating a test data loader and ensure that it returns training
    samples.
    """
    schedule = parse_training_config(read_config_file(training_config_file))
    config = schedule["warm_up"]
    training_dataset = config.get_training_dataset()

    x, y = training_dataset[0]
    assert isinstance(x, torch.Tensor)


def test_get_optimizer_and_scheduler(model_config_file, training_config_file):
    """
    Test instantiating optimizer and scheduler
    """
    schedule = parse_training_config(read_config_file(training_config_file))
    model = load_and_compile_model(model_config_file)
    config = schedule["warm_up"]
    optimizer, scheduler = config.get_optimizer_and_scheduler("warm_up", model)


def test_get_metrics_dict(model_config_file, training_config_file):
    """
    Ensure that metric objects are parsed and instantiated correctly.
    """
    model_config = read_config_file(model_config_file)
    outputs = list(model_config["output"].keys())
    training_config = parse_training_config(read_config_file(training_config_file))
    metrics = training_config["warm_up"].get_metrics_dict(outputs)

    assert len(metrics) == 1
    assert len(metrics["y"]) == 2


def test_training(model_config_file, training_config_file, tmp_path):
    """
    Run training on synthetic data.
    """
    model = load_and_compile_model(model_config_file)
    schedule = parse_training_config(read_config_file(training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    module.current_training_config
    run_training(tmp_path, module, None)

    # Assert that checkpoint files are created.
    ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(ckpts) > 0
