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
hidden_channels = 64
n_layers = 2
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.x]
n_features = 1

[output.y]
kind = "Mean"
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
batch_size = 64
optimizer = "SGD"
optimizer_args = {"lr"= 1e-3}
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
        batch_size = 8
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


def test_get_optimizer_and_scheduler(
        model_config_file,
        training_config_file
):
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

TRAINING_CONFIG_VAL_SPLIT = """
[warm_up]
dataset_module = "pytorch_retrieve.data.synthetic"
training_dataset = "Synthetic1d"
training_dataset_args = {"n_samples"=256}
validation_split = 0.25
n_epochs = 2
batch_size = 64
optimizer = "SGD"
optimizer_args = {"lr"= 1e-3}
metrics = ["Bias", "CorrelationCoef"]
"""


@pytest.fixture
def training_config_file_val_split(tmp_path):
    """
    Provides a path to a training config file with a the 'validation_split' set in a temporary directory.
    """
    output_path = tmp_path / "training.toml"
    with open(output_path, "w") as output:
        output.write(TRAINING_CONFIG_VAL_SPLIT)
    return output_path


def test_get_training_and_validation_dataset_from_split(training_config_file_val_split):
    """
    Ensure that create of training and validation datsets from a validation split works.
    """
    schedule = parse_training_config(read_config_file(training_config_file_val_split))
    training_data = schedule["warm_up"].get_training_dataset()
    validation_data = schedule["warm_up"].get_validation_dataset()

    assert validation_data is not None
    assert len(training_data) // 3 == len(validation_data)


def test_training(
        model_config_file,
        training_config_file,
        tmp_path,
        cpu_compute_config
):
    """
    Run training on synthetic data.
    """
    model = load_and_compile_model(model_config_file)
    schedule = parse_training_config(read_config_file(training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_training(tmp_path, module, compute_config=cpu_compute_config)

    # Assert that checkpoint files are created.
    ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(ckpts) > 0


def test_training_encoder_decoder(
        encoder_decoder_model_config_file,
        encoder_decoder_training_config_file,
        cpu_compute_config,
        tmp_path
):
    """
    Run training on synthetic data.
    """
    model = load_and_compile_model(encoder_decoder_model_config_file)
    schedule = parse_training_config(read_config_file(encoder_decoder_training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_training(tmp_path, module, compute_config=cpu_compute_config)

    # Assert that checkpoint files are created.
    ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(ckpts) > 0



def test_load_weights(
        model_config_file,
        training_config_file,
        tmp_path,
        cpu_compute_config
):
    """
    Run training on synthetic data.
    """
    model = load_and_compile_model(model_config_file)
    schedule = parse_training_config(read_config_file(training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_training(tmp_path, module, compute_config=cpu_compute_config)

    schedule["warm_up"].load_weights = str(tmp_path / "retrieval_model.pt")
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_training(tmp_path, module, compute_config=cpu_compute_config)

    # Assert that checkpoint files are created.
    ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(ckpts) > 0

    # Check loading weights from checkpoint.
    schedule["warm_up"].load_weights = str(ckpts[0])
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_training(tmp_path, module, compute_config=cpu_compute_config)
