"""
Tests for the pytorch_retrieve.training module.
"""
import os

import pytest
import torch
import toml
from typing import List

from pytorch_retrieve.eda import (
    run_eda
)
from pytorch_retrieve.training import (
    TrainingConfig,
    parse_training_config,
    run_training,
    freeze_modules
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

[input.x_1]
n_features = 1
normalize = "minmax"

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


MULTI_OUTPUT_MODEL_CONFIG = """
[architecture]
name = "MLP"

[architecture.body]
hidden_channels = 64
n_layers = 2
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.x_1]
n_features = 1
normalize = "minmax"

[output.y]
kind = "Mean"
shape = [1,]

[output.-y]
kind = "Mean"
shape = [1,]
"""


@pytest.fixture
def multi_output_model_config_file(tmp_path):
    """
    Provides a path to a trainign config file in a temporary directory.
    """
    output_path = tmp_path / "model.toml"
    with open(output_path, "w") as output:
        output.write(MULTI_OUTPUT_MODEL_CONFIG)
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


MULTI_OUTPUT_TRAINING_CONFIG = """
[warm_up]
dataset_module = "pytorch_retrieve.data.synthetic"
training_dataset = "Synthetic1dMultiOutput"
training_dataset_args = {"n_samples"=256}
validation_dataset_args = {"n_samples"=128}
n_epochs = 2
batch_size = 64
optimizer = "SGD"
optimizer_args = {"lr"= 1e-3}
metrics = ["Bias", "CorrelationCoef"]
"""

@pytest.fixture
def multi_output_training_config_file(tmp_path):
    """
    Provides a path to a trainign config file in a temporary directory.
    """
    output_path = tmp_path / "training.toml"
    with open(output_path, "w") as output:
        output.write(MULTI_OUTPUT_TRAINING_CONFIG)
    return output_path


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
        monkeypatch,
        model_config_file,
        training_config_file,
        tmp_path,
        cpu_compute_config
):
    """
    Run training on synthetic data.
    """
    monkeypatch.chdir(tmp_path)
    model = load_and_compile_model(model_config_file)
    schedule = parse_training_config(read_config_file(training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_eda(
        tmp_path / "stats",
        model.input_config,
        model.output_config,
        schedule["warm_up"],
        compute_config=cpu_compute_config
    )

    model = load_and_compile_model(model_config_file)
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_training(tmp_path, module, compute_config=cpu_compute_config)

    # Assert that checkpoint files are created.
    ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(ckpts) > 0


def test_training_multi_output(
        monkeypatch,
        multi_output_model_config_file,
        multi_output_training_config_file,
        tmp_path,
        cpu_compute_config
):
    """
    Test training of MLP model with multiple outputs.
    """
    monkeypatch.chdir(tmp_path)
    model = load_and_compile_model(multi_output_model_config_file)
    schedule = parse_training_config(read_config_file(multi_output_training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_eda(
        tmp_path / "stats",
        model.input_config,
        model.output_config,
        schedule["warm_up"],
        compute_config=cpu_compute_config
    )

    model = load_and_compile_model(multi_output_model_config_file)
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_training(
        tmp_path,
        module,
        compute_config=cpu_compute_config
    )

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
    Run training on multi-output synthetic data.
    """
    model = load_and_compile_model(encoder_decoder_model_config_file)
    schedule = parse_training_config(read_config_file(encoder_decoder_training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_eda(
        tmp_path / "stats",
        model.input_config,
        model.output_config,
        schedule["warm_up"],
        compute_config=cpu_compute_config
    )
    run_training(tmp_path, module, compute_config=cpu_compute_config)

    # Assert that checkpoint files are created.
    ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(ckpts) > 0



def test_load_weights(
        monkeypatch,
        model_config_file,
        training_config_file,
        tmp_path,
        cpu_compute_config
):
    """
    Test that loading of weights from a pre-trained model works.
    """
    monkeypatch.chdir(tmp_path)
    model = load_and_compile_model(model_config_file)
    schedule = parse_training_config(read_config_file(training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_eda(
        tmp_path / "stats",
        model.input_config,
        model.output_config, schedule["warm_up"],
        compute_config=cpu_compute_config
    )

    model = load_and_compile_model(model_config_file)
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


MODEL_CONFIG_EXTRA_INPUTS = """
[architecture]
name = "MLP"

[architecture.body]
hidden_channels = 64
n_layers = 2
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.x_1]
n_features = 1
normalize = "minmax"

[input.x_2]
n_features = 1

[output.y]
kind = "Mean"
shape = [1,]
"""

@pytest.fixture
def model_config_file_extra_inputs(tmp_path):
    """
    Provides a path to a training config file in a temporary directory.
    """
    output_path = tmp_path / "model_extra_inputs.toml"
    with open(output_path, "w") as output:
        output.write(MODEL_CONFIG_EXTRA_INPUTS)
    return output_path


TRAINING_CONFIG_EXTRA_INPUTS = """
[warm_up]
dataset_module = "pytorch_retrieve.data.synthetic"
training_dataset = "Synthetic1dMultiInput"
training_dataset_args = {"n_samples"=256}
validation_dataset_args = {"n_samples"=128}
n_epochs = 2
batch_size = 64
optimizer = "SGD"
optimizer_args = {"lr"= 1e-3}
metrics = ["Bias", "CorrelationCoef"]
"""


@pytest.fixture
def training_config_file_extra_inputs(tmp_path):
    """
    Provides a path to a trainign config file in a temporary directory.
    """
    output_path = tmp_path / "training_extra_inputs.toml"
    with open(output_path, "w") as output:
        output.write(TRAINING_CONFIG_EXTRA_INPUTS)
    return output_path



def test_load_weights_non_strict(
        monkeypatch,
        model_config_file,
        model_config_file_extra_inputs,
        training_config_file,
        training_config_file_extra_inputs,
        tmp_path,
        cpu_compute_config
):
    """
    Test that loading of weights works even for slightly difference architectures.
    """
    monkeypatch.chdir(tmp_path)
    model = load_and_compile_model(model_config_file)
    schedule = parse_training_config(read_config_file(training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_eda(
        tmp_path / "stats",
        model.input_config,
        model.output_config,
        schedule["warm_up"],
        compute_config=cpu_compute_config
    )

    model = load_and_compile_model(model_config_file)
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)

    run_training(tmp_path, module, compute_config=cpu_compute_config)


    schedule["warm_up"].load_weights = str(tmp_path / "retrieval_model.pt")
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_training(tmp_path, module, compute_config=cpu_compute_config)

    # Assert that checkpoint files are created.
    ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert len(ckpts) > 0

    # Check loading weights from checkpoint.
    model_new = load_and_compile_model(model_config_file_extra_inputs)
    schedule = parse_training_config(read_config_file(training_config_file_extra_inputs))
    schedule["warm_up"].load_weights = str(ckpts[0])
    module = LightningRetrieval(model_new, training_schedule=schedule, model_dir=tmp_path)
    run_training(tmp_path, module, compute_config=cpu_compute_config)


def test_freeze(
        monkeypatch,
        model_config_file,
        training_config_file,
        tmp_path,
        cpu_compute_config
):
    """
    Test that loading of weights from a pre-trained model works.
    """
    monkeypatch.chdir(tmp_path)
    model = load_and_compile_model(model_config_file)
    schedule = parse_training_config(read_config_file(training_config_file))
    module = LightningRetrieval(model, training_schedule=schedule, model_dir=tmp_path)
    run_eda(
        tmp_path / "stats",
        model.input_config,
        model.output_config,
        schedule["warm_up"],
        compute_config=cpu_compute_config
    )

    model = load_and_compile_model(model_config_file)
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


MODEL_CONFIG_EXTRA_INPUTS = """
[architecture]
name = "MLP"

[architecture.body]
hidden_channels = 64
n_layers = 2
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.x_1]
n_features = 1
normalize = "minmax"

[input.x_2]
n_features = 1

[output.y]
kind = "Mean"
shape = [1,]
"""

@pytest.fixture
def model_config_file_extra_inputs(tmp_path):
    """
    Provides a path to a training config file in a temporary directory.
    """
    output_path = tmp_path / "model_extra_inputs.toml"
    with open(output_path, "w") as output:
        output.write(MODEL_CONFIG_EXTRA_INPUTS)
    return output_path


TRAINING_CONFIG_EXTRA_INPUTS = """
[warm_up]
dataset_module = "pytorch_retrieve.data.synthetic"
training_dataset = "Synthetic1dMultiInput"
training_dataset_args = {"n_samples"=256}
validation_dataset_args = {"n_samples"=128}
n_epochs = 2
batch_size = 64
optimizer = "SGD"
optimizer_args = {"lr"= 1e-3}

[[warm_up.metrics]]
name = "Bias"
conditional = {x_1 = [1.5, 25.5, 8.0]}

[[warm_up.metrics]]
name = "CorrelationCoef"
conditional = {x_1 = [1.5, 25.5, 8.0]}

[[warm_up.metrics]]
name = "MSE"
conditional = {x_1 = [1.5, 25.5, 8.0]}

[[warm_up.metrics]]
name = "PlotSamples"
conditional = {x_1 = [1.5, 25.5, 8.0]}
"""


@pytest.fixture
def training_config_file_extra_inputs(tmp_path):
    """
    Provides a path to a trainign config file in a temporary directory.
    """
    output_path = tmp_path / "training_extra_inputs.toml"
    with open(output_path, "w") as output:
        output.write(TRAINING_CONFIG_EXTRA_INPUTS)
    return output_path


TRAINING_CONFIG_FREEZE = """
[stage_1]
dataset_module = "pytorch_retrieve.data.synthetic"
training_dataset = "Synthetic1d"
training_dataset_args = {"n_samples"=256}
validation_dataset_args = {"n_samples"=128}
n_epochs = 2
batch_size = 64
optimizer = "SGD"
optimizer_args = {"lr"= 1e-3}
metrics = ["Bias", "CorrelationCoef"]
freeze = ["body"]
"""


@pytest.fixture
def training_config_file_freeze(tmp_path):
    """
    Provides a path to a trainign config file in a temporary directory.
    """
    output_path = tmp_path / "training.toml"
    with open(output_path, "w") as output:
        output.write(TRAINING_CONFIG_FREEZE)
    return output_path


def test_freeze_modules(
        monkeypatch,
        model_config_file,
        model_config_file_extra_inputs,
        training_config_file_freeze,
        tmp_path,
        cpu_compute_config
):
    """
    Test freezing of components listed in training config.
    """
    monkeypatch.chdir(tmp_path)
    model = load_and_compile_model(model_config_file)
    schedule = parse_training_config(read_config_file(training_config_file_freeze))


    for param in model.body.parameters():
        assert param.requires_grad

    freeze = schedule["stage_1"].freeze
    freeze_modules(model, freeze)

    for param in model.body.parameters():
        assert not param.requires_grad
