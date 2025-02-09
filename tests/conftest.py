import pytest
from torch.utils.data import DataLoader
import toml

from pytorch_retrieve.architectures import compile_architecture
from pytorch_retrieve.config import ComputeConfig
from pytorch_retrieve.training import TrainingConfig
from pytorch_retrieve.data.synthetic import (
    Synthetic1d,
    Synthetic1dMultiOutput,
    Synthetic3d
)


def data_loader_1d(n_samples: int, batch_size: int) -> DataLoader:
    """
    A DataLoader providing batch of Synthetic1D data.
    """
    data = Synthetic1d(n_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)

def data_loader_3d(n_samples: int, batch_size: int) -> DataLoader:
    """
    A DataLoader providing batch of Synthetic1D data.
    """
    data = Synthetic3d(n_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


CPU_COMPUTE_CONFIG = """
accelerator="cpu"
precision=32
devices=1
strategy="auto"
"""

@pytest.fixture
def cpu_compute_config():
    """
    A compute config object for training on the CPU.
    """
    compute_config = ComputeConfig.parse(toml.loads(CPU_COMPUTE_CONFIG))
    return compute_config

MODEL_CONFIG_MLP = """
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

TRAINING_CONFIG_MLP = """
[warm_up]
dataset_module = "pytorch_retrieve.data.synthetic"
training_dataset = "Synthetic1d"
training_dataset_args = {"n_samples"=256}
validation_dataset_args = {"n_samples"=128}
n_epochs = 2
batch_size = 8
optimizer = "SGD"
optimizer_args = {"lr"= 1e-3}
metrics = ["Bias", "CorrelationCoef"]
n_data_loader_workers = 0
persistent_workers = false
"""


@pytest.fixture
def mlp():
    return compile_architecture(toml.loads(MODEL_CONFIG_MLP))


@pytest.fixture
def mlp_training_schedule():
    training_config = toml.loads(TRAINING_CONFIG_MLP)
    training_schedule = {
        name: TrainingConfig.parse(name, cfg) for name, cfg in training_config.items()
    }
    return training_schedule


ENCODER_DECODER_CONFIG = """

[architecture]
name = "EncoderDecoder"

[architecture.stem]
depth = 1
activation_factory = "ReLU"
block_factory = "BasicConv"
out_channels = 16

[architecture.encoder]
channels = [16, 32]
stage_depths = [2, 2]
activation_factory = "ReLU"
block_factory = "BasicConv"
normalization_factory = "none"
dowsampling_factory = "MaxPool"

[architecture.decoder]
channels = [16]
stage_depths = [2]
block_factory = "BasicConv"
activation_factory = "ReLU"
upsampling_factory = "ConvTranspose"
skip_connections = true
normalization_factory = "none"

[input.x]
n_features = 4
normalize = "none"

[output.y]
kind = "Mean"
shape = [1,]
"""


@pytest.fixture
def encoder_decoder_model_config_file(tmp_path):
    """
    Provides a path to a trainign config file in a temporary directory.
    """
    output_path = tmp_path / "model.toml"
    with open(output_path, "w") as output:
        output.write(ENCODER_DECODER_CONFIG)
    return output_path


ENCODER_DECODER_TRAINING_CONFIG = """
[warm_up]
dataset_module = "pytorch_retrieve.data.synthetic"
training_dataset = "Synthetic3d"
training_dataset_args = {"n_samples"=64}
validation_dataset_args = {"n_samples"=16}
n_epochs = 2
batch_size = 8
optimizer = "SGD"
optimizer_args = {"lr"= 1e-3}
metrics = ["Bias", "CorrelationCoef", "PlotSamples"]
n_data_loader_workers = 0
persistent_workers = false
"""


@pytest.fixture
def encoder_decoder_training_config_file(tmp_path):
    """
    Provides a path to a trainign config file in a temporary directory.
    """
    output_path = tmp_path / "training.toml"
    with open(output_path, "w") as output:
        output.write(ENCODER_DECODER_TRAINING_CONFIG)
    return output_path
