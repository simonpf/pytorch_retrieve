import pytest
from torch.utils.data import DataLoader
import toml

from pytorch_retrieve.architectures import compile_architecture
from pytorch_retrieve.training import TrainingConfig
from pytorch_retrieve.data.synthetic import Synthetic1d, Synthetic1dMultiOutput


def data_loader_1d(n_samples: int, batch_size: int) -> DataLoader:
    """
    A DataLoader providing batch of Synthetic1D data.
    """
    data = Synthetic1d(n_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


MODEL_CONFIG_MLP = """
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

TRAINING_CONFIG_MLP = """
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
def mlp():
    return compile_architecture(toml.loads(MODEL_CONFIG_MLP))


@pytest.fixture
def mlp_training_schedule():
    training_config = toml.loads(TRAINING_CONFIG_MLP)
    training_schedule = {
        name: TrainingConfig.parse(name, cfg) for name, cfg in training_config.items()
    }
    return training_schedule
