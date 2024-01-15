"""
pytorch_retrieve.eda
====================

Implements functionality to perform explorative data analysis extracting
basic statistics from datasets.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import click
import lightning as L
import torch
from torch import nn

from pytorch_retrieve.modules.input import InputLayer
from pytorch_retrieve.config import read_config_file, InputConfig, ComputeConfig
from pytorch_retrieve.training import parse_training_config, TrainingConfig
from pytorch_retrieve.utils import read_model_config, read_training_config

LOGGER = logging.getLogger(__name__)


class EDAModule(L.LightningModule):
    """
    Lightning module for performing EDA on training data.

    The main purpose of the lightning module is to piggy-back the distributed
    data loading provided by pytorch to speed up the EDA.
    """

    def __init__(self, input_configs: Dict[str, InputConfig], model_directory: Path):
        """
        Args:
            input_configs: Dictionary containing the input configurations for all
                retrieval inputs.
            model_directory: The directory where the retrieval artifacts will
                stored.
        """
        super().__init__()
        self.input_modules = nn.ModuleDict(
            {
                name: InputLayer(name, cfg.n_features, model_path=model_directory)
                for name, cfg in input_configs.items()
            }
        )
        self.params = nn.Parameter(torch.zeros(1), requires_grad=True)
        for mod in self.input_modules.values():
            mod.reset()

    def configure_optimizers(self):
        """
        Dummy function required by lightning.
        """
        params = nn.Parameter(torch.zeros(1), requires_grad=True)
        optimizer = torch.optim.Adam([params], lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        This just records input data statistics for each batch.
        """
        inputs, outputs = batch

        for name, mod in self.input_modules.items():
            if not isinstance(inputs, dict):
                inputs = {name: inputs}
            mod(inputs[name])

        return None

    def on_train_epoch_end(self):
        """
        Signal the end of the epoch to all input modules.
        """
        for name, mod in self.input_modules.items():
            mod.epoch_finished(self)


def run_eda(
    model_directory: Path,
    input_configs: Dict[str, InputConfig],
    training_configs: Dict[str, TrainingConfig],
    compute_config: Optional[ComputeConfig] = None,
) -> None:
    """
    Performs EDA for given input and training configs.

    Args:
        model_directory: The directory in which to store the computed statistics.
        input_configs: A dictionary mapping input names to corresponding input
            configurations.
        training_configs: A dictionary mapping training stage names to corresponding
            training configs.
    """
    training_config = next(iter(training_configs.values()))
    training_loader = training_config.get_training_data_loader()

    if compute_config is None:
        compute_config = ComputeConfig()

    eda_module = EDAModule(input_configs, model_directory)
    trainer = L.Trainer(
        max_epochs=2,
        logger=None,
        precision=compute_config.precision,
        accelerator=compute_config.accelerator,
        devices=compute_config.devices,
    )
    trainer.fit(
        eda_module,
        train_dataloaders=training_loader,
    )


@click.option(
    "--model_path",
    default=None,
    help="The model directory. Defaults to the current working directory",
)
@click.option(
    "--model_config",
    default=None,
    help=(
        "Path to the model config file. If not provided, pytorch_retrieve "
        " will look for a 'model.toml' or 'model.yaml' file in the current "
        " directory."
    ),
)
@click.option(
    "--training_config",
    default=None,
    help=(
        "Path to the training config file. If not provided, pytorch_retrieve "
        " will look for a 'training.toml' or 'training.yaml' file in the current "
        " directory."
    ),
)
def cli(
    model_path: Path,
    model_config: Path,
    training_config: Path,
) -> int:
    """
    Performs EDA on training and validation data.

    The EDA iterates over the training data for two epochs. During the
    first epoch, basic summary statistics of the input data is collected
    (mean, covariance, minimum, maximum). During the second epoch
    histograms of all input features are calculated.

    If successful, the EDA stores the calculated statistics as NetCDF files
    in folder named 'stats' in the model directory.

    Returns: A zero exit code is returned if the EDA was successful. Otherwise
    a non-zero exit code is returned.

    """
    if model_path is None:
        model_path = Path(".")

    model_config = read_model_config(LOGGER, model_path, model_config)
    if model_config is None:
        return 1

    training_config = read_training_config(LOGGER, model_path, training_config)
    if training_config is None:
        return 1

    input_configs = {
        name: InputConfig.parse(name, cfg)
        for name, cfg in model_config["input"].items()
    }

    training_schedule = parse_training_config(training_config)
    run_eda(model_path, input_configs, training_schedule)
