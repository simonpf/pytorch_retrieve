"""
pytorch_retrieve.eda
====================

Implements functionality to perform explorative data analysis extracting
basic statistics from datasets.
"""
import logging
from pathlib import Path
from typing import Dict

import click

from pytorch_retrieve.modules.input import InputLayer
from pytorch_retrieve.config import read_config_file, InputConfig
from pytorch_retrieve.training import parse_training_config, TrainingConfig
from pytorch_retrieve.utils import read_model_config, read_training_config

LOGGER = logging.getLogger(__name__)


def run_eda(
    model_directory: Path,
    input_configs: Dict[str, InputConfig],
    training_config: TrainingConfig
) -> None:
    """
    Performs EDA for given input and training configs.

    Args:
        model_directory: The directory in which to store the computed statistics.
        input_configs: A dictionary mapping input names to corresponding input
            configurations.
        training_config: A TrainingConfig object defining the training settings
            to use for the EDA.
    """
    training_loader = training_config.get_training_data_loader()
    validation_loader = training_config.get_validation_data_loader()

    input_modules = {
        name: InputLayer(name, cfg.n_features, model_path=model_directory)
        for name, cfg in input_configs.items()
    }
    for mod in input_modules.values():
        mod.reset()

    # First epoch
    for x, y in training_loader:
        for name, mod in input_modules.items():
            if not isinstance(x, dict):
                x = {name: x}
            mod(x[name])
    for mod in input_modules.values():
        mod.epoch_finished()

    # Second epoch
    for x, y in training_loader:
        for name, mod in input_modules.items():
            if not isinstance(x, dict):
                x = {name: x}
            mod(x[name])
    for mod in input_modules.values():
        mod.epoch_finished()


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
@click.option(
    "--stage",
    default=None,
    help=(
        "If provided, training settings for the EDA will be loaded from this "
        "stage of the training schedule."
    )
)
def cli(
    model_path: Path,
    model_config: Path,
    training_config: Path,
    stage: str
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
    if stage is None:
        training_config = next(iter(training_schedule.values()))
    else:
        if stage not in training_schedule:
            LOGGER.error(
                "The given stage '%s' is not a stage in the provided training "
                "schedule.",
                stage
            )
            return 1
        training_config = training_schedule[stage]

    run_eda(model_path, input_configs, training_config)
