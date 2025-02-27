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
from pytorch_retrieve.modules.stats import save_stats
from pytorch_retrieve.config import (
    read_config_file,
    InputConfig,
    OutputConfig,
    ComputeConfig,
)
from pytorch_retrieve.training import parse_training_config, TrainingConfig
from pytorch_retrieve.utils import (
    read_model_config,
    read_training_config,
    read_compute_config,
)

LOGGER = logging.getLogger(__name__)


class EDAModule(L.LightningModule):
    """
    Lightning module for performing EDA on training data.

    The main purpose of the lightning module is to piggy-back the distributed
    data loading provided by pytorch to speed up the EDA.
    """

    def __init__(
        self,
        input_configs: Dict[str, InputConfig],
        output_configs: Dict[str, OutputConfig],
        stats_path: Path,
    ):
        """
        Args:
            input_configs: Dictionary containing the input configurations for all
                retrieval inputs.
            output_configs: Dictionary containing the output configurations for all
                retrieval outputs.
            stats_path: The directory where the retrieval artifacts will
                stored.
        """
        super().__init__()
        stats_path = Path(stats_path)
        self.stats_path = stats_path

        self.input_modules = nn.ModuleDict(
            {
                name: InputLayer(name, cfg.n_features)
                for name, cfg in input_configs.items()
            }
        )
        self.output_modules = nn.ModuleDict(
            {name: cfg.get_output_layer() for name, cfg in output_configs.items()}
        )
        self.params = nn.Parameter(torch.zeros(1), requires_grad=True)

        for mod in self.input_modules.values():
            mod.reset()
        for mod in self.output_modules.values():
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
            mod.track_stats(inputs[name])

        for name, mod in self.output_modules.items():
            if not isinstance(outputs, dict):
                outputs = {name: outputs}
            mod.track_stats(outputs[name])

        return None

    def on_train_epoch_end(self):
        """
        Signal the end of the epoch to all input modules.
        """
        for name, mod in self.input_modules.items():
            mod.epoch_finished()
        for name, mod in self.output_modules.items():
            mod.epoch_finished()

    def on_fit_end(self):
        """
        Save statistics after two passes through the data.
        """
        for name, mod in self.input_modules.items():
            stats = mod.compute_stats(self)
            save_stats(stats, self.stats_path / "input", name)
        for name, mod in self.output_modules.items():
            stats = mod.compute_stats(self)
            save_stats(stats, self.stats_path / "output", name)


def run_eda(
    stats_path: Path,
    input_configs: Dict[str, InputConfig],
    output_configs: Dict[str, OutputConfig],
    training_config: TrainingConfig,
    compute_config: Optional[ComputeConfig] = None,
) -> None:
    """
    Performs EDA for given input and training configs.

    Args:
        stats_path: The directory to which to write the calculated statistics.
        input_configs: A dictionary mapping input names to corresponding input
            configurations.
        output_configs: A dictionary mapping retrieval output names to corresponding
            config objects.
        training_config: A TrainingConfig object defining the training settings
            to use for the EDA.
        compute_config: A ComputeConfig object defining the configuration of the
            compute environment to perform the EDA on.
    """
    training_loader = training_config.get_training_data_loader()

    if compute_config is None:
        compute_config = ComputeConfig()

    eda_module = EDAModule(input_configs, output_configs, stats_path)
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
    "--stats_path",
    default=None,
    help=(
        "Directory to which to write the resulting statistics files. If not "
        "set, they will be written to directory named 'stats' in the model "
        "path. "
    ),
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
    "--compute_config",
    default=None,
    help=(
        "Path to the compute config file defining the compute environment for "
        " the training."
    ),
)
@click.option(
    "--stage",
    default=None,
    help=(
        "If provided, training settings for the EDA will be loaded from this "
        "stage of the training schedule."
    ),
)
def cli(
    model_path: Path,
    stats_path: Path,
    model_config: Path,
    training_config: Path,
    compute_config: Optional[Path],
    stage: str,
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
    else:
        model_path = Path(model_path)

    if stats_path is None:
        stats_path = model_path / "stats"

    model_config = read_model_config(LOGGER, model_path, model_config)
    if model_config is None:
        return 1

    training_config = read_training_config(LOGGER, model_path, training_config)
    if training_config is None:
        return 1

    compute_config = read_compute_config(LOGGER, model_path, compute_config)
    if isinstance(compute_config, dict):
        compute_config = ComputeConfig.parse(compute_config)

    input_configs = {
        name: InputConfig.parse(name, cfg)
        for name, cfg in model_config.get("input", {}).items()
    }

    # Ensure that stats for meta data input are recorded.
    if "meta_data" in model_config:
        input_configs.update(
            {
                name: InputConfig.parse(name, cfg)
                for name, cfg in model_config["meta_data"].items()
            }
        )

    output_configs = {
        name: OutputConfig.parse(name, cfg)
        for name, cfg in model_config["output"].items()
    }
    training_schedule = parse_training_config(training_config)
    if stage is None:
        training_config = next(iter(training_schedule.values()))
    else:
        if stage not in training_schedule:
            LOGGER.error(
                "The given stage '%s' is not a stage in the provided training "
                "schedule.",
                stage,
            )
            return 1
        training_config = training_schedule[stage]

    run_eda(
        stats_path,
        input_configs,
        output_configs,
        training_config,
        compute_config=compute_config,
    )
