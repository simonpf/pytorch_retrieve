"""
pytorch_retrieve.lr_search
==========================

Implements learning-rate search for retrieval models.
"""
from pathlib import Path
from typing import Optional

import click
import lightning as L
from lightning.pytorch.tuner import Tuner

from pytorch_retrieve.config import ComputeConfig
from pytorch_retrieve.lightning import LightningRetrieval


def run_lr_search(
    module: LightningRetrieval,
    min_lr: float = 1e-8,
    max_lr: float = 1e2,
    plot: bool = False,
    n_steps: int = 100,
    compute_config: Optional[ComputeConfig] = None,
) -> None:
    """
    Run learning-rate search using the corresponding Lightning functionality.

    Args:
         module: The LightningRetrieval module implementing the retrieval model.
         compute_config: The compute configuration to use for the tuning.
    """
    if compute_config is None:
        compute_config = ComputeConfig()

    training_config = module.current_training_config
    training_loader = training_config.get_training_data_loader()

    module.lr = 1e-5

    trainer = L.Trainer(
        max_epochs=training_config.n_epochs,
        logger=module.current_logger,
        log_every_n_steps=training_config.log_every_n_steps,
        precision=compute_config.precision,
        accelerator=compute_config.accelerator,
        devices=compute_config.devices,
        strategy=compute_config.get_strategy(),
        callbacks=training_config.get_callbacks(module),
    )
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        module,
        train_dataloaders=training_loader,
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=n_steps,
    )

    if plot:
        try:
            import matplotlib.pyplot as plt

            fig = lr_finder.plot(suggest=True)
            plt.show()
        except Exception:
            pass


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
    "--compute_config",
    default=None,
    help=(
        "Path to the compute config file defining the compute environment for "
        " the training."
    ),
)
@click.option(
    "--min_lr",
    default=1e-8,
    help=("The smallest learning rate to test."),
)
@click.option(
    "--max_lr",
    default=1e2,
    help=("The largest learning rate to test."),
)
@click.option(
    "--n_steps",
    default=100,
    help=("The number of training steps to perform."),
)
@click.option(
    "--plot",
    default=True,
    help=("Whether or not to show a plot of the results."),
)
def cli(
    model_path: Optional[Path],
    model_config: Optional[Path],
    training_config: Optional[Path],
    compute_config: Optional[Path],
    min_lr: float,
    max_lr: float,
    n_steps: int,
    plot: bool,
) -> int:
    """
    Determine optimal learning rate.

    This command uses Lightning's built-in learning-rate finder to determine an
    optimal learning rate.
    """
    if model_path is None:
        model_path = Path(".")

    LOGGER = logging.getLogger(__name__)
    model_config = read_model_config(LOGGER, model_path, model_config)
    if model_config is None:
        return 1
    retrieval_model = compile_architecture(model_config)

    training_config = read_training_config(LOGGER, model_path, training_config)
    if training_config is None:
        return 1

    training_schedule = parse_training_config(training_config)

    module = LightningRetrieval(retrieval_model, training_schedule=training_schedule)

    compute_config = read_compute_config(LOGGER, model_path, compute_config)
    if isinstance(compute_config, dict):
        compute_config = ComputeConfig.parse(compute_config)

    run_lr_search(
        module,
        compute_config=compute_config,
        min_lr=min_lr,
        max_lr=max_lr,
        n_steps=n_steps,
        plot=plot,
    )
