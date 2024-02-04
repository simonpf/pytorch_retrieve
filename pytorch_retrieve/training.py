"""
pytorch_retrieve.training
=========================

The 'pytroch_retrieve.training' module coordinates the training of
retrieval models.
"""
from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from lightning.pytorch import callbacks

from pytorch_retrieve import metrics
from pytorch_retrieve.config import (
    read_config_file,
    replace_environment_variables,
    get_config_attr,
    ComputeConfig,
)
from pytorch_retrieve.architectures import compile_architecture
from pytorch_retrieve.utils import (
    read_model_config,
    read_training_config,
    read_compute_config,
)
from pytorch_retrieve.lightning import LightningRetrieval


LOGGER = logging.getLogger(__name__)


class TrainingConfigBase:
    """
    Base functionality for training configuration objects.
    """

    def get_training_dataset(self):
        """
        Imports and instantiates the training dataset class using the provided
        training_dataset_args.
        """
        try:
            module = importlib.import_module(self.dataset_module)
            dataset_class = getattr(module, self.training_dataset)
            return dataset_class(**self.training_dataset_args)
        except ImportError:
            raise RuntimeError(
                "An error was encountered when trying to import the dataset "
                f" module '{self.dataset_module}'. Please make sure that the "
                " provided dataset is actually importable."
            )

    def get_training_data_loader(self):
        dataset = self.get_training_dataset()
        shuffle = not isinstance(dataset, IterableDataset)
        worker_init_fn = None
        if hasattr(dataset, "worker_init_fn"):
            worker_init_fn = dataset.worker_init_fn
        data_loader = DataLoader(
            dataset,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
            batch_size=self.batch_size,
            num_workers=self.n_data_loader_workers,
            pin_memory=True,
        )
        return data_loader

    def get_validation_dataset(self):
        """
        Imports and instantiates the validation dataset class using the provided
        training_dataset_args.
        """
        if self.validation_dataset_args is None:
            return None
        try:
            module = importlib.import_module(self.dataset_module)
            dataset_class = getattr(module, self.training_dataset)
            return dataset_class(**self.validation_dataset_args)
        except ImportError:
            raise RuntimeError(
                "An error was encountered when trying to import the dataset "
                f" module '{self.dataset_module}'. Please make sure that the "
                " provided dataset is actually importable."
            )

    def get_validation_data_loader(self):
        dataset = self.get_validation_dataset()
        if dataset is None:
            return None
        worker_init_fn = None
        if hasattr(dataset, "worker_init_fn"):
            worker_init_fn = dataset.worker_init_fn
        data_loader = DataLoader(
            dataset,
            worker_init_fn=worker_init_fn,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )
        return data_loader

    def get_optimizer_and_scheduler(
        self, name: str, model: nn.Module, previous_optimizer=None
    ):
        """
        Return torch optimizer, learning-rate scheduler and callback objects
        corresponding to this configuration.

        Args:
            name: The name of the training stage.
            model: The retrieval module to be trained.
            previous_optimizer: Optimizer from the previous stage in case
                it is reused.

        Return:
            A tuple ``(optimizer, scheduler, callbacks)`` containing a PyTorch
            optimizer object ``optimizer``, the corresponding LR scheduler
            ``scheduler`` and a list of callbacks.

        Raises:
            Value error if training configuration specifies to reuse the optimizer
            but 'previous_optimizer' is none.

        """
        if self.reuse_optimizer:
            if previous_optimizer is None:
                raise RuntimeError(
                    "Training stage '{self.name}' has 'reuse_optimizer' "
                    "set to 'True' but no previous optimizer is available."
                )
            optimizer = previous_optimizer

        else:
            optimizer_cls = getattr(torch.optim, self.optimizer)
            optimizer = optimizer_cls(model.parameters(), **self.optimizer_args)

        scheduler = self.scheduler
        if scheduler is None:
            return optimizer, None

        scheduler = getattr(torch.optim.lr_scheduler, scheduler)
        scheduler_args = self.scheduler_args
        if scheduler_args is None:
            scheduler_args = {}
        scheduler = scheduler(
            optimizer=optimizer,
            **scheduler_args,
        )
        scheduler.stepwise = self.stepwise_scheduling

        return optimizer, scheduler

    def get_callbacks(self, module: LightningRetrieval) -> List[callbacks.Callback]:
        """
        Get callbacks for training stage.

        Args:
            module: The retrieval module that is being trained.

        Return:
            A list of callbacks for the current stage of the training.

        """
        cbs = [
            callbacks.ModelCheckpoint(
                dirpath=module.model_dir / "checkpoints",
                filename=module.name,
                save_top_k=0,
                save_last=True,
            ),
        ]
        if self.minimum_lr is not None:
            cbs.append(callbacks.EarlyStopping(monitor="Learning rate", strict=True))
        return cbs

    def get_metrics_dict(self, outputs: List[str]) -> Dict[str, Any]:
        """
        Return dict mapping output names to corresponding metrics objects.

        Args:
            outputs: List containing the names of the model outputs.

        Return:
            A dictionary mapping output names to metric objects.
        """
        if self.metrics is None:
            return {}

        mtrcs = {}
        if isinstance(self.metrics, list):
            for output_name in outputs:
                for name in self.metrics:
                    metric = getattr(metrics, name)()
                    mtrcs.setdefault(output_name, []).append(metric)
            return mtrcs

        for output_name, output_metrics in self.metrics.items():
            mtrcs[output_name] = [
                getattr(metrics, metric)() for metric in output_metrics
            ]
        return mtrcs


@dataclass
class TrainingConfig(TrainingConfigBase):
    """
    A dataclass to hold parameters of a single training stage.
    """

    dataset_module: str
    training_dataset: str
    training_dataset_args: Dict[str, object]
    validation_dataset: str
    validation_dataset_args: Optional[Dict[str, object]]

    n_epochs: int
    batch_size: int
    optimizer: str
    optimizer_args: Optional[dict] = None
    scheduler: str = None
    scheduler_args: Optional[dict] = None
    minimum_lr: Optional[float] = None
    reuse_optimizer: bool = False
    stepwise_scheduling: bool = False
    metrics: Optional[Dict[str, List["Metric"]]] = None

    log_every_n_steps: Optional[int] = None
    gradient_clip_val: Optional[float] = None
    accumulate_grad_batches: int = 1
    load_weights: Optional[str] = None
    n_data_loader_workers: int = 12

    @classmethod
    def parse(cls, name, config_dict: Dict[str, object]):
        """
        Parses a single training stage from a dictionary of training settings.

        Args:
            name: Name of the training stage.
            config_dict: The dictionary containing the training settings.

        Return:
            A TrainingConfig object containing the settings from the dictionary.
        """
        dataset_module = get_config_attr(
            "dataset_module",
            str,
            config_dict,
            f"training stage {name}",
        )
        training_dataset = get_config_attr(
            "training_dataset",
            str,
            config_dict,
            f"training stage {name}",
            required=True,
        )
        training_dataset_args = get_config_attr(
            "training_dataset_args", dict, config_dict, f"training stage {name}"
        )
        if training_dataset_args is not None:
            training_dataset_args = {
                name: replace_environment_variables(val)
                for name, val in training_dataset_args.items()
            }
        validation_dataset = get_config_attr(
            "validation_dataset",
            str,
            config_dict,
            f"training stage {name}",
            training_dataset,
        )
        validation_dataset_args = get_config_attr(
            "validation_dataset_args", dict, config_dict, f"training stage {name}", ""
        )
        if validation_dataset_args == "":
            validation_dataset_args = None
        if isinstance(validation_dataset_args, dict):
            validation_dataset_args = {
                name: replace_environment_variables(val)
                for name, val in validation_dataset_args.items()
            }

        n_epochs = get_config_attr(
            "n_epochs", int, config_dict, f"training stage {name}", required=True
        )
        batch_size = get_config_attr(
            "batch_size", int, config_dict, f"training stage {name}", 8, required=True
        )

        optimizer = get_config_attr(
            "optimizer", str, config_dict, f"training stage {name}"
        )
        optimizer_args = get_config_attr(
            "optimizer_args", dict, config_dict, f"training stage {name}", {}
        )

        scheduler = get_config_attr(
            "scheduler", str, config_dict, f"training stage {name}", "none"
        )
        if scheduler == "none":
            scheduler = None
        scheduler_args = get_config_attr(
            "scheduler_args", dict, config_dict, f"training stage {name}", {}
        )

        minimum_lr = get_config_attr(
            "minimum_lr", float, config_dict, f"training stage {name}", -1.0
        )
        if minimum_lr < 0:
            minimum_lr = None

        reuse_optimizer = get_config_attr(
            "reuse_optimizer", bool, config_dict, f"training stage {name}", False
        )
        stepwise_scheduling = get_config_attr(
            "stepwise_scheduling", bool, config_dict, f"training stage {name}", False
        )

        metrics = config_dict.get("metrics", [])

        log_every_n_steps = config_dict.get("log_every_n_steps", -1)
        if log_every_n_steps < 0:
            if n_epochs < 100:
                log_every_n_steps = 1
            else:
                log_every_n_steps = 50

        gradient_clip_val = get_config_attr(
            "gradient_clip_val", float, config_dict, f"training stage {name}", None
        )
        accumulate_grad_batches = get_config_attr(
            "accumulate_grad_batches", int, config_dict, f"training stage {name}", 1
        )
        load_weights = get_config_attr(
            "load_weights", str, config_dict, f"training stage {name}", None
        )
        n_data_loader_workers = get_config_attr(
            "n_data_loader_workers", int, config_dict, f"training stage {name}", 12
        )

        return TrainingConfig(
            training_dataset=training_dataset,
            dataset_module=dataset_module,
            training_dataset_args=training_dataset_args,
            validation_dataset=validation_dataset,
            validation_dataset_args=validation_dataset_args,
            n_epochs=n_epochs,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            batch_size=batch_size,
            minimum_lr=minimum_lr,
            reuse_optimizer=reuse_optimizer,
            stepwise_scheduling=stepwise_scheduling,
            metrics=metrics,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            load_weights=load_weights,
            n_data_loader_workers=n_data_loader_workers
        )


def parse_training_config(config_dict: Dict[str, object]) -> Dict[str, TrainingConfig]:
    """
    Parse training schedule from a training configuration dictionary.

    Args:
        config_dict: A dictionary containing a dictionary representation
             to parse.

    Return:
        A dictionary mapping stage names to corresponding training config
        objects.
    """
    return {name: TrainingConfig.parse(name, dct) for name, dct in config_dict.items()}


def run_training(
    model_dir: Path,
    module: "pytorch_retrieve.lightning.LightningRetrieval",
    compute_config: Optional[ComputeConfig] = None,
    checkpoint: Optional[Path] = None,
) -> None:
    if model_dir is None:
        model_dir = Path(".")
    if compute_config is None:
        compute_config = ComputeConfig()

    if checkpoint is not None:
        module = LightningRetrieval.load_from_checkpoint(
            checkpoint,
            model=module.model,
            training_schedule=module.training_schedule,
            model_dir = module.model_dir,
            logger=module.logger_class
        )

    while not module.training_finished:
        training_config = module.current_training_config

        # Try to load weights, if 'load_weight' argument is set.
        if training_config.load_weights is not None:
            weight_path = Path(training_config.load_weights)
            if weight_path.exists():
                data = torch.load(weight_path)
                module.model.load_state_dict(data["state_dict"])
            else:
                LOGGER.error(
                    "Path provided as 'load_weights' argument does not point "
                    "to an existing file."
                )

        ckpt_path = model_dir / "checkpoints"
        ckpt_path.mkdir(exist_ok=True)

        training_loader = training_config.get_training_data_loader()
        validation_loader = training_config.get_validation_data_loader()

        trainer = L.Trainer(
            max_epochs=training_config.n_epochs,
            logger=module.current_logger,
            log_every_n_steps=training_config.log_every_n_steps,
            precision=compute_config.precision,
            accelerator=compute_config.accelerator,
            devices=compute_config.devices,
            strategy=compute_config.get_strategy(),
            callbacks=training_config.get_callbacks(module),
            accumulate_grad_batches=training_config.accumulate_grad_batches,
            num_sanity_val_steps=0,
            gradient_clip_val=training_config.gradient_clip_val,
        )
        trainer.fit(
            module,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
            ckpt_path=checkpoint
        )
        module = module.to(torch.device("cpu"))
        module.save_model(model_dir)
        checkpoint = None


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
    "--resume",
    "-r",
    "resume",
    default=False,
    help=("If set, training will continue from a checkpoint file if available."),
)
def cli(
    model_path: Optional[Path],
    model_config: Optional[Path],
    training_config: Optional[Path],
    compute_config: Optional[Path],
    resume: bool = False,
) -> int:
    """
    Train retrieval model.

    This command runs the training of the retrieval model specified by the
    model and training configuration files.

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

    module = LightningRetrieval(retrieval_model, "retrieval_module", training_schedule)

    compute_config = read_compute_config(LOGGER, model_path, compute_config)
    if isinstance(compute_config, dict):
        compute_config = ComputeConfig.parse(compute_config)

    checkpoint = None
    if resume:
        checkpoint = find_most_recent_checkpoint(
            model_path / "checkpoints", module.name
        )

    run_training(
        model_path,
        module,
        compute_config=compute_config,
        checkpoint=checkpoint,
    )
