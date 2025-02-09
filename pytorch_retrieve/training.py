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
from typing import Any, Dict, List, Optional, Tuple, Union
import sys

import click
import lightning as L
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, IterableDataset, Subset, random_split
from torch.optim.lr_scheduler import SequentialLR
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
    find_most_recent_checkpoint,
)
from pytorch_retrieve.lightning import LightningRetrieval


LOGGER = logging.getLogger(__name__)


def load_weights(path: Path, model: nn.Module) -> None:
    """
    Load model weights from existing model file.

    This function first removes mis-matching tensors from the state dict of the
    model and the tries loads the tensors from the state dict in non-strict manner,
    i.e., not requiring all keys to be present..

    Args:
        path: A path pointing to the model file or checkpoint containing the weights to load.
        model: The pytorch Module object into which to load the pre-trained weights.
    """
    path = Path(path)
    if path.exists():
        data = torch.load(path, map_location="cpu")
        state = data["state_dict"]
        if path.suffix == ".ckpt":
            state = {key[6:]: val for key, val in state.items()}

        model_state = model.state_dict()
        matched_state = {}
        mismatch = []
        ignored = []
        for key, tensor in state.items():
            if key in model_state:
                if not isinstance(tensor, torch.Tensor):
                    continue

                if model_state[key].shape == tensor.shape:
                    matched_state[key] = tensor
                else:
                    mismatch.append(key)
            else:
                ignored.append(key)

        model.load_state_dict(matched_state, strict=False)
        if len(mismatch) > 0:
            LOGGER.warning(
                "The following layers loaded from the model at %s were discarded "
                "due to shape mis-match: %s",
                path,
                mismatch,
            )
        if len(ignored) > 0:
            LOGGER.warning(
                "The following layers loaded from the model at %s were ignored "
                "because the current model contains no matching layer: %s",
                path,
                ignored,
            )
    else:
        LOGGER.error(
            "Path provided as 'load_weights' argument does not point "
            "to an existing file."
        )


def freeze_modules(model: nn.Module, freeze: List[str]) -> None:
    """
    Freeze list of modules in model.

    Args:
        model: The model whose modules to freeze
        freeze: A list containing the names of the modules to freeze.
    """
    modules = dict(model.named_modules())
    for name in freeze:
        if name in modules:
            for param in modules[name].parameters():
                param.requires_grad = False
        else:
            LOGGER.warning(
                "Couldn't freeze module '%s' because it isn't a named module of the model."
            )


class TrainingConfigBase:
    """
    Base functionality for training configuration objects.
    """

    def get_training_and_validation_splits(self) -> Tuple[Subset, Subset]:
        """
        Get training and validation datasets by splitting the training dataset using
        the given validation split.

        Return:
             A tuple ``(train, val)`` containing the training and validation datasets.
        """
        try:
            if self.dataset_module.startswith("."):
                sys.path.append(".")
                dataset_module = self.dataset_module[1:]
            else:
                dataset_module = self.dataset_module
            module = importlib.import_module(dataset_module)
            dataset_class = getattr(module, self.training_dataset)
            dataset = dataset_class(**self.training_dataset_args)
        except ImportError:
            raise RuntimeError(
                "An error was encountered when trying to import the dataset "
                f" module '{self.dataset_module}'. Please make sure that the "
                " provided dataset is actually importable."
            )
        generator = torch.Generator().manual_seed(42)
        train, val = random_split(
            dataset, [1.0 - self.validation_split, self.validation_split]
        )
        return train, val

    def get_training_dataset(self):
        """
        Imports and instantiates the training dataset class using the provided
        training_dataset_args.
        """
        if self.validation_dataset_args is None and self.validation_split is not None:
            return self.get_training_and_validation_splits()[0]

        try:
            if self.dataset_module.startswith("."):
                sys.path.append(".")
                dataset_module = self.dataset_module[1:]
            else:
                dataset_module = self.dataset_module
            module = importlib.import_module(dataset_module)
            dataset_class = getattr(module, self.training_dataset)
            dataset = dataset_class(**self.training_dataset_args)
        except ImportError:
            raise RuntimeError(
                "An error was encountered when trying to import the dataset "
                f" module '{self.dataset_module}'. Please make sure that the "
                " provided dataset is actually importable."
            )
        return dataset

    def get_training_data_loader(self) -> DataLoader:
        """
        Returns the training data loader for the training.
        """
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
            persistent_workers=self.persistent_workers
        )
        return data_loader

    def get_validation_dataset(self) -> Union[Dataset, None]:
        """
        If 'validation_dataset_args' is not None, this method instantiates the dataset module
        with those arguments and returns the resulting dataset. If this is not the case, but
        'validation_split' is not None, the validation dataset is created as a random split from
        the training dataset. Otherwise None is returned.
        """
        if self.validation_dataset_args is None:
            if self.validation_split is None:
                return None
            return self.get_training_and_validation_splits()[1]
        try:
            if self.dataset_module.startswith("."):
                sys.path.append(".")
                dataset_module = self.dataset_module[1:]
            else:
                dataset_module = self.dataset_module
            module = importlib.import_module(dataset_module)
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
            num_workers=self.n_data_loader_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
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
                    f"Training stage '{name}' has 'reuse_optimizer' "
                    "set to 'True' but no previous optimizer is available."
                )
            optimizer = previous_optimizer
            if "lr" in self.optimizer_args:
                lr = self.optimizer_args["lr"]
                for group in optimizer.param_groups:
                    group["lr"] = lr

        else:
            optimizer_cls = getattr(torch.optim, self.optimizer)
            optimizer = optimizer_cls(model.parameters(), **self.optimizer_args)

        scheduler = self.scheduler
        if scheduler is None:
            return optimizer, None

        if isinstance(scheduler, list):
            milestones = self.milestones
            if milestones is None:
                raise RuntimeError(
                    "If a list of schedulers is provided, 'milestones' must be "
                    "provided as well."
                )
            schedulers = scheduler
            scheds = []
            for scheduler, args in zip(schedulers, self.scheduler_args):
                scheduler = getattr(torch.optim.lr_scheduler, scheduler)
                scheduler_args = self.scheduler_args
                if scheduler_args is None:
                    scheduler_args = {}
                scheds.append(
                    scheduler(
                        optimizer=optimizer,
                        **args,
                    )
                )
            scheduler = SequentialLR(
                optimizer, schedulers=scheds, milestones=self.milestones
            )
            scheduler.stepwise = self.stepwise_scheduling
            return optimizer, scheduler

        if scheduler == "Warmup":
            total_iters = self.scheduler_args.get("n_iterations", self.n_epochs - 1)
            stepwise = False
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=total_iters
            )
            return optimizer, scheduler

        if scheduler == "ReduceLROnPlateau":
            print("OPTIM :: ", optimizer)
            monitor = self.scheduler_args.pop("monitor", "Validation loss")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **self.scheduler_args
            )
            lr_scheduler = {"scheduler": scheduler, "strict": True, "monitor": monitor}
            return optimizer, scheduler

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
        checkpoint_cb = callbacks.ModelCheckpoint(
            dirpath=module.model_dir / "checkpoints",
            filename=module.name,
            save_top_k=0,
            save_last=True,
        )
        checkpoint_cb.CHECKPOINT_NAME_LAST = module.name

        cbs = [checkpoint_cb]
        if self.minimum_lr is not None:
            cbs.append(
                callbacks.EarlyStopping(
                    monitor="Learning rate",
                    stopping_threshold=self.minimum_lr,
                    patience=self.n_epochs,
                    strict=True
                )
            )
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
    validation_split: Optional[float]

    n_epochs: int
    batch_size: int
    optimizer: str
    optimizer_args: Optional[dict] = None
    scheduler: str = None
    scheduler_args: Optional[dict] = None
    milestones: Optional[List[int]] = None
    minimum_lr: Optional[float] = None
    reuse_optimizer: bool = False
    stepwise_scheduling: bool = False
    metrics: Optional[Dict[str, List["Metric"]]] = None

    log_every_n_steps: Optional[int] = None
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    accumulate_grad_batches: int = 1
    load_weights: Optional[str] = None
    n_data_loader_workers: int = 12
    persistent_workers: bool = True
    freeze: Optional[List[str]] = None
    debug: bool = False

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
        validation_split = get_config_attr(
            "validation_split",
            None,
            config_dict,
            f"training stage {name}",
            required=False,
            default=None,
        )

        n_epochs = get_config_attr(
            "n_epochs", int, config_dict, f"training stage {name}", required=True
        )
        batch_size = get_config_attr(
            "batch_size", int, config_dict, f"training stage {name}", 8, required=True
        )
        if batch_size == 0:
            batch_size = None

        optimizer = get_config_attr(
            "optimizer", str, config_dict, f"training stage {name}", "AdamW"
        )
        optimizer_args = get_config_attr(
            "optimizer_args", dict, config_dict, f"training stage {name}", {"lr": 1e-3}
        )

        scheduler = get_config_attr(
            "scheduler", None, config_dict, f"training stage {name}", None
        )
        scheduler_args = get_config_attr(
            "scheduler_args", None, config_dict, f"training stage {name}", {}
        )
        milestones = get_config_attr(
            "milestones", list, config_dict, f"training stage {name}", None
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
        gradient_clip_algorithm = get_config_attr(
            "gradient_clip_algorithm", str, config_dict, f"training stage {name}", None
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
        persistent_workers = get_config_attr(
            "persistent_workers", bool, config_dict, f"training stage {name}", True
        )
        freeze = get_config_attr(
            "freeze", list, config_dict, f"training state {name}", []
        )
        debug = get_config_attr(
            "debug", bool, config_dict, f"training stage {name}", False
        )

        return TrainingConfig(
            training_dataset=training_dataset,
            dataset_module=dataset_module,
            training_dataset_args=training_dataset_args,
            validation_dataset=validation_dataset,
            validation_dataset_args=validation_dataset_args,
            validation_split=validation_split,
            n_epochs=n_epochs,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            milestones=milestones,
            batch_size=batch_size,
            minimum_lr=minimum_lr,
            reuse_optimizer=reuse_optimizer,
            stepwise_scheduling=stepwise_scheduling,
            metrics=metrics,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            accumulate_grad_batches=accumulate_grad_batches,
            load_weights=load_weights,
            n_data_loader_workers=n_data_loader_workers,
            persistent_workers=persistent_workers,
            freeze=freeze,
            debug=debug,
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
) -> Path:
    """
    Train a pytorch_retrieve retrieval module.

    Args:
        model_dir: A path object pointing to the directory to use to store
            training artifacts.
        module: The retrieval module to train.
        compute_config: A ComputeConfig object specfiying the compute
            configuration for the training.
        checkpoint: An optional path object pointing to a checkpoint path
            from which to continue the training.

    Return:
        A path object pointing to the file to which the trained model was
        written.
    """
    if model_dir is None:
        model_dir = Path(".")
    if compute_config is None:
        compute_config = ComputeConfig()

    if checkpoint is not None:
        LOGGER.info("Resuming training from checkpoint %s.", checkpoint)
        module = LightningRetrieval.load_from_checkpoint(
            checkpoint,
            model=module.model,
            training_schedule=module.training_schedule,
            model_dir=module.model_dir,
            logger=module.logger_class,
        )

    while not module.training_finished:
        training_config = module.current_training_config

        # Try to load weights, if 'load_weight' argument is set.
        if training_config.load_weights is not None:
            load_weights(training_config.load_weights, module.model)

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
            gradient_clip_algorithm=training_config.gradient_clip_algorithm,
            detect_anomaly=False,
        )


        #for name, mod in module.named_modules():
        #    def make_hook(name):
        #        def detect_non_finite_with_exception(module, input, output):
        #            if isinstance(output, torch.Tensor) and not torch.isfinite(output).all():
        #                raise ValueError(f"Non-finite value detected in output of layer: {name, module, output.shape}")
        #        return detect_non_finite_with_exception
        #    mod.register_forward_hook(make_hook(name))

        trainer.fit(
            module,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
            ckpt_path=checkpoint,
        )
        module = module.to(torch.device("cpu"))
        model_path = module.save_model(model_dir)
        checkpoint = None
    return model_path


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
    is_flag=True,
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
    module_name = None
    if "name" in model_config:
        module_name = model_config["name"]

    if model_config is None:
        return 1
    retrieval_model = compile_architecture(model_config)

    training_config = read_training_config(LOGGER, model_path, training_config)
    if training_config is None:
        return 1

    training_schedule = parse_training_config(training_config)

    module = LightningRetrieval(
        retrieval_model, training_schedule=training_schedule, name=module_name
    )

    compute_config = read_compute_config(LOGGER, model_path, compute_config)
    if compute_config is not None:
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
