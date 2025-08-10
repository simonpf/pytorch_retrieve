"""
pytorch_retrieve.utils
======================

Shared utility functions.
"""
import logging
from itertools import chain
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .config import read_config_file, ComputeConfig


def read_model_config(
    logger: logging.Logger,
    model_path: Optional[Path],
    model_config: Optional[Path],
) -> Dict[str, object]:
    """
    Try to read model config file given an optional model path and config file
    name.

    Read model configuration dict from 'model_config' if not None. Otherwise,
    will look for a .toml or .yaml file in 'model_path' and parse the model
    configuration dict from this file if the file exists. If 'model_path' if not
    given, will look for 'model.toml' or 'model.yaml' file in the current
    working directory.

    Return:
        A retrieval-model configuration dict or None if no model configuration
        file could be found.
    """
    # Parse model config
    if model_config is None:
        if model_path is None:
            model_path = Path(".")

        model_config = list(model_path.glob("model.????"))
        if len(model_config) == 0:
            logger.error(
                "No explicit path to model configuration file provided and "
                " the working directory does not contain any model.???? "
                "file."
            )
            return None
        if len(model_config) > 1:
            logger.error(
                "No explicit path to model configuration file provided and "
                " the working directory contains more than one model.???? "
                "file."
            )
            return None
        model_config = model_config[0]
        if not model_config.suffix in [".toml", ".yaml"]:
            logger.error(
                "Model configuration file should be in '.toml' or '.yaml' " "format."
            )
            return None
    return read_config_file(model_config)


def read_training_config(
    logger: logging.Logger,
    model_path: Optional[Path],
    training_config: Optional[Path],
) -> Dict[str, object]:
    """
    Try to read training config file given an optional training path and config file
    name.

    Read training configuration dict from 'training_config' if not None. Otherwise,
    will look for a 'training.toml' or 'training.yaml' file in 'model_path' and
    parse the training configuration dict from this file if the file exists. If
    'model_path' if not given, will look for 'training.toml' or 'training.yaml' file
    in the current working directory.

    Return:
        A training configuration dict or None if no training configuration
        file could be found.
    """
    # Parse training config
    if training_config is None:
        if model_path is None:
            model_path = Path(".")

        training_config = list(model_path.glob("training.????"))
        if len(training_config) == 0:
            logger.error(
                "No explicit path to training configuration file provided and "
                " the working directory does not contain any 'training.????'."
                "file."
            )
            return None
        if len(training_config) > 1:
            logger.error(
                "No explicit path to training configuration file provided and "
                " the working directory contains more than one training.???? "
                "file."
            )
            return None
        training_config = training_config[0]
        if not training_config.suffix in [".toml", ".yaml"]:
            logger.error(
                "Training configuration file should be in '.toml' or '.yaml' " "format."
            )
            return None
    return read_config_file(training_config)


def read_compute_config(
    logger: logging.Logger,
    model_path: Optional[Path],
    compute_config: Optional[Path],
) -> Dict[str, object]:
    """
    Try to read compute config file given an optional model path and config file
    name.

    Read compute configuration dict from 'model_config' if not None. Otherwise,
    will look for a .toml or .yaml file in 'model_path' and parse the model
    configuration dict from this file if the file exists. If 'model_path' if not
    given, will look for 'model.toml' or 'model.yaml' file in the current
    working directory.

    Return:
        A retrieval-model configuration dict or None if no model configuration
        file could be found.
    """
    if compute_config is None:
        if model_path is None:
            model_path = Path(".")

        compute_config = list(model_path.glob("compute.????"))

        if len(compute_config) == 0:
            return {}

        if len(compute_config) > 1:
            logger.error(
                "No explicit path to compute configuration file provided and "
                " the working directory contains more than one compute.???? "
                "files."
            )
            return None
        compute_config = compute_config[0]
        if not compute_config.suffix in [".toml", ".yaml"]:
            logger.error(
                "Compute configuration file should be in '.toml' or '.yaml' " "format."
            )
            return None

    return read_config_file(compute_config)


def find_most_recent_checkpoint(path: Path, model_name: str) -> Path:
    """
    Find most recente Pytorch lightning checkpoint files.

    Args:
        path: A pathlib.Path object pointing to the folder containing the
            checkpoints.
        model_name: The model name as defined by the user.

    Return:
        If a checkpoint was found, returns a object pointing to the
        checkpoint file with the highest version number. Otherwise
        returns 'None'.
    """
    path = Path(path)

    checkpoint_files = list(path.glob(f"{model_name}*.ckpt"))
    if len(checkpoint_files) == 0:
        return None
    if len(checkpoint_files) == 1:
        return checkpoint_files[0]

    checkpoint_regexp = re.compile(rf"{model_name}(-v\d*)?.ckpt")
    versions = []
    for checkpoint_file in checkpoint_files:
        match = checkpoint_regexp.match(checkpoint_file.name)
        if match is None:
            return None
        if match.group(1) is None:
            versions.append(-1)
        else:
            versions.append(int(match.group(1)[2:]))
    ind = np.argmax(versions)
    return checkpoint_files[ind]


def update_recursive(dest: Dict[Any, Any], src: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively update content in dictionary.

    Args:
        dest: The dictionary to update.
        src: The keys to update.

    Return:
        An updated dictionary
    """
    res = {key: value for key, value in src.items() if key not in dest}
    for key, value in dest.items():
        if key in src:
            if isinstance(value, dict) and isinstance(src[key], dict):
                res[key] = update_recursive(dest[key], src[key])
            else:
                res[key] = src[key]
        else:
            res[key] = value
    return res


class InterleaveDatasets(Dataset):
    """
    This special dataset class combines two datasets be interleaving them. This doubles the effective
    batch size of the data but ensures that each batch contains an even number of samples from each
    dataset.
    """
    def __init__(
            self,
            datasets: List[Dataset]
    ):
        """
        Args:
            datasets: List of the datasets to interleave.
        """
        self.datasets = datasets
        seed = int.from_bytes(os.urandom(4), "big")
        self.rng = np.random.default_rng(seed)

    def collate_fn(self, batch):
        """
        Special collate function that handles the nested samples returned from the dataset.

        """
        full_batch = list(chain.from_iterable(batch))
        return default_collate(full_batch)

    def __len__(self) -> int:
        """
        The number of samples in the dataset is the maximum length of the two datasets.
        """
        return len(self.datasets[0])

    def __getitem__(self, ind: int) -> List[Any]:
        """
        Returns list containing a sample from each dataset.

        Args:
            ind: Index used to select the sample from the first dataset.
        """
        fill_values_target = {}
        fill_values_input = {}
        samples = []

        inpt, target = self.datasets[0][ind]

        if isinstance(inpt, dict):
            for name, tnsr in inpt.items():
                if isinstance(tnsr, list):
                    fill_values_input[name] = [torch.nan * torch.zeros_like(elem) for elem in tnsr]
                else:
                    fill_values_input[name] = torch.nan * torch.zeros_like(tnsr)
        if isinstance(target, dict):
            for name, tnsr in target.items():
                if isinstance(tnsr, list):
                    fill_values_target[name] = [torch.nan * torch.zeros_like(elem) for elem in tnsr]
                else:
                    fill_values_target[name] = torch.nan * torch.zeros_like(tnsr)

        samples.append((inpt, target))

        for dataset in self.datasets[1:]:
            ind = self.rng.integers(0, len(dataset))
            inpt, target = dataset[ind]

            if isinstance(inpt, dict):
                for name, tnsr in inpt.items():
                    if isinstance(tnsr, list):
                        fill_values_input[name] = [torch.nan * torch.zeros_like(elem) for elem in tnsr]
                    else:
                        fill_values_input[name] = torch.nan * torch.zeros_like(tnsr)

            if isinstance(target, dict):
                for name, tnsr in target.items():
                    if isinstance(tnsr, list):
                        fill_values_target[name] = [torch.nan * torch.zeros_like(elem) for elem in tnsr]
                    else:
                        fill_values_target[name] = torch.nan * torch.zeros_like(tnsr)

            samples.append((inpt, target))

        for smpl in samples:
            inpt, target = smpl

            if isinstance(inpt, dict):
                for name, tnsr in fill_values_input.items():
                    if name not in inpt:
                        inpt[name] = tnsr

            if isinstance(target, dict):
                for name, tnsr in fill_values_target.items():
                    if name not in target:
                        target[name] = tnsr

        return samples


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Scheduler that linearly increases the learning rate.
    """
    def __init__(self, optimizer, warmup_ratio=0.1, total_steps=None, last_epoch=-1):
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps
        self.scheduler = None  # Will be created later
        self.optimizer = optimizer
        self._initialized = False
        super().__init__(optimizer, last_epoch)

    def _init_scheduler(self):
        if self.total_steps is None:
            raise ValueError("total_steps must be set before stepping the scheduler.")

        warmup_steps = int(self.total_steps * self.warmup_ratio)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(self.total_steps - current_step) / float(max(1, self.total_steps - warmup_steps))
            )

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self._initialized = True

    def step(self):
        if not self._initialized:
            self._init_scheduler()
        self.scheduler.step()

    def get_last_lr(self):
        if not self._initialized:
            return [group['lr'] for group in self.optimizer.param_groups]
        return self.scheduler.get_last_lr()

class WarmupLR(LRScheduler):
    """
    Special scheduler for warming up the learning rate.

    This is mostly copied from PyTorch's LinearLR scheduler. It is redefined here to allow identifying
    the scheduler from the lightning module and adaptively setting the number of steps expected during
    training.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 0.1,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            start_factor (float): The number we multiply learning rate in the first epoch.
                The multiplication factor changes towards end_factor in the following epochs.
                Default: 1./3.
            end_factor (float): The number we multiply learning rate at the end of linear changing
                process. Default: 1.0.
            total_iters (int): The number of iterations that multiplicative factor reaches to 1.
                Default: 5.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(
                "Starting multiplicative factor expected to be greater than 0 and less or equal to 1."
            )

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate."""
        if self.total_iters == -1:
            LOGGER.warning(
                "Could not infer number of steps in epoch for Warmup schedulear. Will linearly increase learning rate "
                "over 1000 steps."
            )
            self.total_iters = 1_000

        if self.last_epoch == 0:
            return [
                group["lr"] * self.start_factor for group in self.optimizer.param_groups
            ]

        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]


import os
import re
from typing import Optional, Tuple
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class BestScoreCheckpoint(Callback):
    """
    Saves a checkpoint of the model with the best validation loss, falling back
    to the training loss if it is not available.


    """
    def __init__(
        self,
        checkpoint_dir: str,
        prefix: str,
        primary: str = "Validation loss",
        fallback: str = "Training loss",
        mode: str = "min",
        score_fmt: str = "{score:.6f}",
    ):
        Path(checkpoint_dir).mkdir(exist_ok=True)
        assert mode in ("min", "max")
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix
        self.primary = primary
        self.fallback = fallback
        self.mode = mode
        self.score_fmt = score_fmt

        self.best_score: Optional[float] = None
        self.best_ckpt = None
        self._re = re.compile(
            rf"^{re.escape(prefix + '_best_val')}_"
            r"(?P<score>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?).ckpt"
        )


    def is_better(self, new: float, ref: Optional[float]) -> bool:
        """
        Determine whether current loss is better than the reference value.

        Args:
            new: The new value.
            ref: The reference value.

        Return:
            A boolean value indicating whether new is better than ref.
        """
        if ref is None:
            return True
        return (new < ref) if self.mode == "min" else (new > ref)

    def get_metric(self, trainer) -> Tuple[Optional[float], Optional[str]]:
        """
        Get primary or secondary metric value from trainer.
        """
        mtrcs = trainer.callback_metrics
        if self.primary in mtrcs and mtrcs[self.primary] is not None:
            try:
                return float(mtrcs[self.primary]), self.primary
            except (TypeError, ValueError):
                pass
        if self.fallback in mtrcs and mtrcs[self.fallback] is not None:
            try:
                return float(mtrcs[self.fallback]), self.fallback
            except (TypeError, ValueError):
                pass
        return None, None

    def make_filename(self, score: float) -> str:
        score_str = self.score_fmt.format(score=score)
        return f"{self.prefix}_best_val_{score_str}.ckpt"

    def find_best_score(self):
        """
        Initializes the best score attribute looking for available checkpoints
        in the checkpoint directory setting it to None of no previous checkpoints
        are available.
        """
        try:
            for path in Path(self.checkpoint_dir).glob("*.ckpt"):
                match = self._re.match(path.name)
                if not match:
                    continue
                try:
                    score = float(match.group("score"))
                except ValueError:
                    continue
                self.best_score = score
        except FileNotFoundError:
            Path(self.checkpoint_dir).mkdir(exist_ok=True)

    @rank_zero_only
    def save_if_better(self, trainer, score: float, metric_name: str):
        if not self.is_better(score, self.best_score):
            return None

        self.best_score = score
        fname = self.make_filename(score)
        ckpt_path = Path(self.checkpoint_dir) / fname
        trainer.save_checkpoint(ckpt_path)

        if self.best_ckpt is not None and Path(self.best_ckpt).exists():
            Path(self.best_ckpt).unlink()

        self.best_ckpt = ckpt_path

    @rank_zero_only
    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        self.find_best_score()

    def on_validation_epoch_end(self, trainer, pl_module):
        score, name = self.get_metric(trainer)
        if score is not None:
            self.save_if_better(trainer, score, name)
