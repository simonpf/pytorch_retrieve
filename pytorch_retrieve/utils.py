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
        fill_values = {}
        samples = []

        sample = self.datasets[0][ind]
        for target in sample[1:]:
            if isinstance(target, dict):
                for name, tnsr in target.items():
                    fill_values[name] = torch.nan * torch.zeros_like(tnsr)
        samples.append(sample)

        for dataset in self.datasets[1:]:
            ind = self.rng.integers(0, len(dataset))
            sample = dataset[ind]
            for target in sample[1:]:
                if isinstance(target, dict):
                    for name, tnsr in target.items():
                        fill_values[name] = torch.nan * torch.zeros_like(tnsr)
            samples.append(sample)

        for smpl in samples:
            for target in smpl[1:]:
                if isinstance(target, dict):
                    for name, tnsr in fill_values.items():
                        if name not in target:
                            target[name] = tnsr

        return samples
