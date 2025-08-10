"""
Tests for the pytorch_retrieve.utils module.
"""
import logging
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_retrieve.utils import (
    read_model_config,
    read_training_config,
    read_compute_config,
    find_most_recent_checkpoint,
    update_recursive,
    InterleaveDatasets,
    BestScoreCheckpoint
)


def test_read_model_config(tmp_path):
    """
    Test reading of model config file and ensure that:
        - None is returned if no file is present.
        - File with default name 'model.toml' is found if model_path is given.
        - File with special name 'model_cfg.toml' is found in working directory
          if model_config is given.

    """
    LOGGER = logging.getLogger(__name__)
    cfg = read_model_config(LOGGER, tmp_path, None)
    assert cfg is None

    with open(tmp_path / "model.toml", "w") as cfg_file:
        cfg_file.write(
            """
            [input.x]
            n_features = 32
            scale = 8
            """
        )
    cfg = read_model_config(LOGGER, tmp_path, None)
    assert cfg is not None
    assert cfg["input"]["x"]["n_features"] == 32

    os.chdir(tmp_path)
    cfg = read_model_config(LOGGER, None, None)
    assert cfg is not None
    assert cfg["input"]["x"]["n_features"] == 32

    with open(tmp_path / "model_cfg.toml", "w") as cfg_file:
        cfg_file.write(
            """
            [input.x]
            n_features = 32
            scale = 8
            """
        )
    cfg = read_model_config(LOGGER, None, "model_cfg.toml")
    assert cfg is not None
    assert cfg["input"]["x"]["n_features"] == 32


def test_read_training_config(tmp_path):
    """
    Test reading of training config file and ensure that:
        - None is returned if no file is present.
        - File with default name 'training.toml' is found if model_path is given.
        - File with special name 'training_cfg.toml' is found in working directory
          if training_config is given.

    """
    LOGGER = logging.getLogger(__name__)
    cfg = read_training_config(LOGGER, tmp_path, None)
    assert cfg is None

    with open(tmp_path / "training.toml", "w") as cfg_file:
        cfg_file.write(
            """
            ['stage 1']
            n_epochs = 10
            """
        )
    cfg = read_training_config(LOGGER, tmp_path, None)
    assert cfg is not None
    assert cfg["stage 1"]["n_epochs"] == 10

    os.chdir(tmp_path)
    cfg = read_training_config(LOGGER, None, None)
    assert cfg is not None
    assert cfg["stage 1"]["n_epochs"] == 10

    with open(tmp_path / "training_cfg.toml", "w") as cfg_file:
        cfg_file.write(
            """
            ['stage 1']
            n_epochs = 10
            """
        )
    cfg = read_training_config(LOGGER, None, "training_cfg.toml")
    assert cfg is not None
    assert cfg["stage 1"]["n_epochs"] == 10


def test_read_compute_config(tmp_path):
    """
    Test reading of compute config file and ensure that:
        - Config is returned even if no file is present.
        - File with default name 'compute.toml' is found if model_path is given.
        - File with special name 'compute_cfg.toml' is found in working directory
          if training_config is given.

    """
    LOGGER = logging.getLogger(__name__)
    cfg = read_compute_config(LOGGER, tmp_path, None)
    assert cfg is not None

    with open(tmp_path / "compute.toml", "w") as cfg_file:
        cfg_file.write(
            """
            strategy = "ddp"
            """
        )
    cfg = read_compute_config(LOGGER, tmp_path, None)
    assert cfg is not None
    assert cfg["strategy"] == "ddp"

    os.chdir(tmp_path)
    cfg = read_compute_config(LOGGER, None, None)

    with open(tmp_path / "compute_cfg.toml", "w") as cfg_file:
        cfg_file.write(
            """
            strategy = "ddp"
            """
        )
    cfg = read_compute_config(LOGGER, None, "compute_cfg.toml")
    assert cfg is not None
    assert cfg["strategy"] == "ddp"


def test_find_most_recent_checkpoint(tmp_path):
    """
    Test search for most recent checkpoint and ensure that:
        - None is returned when no checkpoint is found
        - A checkpoint is found when one is available.
        - The checkpoint file with the highest version is returned when
          multiple checkpoint files are present.
    """
    ckpt = find_most_recent_checkpoint(tmp_path, "retrieval_model")
    assert ckpt is None

    ckpt_path = tmp_path / "checkpoints"
    ckpt_path.mkdir()
    (ckpt_path / "retrieval_model.ckpt").touch()
    ckpt = find_most_recent_checkpoint(ckpt_path, "retrieval_model")
    assert ckpt is not None

    (ckpt_path / "retrieval_model-v1.ckpt").touch()
    ckpt = find_most_recent_checkpoint(ckpt_path, "retrieval_model")
    assert ckpt.name == "retrieval_model-v1.ckpt"


def test_update_recursive():
    """
    Ensure that update_recursive:
      - Correctly replaces keys in 'dest' that are in 'src'.
      - Includes keys present in 'src' but not in 'dest'.

    """

    dest = {"a": 0, "b": 1}
    src = {
        "b": 2,
        "c": 3,
    }
    res = update_recursive(dest, src)
    assert res["b"] == 2
    assert "c" in res

    dest = {"a": 0, "b": 1, "c": {"a": 0}}
    src = {
        "b": 2,
        "c": 3,
    }
    res = update_recursive(dest, src)
    assert res["c"] == 3

    src = {
        "b": 2,
        "c": {"a": 1},
    }
    res = update_recursive(dest, src)
    assert res["c"]["a"] == 1


def test_inverleave_datasets():
    """
    Ensure that interleave datasets dataset works as expected.
    """
    x = torch.arange(0, 100, 2)
    ds_1 = TensorDataset(x, x)
    y = torch.arange(1, 100, 2)
    ds_2 = TensorDataset(y, y)
    dataset = InterleaveDatasets([ds_1, ds_2])
    data_loader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=dataset.collate_fn,
        drop_last=True
    )

    for batch in data_loader:
        x, y = batch
        assert x.shape[0] == 16
        assert ((x[::2] % 2) == 0).all()
        assert ((x[1::2] % 2) == 1).all()


def test_best_checkpoint(tmp_path):
    """
    Ensure that the BestScoreCheckpoint correctly parses the score from exsisting
    checkpoints.
    """

    ckpt = (tmp_path / "checkpoints").mkdir()
    ckpt = (tmp_path / "checkpoints" / "model_best_val_0.1.ckpt").touch()
    callback = BestScoreCheckpoint(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        prefix="model",
    )
    callback.find_best_score()
    assert callback.best_score == 0.1


    # Ensure that
