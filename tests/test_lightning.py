"""
Tests for the lightning interface defined in pytorch_retrieve.lightning.
"""
from pytorch_retrieve.lightning import LightningRetrieval
from pytorch_retrieve.training import (
    run_training,
)
from pytorch_retrieve import load_model


def test_save_load_checkpoint(
        mlp,
        mlp_training_schedule,
        tmp_path,
        cpu_compute_config
):
    """
    Test saving and loading of checkpoints.
    """
    lret = LightningRetrieval(
        mlp,
        training_schedule=mlp_training_schedule,
        model_dir=tmp_path
    )
    run_training(tmp_path, lret, compute_config=cpu_compute_config)

    ckpts = list((tmp_path / "checkpoints").glob("*.ckpt"))
    model = load_model(ckpts[0])
    assert model is not None

    ckpts = list((tmp_path / "checkpoints").glob("*best_val*.ckpt"))
    assert len(ckpts) == 1
