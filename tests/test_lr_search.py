"""
Tests for the pytorch_retrieve.lr_search module.
"""
from pytorch_retrieve.lightning import LightningRetrieval
from pytorch_retrieve.lr_search import run_lr_search


def test_lr_search(mlp, mlp_training_schedule, cpu_compute_config):
    """
    Ensure that running the lr search works as expected.
    """
    mlp_module = LightningRetrieval(mlp, training_schedule=mlp_training_schedule)
    run_lr_search(
        mlp_module,
        n_steps=20,
        plot=False,
        compute_config=cpu_compute_config
    )
