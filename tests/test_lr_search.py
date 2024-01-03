"""
Tests for the pytorch_retrieve.lr_search module.
"""
from pytorch_retrieve.lightning import LightningRetrieval
from pytorch_retrieve.lr_search import run_lr_search


def test_lr_search(mlp, mlp_training_schedule):
    """
    Ensure that running the lr search works as expected.
    """
    mlp_module = LightningRetrieval(mlp, "retrieval_module", mlp_training_schedule)
    run_lr_search(mlp_module, n_steps=20, plot=False)
