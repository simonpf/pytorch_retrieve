"""
Tests for the pytorch_retrieve.eda module.
"""
from pytorch_retrieve.config import InputConfig
from pytorch_retrieve.eda import run_eda


def test_eda(tmp_path, mlp_training_schedule):
    """
    Ensure that running EDA produces the expected artifacts.
    """
    input_configs = {"x": InputConfig(n_features=1)}
    run_eda(tmp_path, input_configs, mlp_training_schedule)
    assert (tmp_path / "stats" / "x.nc").exists()
