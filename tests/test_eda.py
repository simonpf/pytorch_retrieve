"""
Tests for the pytorch_retrieve.eda module.
"""
from pytorch_retrieve.config import InputConfig, OutputConfig
from pytorch_retrieve.eda import run_eda


def test_eda(tmp_path, mlp_training_schedule):
    """
    Ensure that running EDA produces the expected artifacts.
    """
    input_configs = {"x": InputConfig(n_features=1)}
    output_configs = {"y": OutputConfig(target="y", kind="Mean", shape=1)}
    run_eda(tmp_path / "stats", input_configs, output_configs, mlp_training_schedule)

    assert (tmp_path / "stats" / "input" / "x.nc").exists()
    assert (tmp_path / "stats" / "output" / "y.nc").exists()
