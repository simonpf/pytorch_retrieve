"""
Tests for the pytorch_retrieve.eda module.
"""
from pytorch_retrieve.config import InputConfig, OutputConfig
from pytorch_retrieve.eda import run_eda


def test_eda(
        tmp_path,
        mlp_training_schedule,
        cpu_compute_config,
):
    """
    Ensure that running EDA produces the expected artifacts.
    """
    input_configs = {"x": InputConfig(n_features=1)}
    output_configs = {"y": OutputConfig(target="y", kind="Mean", shape=1)}
    training_config = next(iter(mlp_training_schedule.values()))
    run_eda(
        tmp_path / "stats",
        input_configs,
        output_configs,
        training_config,
        compute_config=cpu_compute_config
    )

    assert (tmp_path / "stats" / "input" / "x.nc").exists()
    assert (tmp_path / "stats" / "output" / "y.nc").exists()
