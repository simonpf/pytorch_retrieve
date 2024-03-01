import numpy as np
import toml
import torch

from pytorch_retrieve.config import InputConfig, OutputConfig
from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.architectures.direct_forecast import (
    encode_times,
    DirectForecastConfig,
    DirectForecast
)


def test_encode_times():
    """
    Test the encoding of lead times.
    """

    lead_times = torch.tensor([
        [60, 120, 180],
        [30, 150, 180]
    ])

    lead_times_enc = encode_times(lead_times, 30, 240)

    assert lead_times_enc.shape == (6, 8)
    assert lead_times_enc[0, 0] == 0
    assert lead_times_enc[0, 1] == 1
    assert lead_times_enc[0, 2] == 0
    assert lead_times_enc[0, 3] == 0
    assert lead_times_enc[0, 4] == 0
    assert lead_times_enc[0, 5] == 0


DIRECT_FORECAST_CONFIG = (
    """
    [architecture]
    name = "DirectForecast"

    latent_dim = 64
    time_step = 30
    forecast_range = 240

    [architecture.encoder.encoder]
    channels = [32, 32, 64]
    stage_depths = [2, 2, 2]
    [architecture.encoder.decoder]
    channels = [32, 16]
    stage_depths= [2, 2]

    [architecture.temporal_encoder]
    kind="direct"
    n_inputs = 4
    channels = [256, 32, 64]
    input_size = [64, 64]

    [architecture.temporal_encoder.encoder]
    stage_depths = [2, 2, 2]
    [architecture.temporal_encoder.decoder]
    channels = [32, 16]
    stage_depths= [2, 2]

    [architecture.decoder]
    channels = [64]
    stage_depths = [1]
    upsampling_factors = [1]

    [architecture.encoder.stem]
    kind = "BasicConv"
    depth = 1
    out_channels = 32

    [input.seviri]
    n_features = 12

    [output.surface_precip]
    shape = 1
    kind = "Mean"

    """
)


def test_direct_forecast():
    """
    Ensure that an Direct_ForecastConfig can be parsed and inputs propagated through
    it.
    """

    config = toml.loads(DIRECT_FORECAST_CONFIG)

    output_configs = config["output"]
    output_configs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_configs.items()
    }
    input_configs = config["input"]
    input_configs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_configs.items()
    }
    arch_config = config["architecture"]

    direct_forecast_config = DirectForecastConfig.parse(
        input_configs, output_configs, arch_config
    )
    direct_forecast = DirectForecast(
        input_configs,
        output_configs,
        direct_forecast_config
    )

    x = {"seviri": [torch.rand(2, 12, 64, 64) for _ in range(4)]}
    x["lead_time"] = torch.tensor([[30, 60, 90, 120]] * 2)
    y = direct_forecast(x)

    assert "surface_precip" in y
    assert len(y["surface_precip"]) == 4
    assert y["surface_precip"][0].shape == (2, 1, 64, 64)


def test_direct_forecast_save_and_load(tmp_path):
    """
    Ensure that Direct_Forecast models can be saved and loaded.
    """
    config = toml.loads(DIRECT_FORECAST_CONFIG)

    output_configs = config["output"]
    output_configs = {
        name: OutputConfig.parse(name, cfg) for name, cfg in output_configs.items()
    }
    input_configs = config["input"]
    input_configs = {
        name: InputConfig.parse(name, cfg) for name, cfg in input_configs.items()
    }
    arch_config = config["architecture"]

    direct_forecast_config = DirectForecastConfig.parse(
        input_configs, output_configs, arch_config
    )
    direct_forecast = DirectForecast(
        input_configs,
        output_configs,
        direct_forecast_config
    )

    direct_forecast.save(tmp_path / "model.pt")
    loaded = load_model(tmp_path / "model.pt")

    assert loaded.n_params == direct_forecast.n_params
