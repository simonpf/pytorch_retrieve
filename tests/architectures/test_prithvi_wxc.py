"""
Tests for the pytorch_retrieve.architectures.prithvi_wxc module.
===============================================================
"""
from pathlib import Path
import os
import pytest
import toml

import torch


from pytorch_retrieve.config import InputConfig, OutputConfig


try:
    import PrithviWxC
    from PrithviWxC.dataloaders.merra2 import input_scalers, output_scalers, static_input_scalers
    from pytorch_retrieve.models.prithvi_wxc import PrithviWxCObs
    HAS_PRITHVI = True
except ImportError:
    HAS_PRITHVI = False


NEEDS_PRITHVI = pytest.mark.skipif(not HAS_PRITHVI, reason="Needs PrithviWxC module installed.")
NEEDS_PRITHVI_DATA = pytest.mark.skipif(
    not "PRITHVI_DATA_PATH" in os.environ,
    reason="Needs PrithviWxC data."
)


def compile_prithvi_wxc_obs():
    """
    Compiles a slimmed-down PrithviWxC model.
    """
    prithvi_data_path = Path(os.environ["PRITHVI_DATA_PATH"])
    if not prithvi_data_path.exists():
        raise ValueError(
            "PRITHVI_DATA_PATH must point to an existing directory and contain the PrithviWxC scaling factors."
        )

    VERTICAL_VARS = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
    STATIC_SURFACE_VARS = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
    SURFACE_VARS = [
        "EFLUX", "GWETROOT", "HFLUX", "LAI", "LWGAB", "LWGEM", "LWTUP", "PS", "QV2M", "SLP",
        "SWGNT", "SWTNT", "T2M", "TQI", "TQL", "TQV", "TS", "U10M", "V10M", "Z0M"
    ]
    LEVELS = [
        34.0, 39.0, 41.0, 43.0, 44.0, 45.0, 48.0, 51.0, 53.0, 56.0, 63.0, 68.0, 71.0, 72.0
    ]

    in_mu, in_sig = input_scalers(
        SURFACE_VARS,
        VERTICAL_VARS,
        LEVELS,
        prithvi_data_path / "musigma_surface.nc",
        prithvi_data_path / "musigma_vertical.nc",
    )
    output_sig = output_scalers(
        SURFACE_VARS,
        VERTICAL_VARS,
        LEVELS,
        prithvi_data_path / "anomaly_variance_surface.nc",
        prithvi_data_path / "anomaly_variance_vertical.nc",
    )

    static_mu, static_sig = static_input_scalers(
        prithvi_data_path / "musigma_surface.nc",
        STATIC_SURFACE_VARS,
    )

    kwargs = {
        "in_channels": 160,
        "input_size_time": 2,
        "in_channels_static": 8,
        "input_scalers_epsilon": 1e-5,
        "static_input_scalers_epsilon": 1e-5,
        "n_lats_px": 360,
        "n_lons_px": 576,
        "patch_size_px": (2, 2),
        "mask_unit_size_px": (30, 32),
        "embed_dim": 512,
        "n_blocks_encoder": 4,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "drop_path": 0.0,
        "parameter_dropout": 0.0,
        "positional_encoding": "fourier",
        "obs_patch_size": (6, 4),
        "obs_features": 32,
        "decoder_shifting": True,
        "mask_ratio_inputs": 0.0,
        "residual": 'ignore',
        "masking_mode": "both",
        "mask_ratio_targets": 0.0
    }

    kwargs["input_scalers_mu"] = in_mu
    kwargs["input_scalers_sigma"] = in_sig
    kwargs["static_input_scalers_mu"] = static_mu
    kwargs["static_input_scalers_sigma"] = static_sig
    kwargs["output_scalers"] = output_sig ** 0.5
    kwargs["masking_mode"] = "local"
    kwargs["decoder_shifting"] = False
    kwargs["mask_ratio_inputs"] = 0.0

    model = PrithviWxCObs(**kwargs)
    return model

@NEEDS_PRITHVI
def test_backbone_config():
    """
    Test backbone configuration.
    """
    from pytorch_retrieve.architectures import prithvi_wxc
    backbone_config = prithvi_wxc.BackboneConfig.parse({})
    assert backbone_config.embed_dim == 1024


PRITHVI_CONFIG = """
[architecture.name]
name = "PrithviWxC"

[architecture.backbone]
in_channels = 160
input_size_time = 2
in_channels_static = 8
input_scalers_epsilon = 0.0
static_input_scalers_epsilon = 0.0
n_lats_px = 360
n_lons_px = 576
patch_size_px = [2, 2]
mask_unit_size_px = [30, 32]
embed_dim = 1024
n_blocks_encoder = 8
n_blocks_decoder = 2
mlp_multiplier = 4
n_heads = 16
dropout = 0.0
drop_path = 0.0
parameter_dropout = 0.0
positional_encoding = "fourier"

[input.geo_ir]
n_features = 1

[output.surface_precip]
kind = "Mean"
"""

@pytest.fixture
def prithvi_wxc_config():
    return toml.loads(PRITHVI_CONFIG)

@NEEDS_PRITHVI
def test_parse_prithvi_config(prithvi_wxc_config):
    """
    Test parsing of PrithviWxC model config.
    """
    from pytorch_retrieve.architectures import prithvi_wxc
    input_configs = {name: InputConfig.parse(name, dct) for name, dct in prithvi_wxc_config["input"].items()}
    output_configs = {name: OutputConfig.parse(name, dct) for name, dct in prithvi_wxc_config["output"].items()}
    arch_config = prithvi_wxc.PrithviWxCConfig.parse(
        input_configs,
        output_configs,
        prithvi_wxc_config["architecture"]
    )


@NEEDS_PRITHVI
@NEEDS_PRITHVI_DATA
def test_prithvi_model_unroll(prithvi_wxc_config):
    """
    Test prediction using unrolling.
    """
    from pytorch_retrieve.architectures import prithvi_wxc
    model = prithvi_wxc.PrithviWxCModel.from_config_dict(prithvi_wxc_config)

    x = {
        "x": torch.rand((1, 2, 160, 360, 576)),
        "static": torch.rand((1, 2, 10, 360, 576)),
        "lead_time": torch.tensor([3.0]),
        "input_time": torch.tensor([3.0])
    }
    with torch.no_grad():
        pred = model(x)
    assert "surface_precip" in pred
    assert len(pred["surface_precip"]) == 2



@NEEDS_PRITHVI
@NEEDS_PRITHVI_DATA
def test_prithvi_model_direct(prithvi_wxc_config):
    """
    Test parsing of PrithviWxC model config.
    """
    from pytorch_retrieve.architectures import prithvi_wxc
    model = prithvi_wxc.PrithviWxCModel.from_config_dict(prithvi_wxc_config)

    x = {
        "x": torch.rand((1, 2, 160, 360, 576)),
        "static": torch.rand((1, 10, 360, 576)),
        "lead_time": torch.tensor([3.0]),
        "input_time": torch.tensor([3.0])
    }
    with torch.no_grad():
        pred = model(x)
    assert "surface_precip" in pred
    assert pred["surface_precip"].shape == (1, 1, 360, 576)


PRITHVI_OBS_CONFIG = """
[architecture.name]
name = "PrithviWxC"

[architecture.backbone]
in_channels = 160
input_size_time = 2
in_channels_static = 8
input_scalers_epsilon = 0.0
static_input_scalers_epsilon = 0.0
n_lats_px = 360
n_lons_px = 576
patch_size_px = [2, 2]
mask_unit_size_px = [30, 32]
embed_dim = 256
n_blocks_encoder = 8
n_blocks_decoder = 2
mlp_multiplier = 4
n_heads = 16
dropout = 0.0
drop_path = 0.0
parameter_dropout = 0.0
positional_encoding = "fourier"
variant = "obs"
obs_features = 16
obs_layers = 16
obs_patch_size = [6, 4]

[input.geo_ir]
n_features = 1

[output.surface_precip]
kind = "Mean"
"""

@pytest.fixture
def prithvi_wxc_obs_config():
    return toml.loads(PRITHVI_OBS_CONFIG)

@NEEDS_PRITHVI
@NEEDS_PRITHVI_DATA
def test_prithvi_wxc_obs(prithvi_wxc_obs_config):
    """
    Test propagating input and observations through Prithvi-WxC Obs.
    """
    from pytorch_retrieve.architectures import prithvi_wxc
    model = prithvi_wxc.PrithviWxCModel.from_config_dict(prithvi_wxc_obs_config)

    inpt = {
        "x": torch.rand(1, 2, 160, 360, 576),
        "climate": torch.rand(1, 160, 360, 576),
        "static": torch.rand(1, 10, 360, 576),
        "y": torch.rand(1, 1, 360, 576),
        "obs": torch.rand(1, 2, 12, 18, 32, 1, 30, 32),
        "obs_meta": torch.rand(1, 2, 12, 18, 32, 8, 30, 32),
        "obs_mask": torch.rand(1, 2, 12, 18, 32, 1, 30, 32),
        "input_time": torch.tensor([3.0]),
        "lead_time": torch.tensor([3.0]),
    }
    pred = model(inpt)


@NEEDS_PRITHVI
@NEEDS_PRITHVI_DATA
def test_prithvi_wxc_obs_unroll(prithvi_wxc_obs_config):
    """
    Test propagating input and observations through Prithvi-WxC Obs.
    """
    from pytorch_retrieve.architectures import prithvi_wxc
    model = prithvi_wxc.PrithviWxCModel.from_config_dict(prithvi_wxc_obs_config)
    inpt = {
        "x": torch.rand(1, 2, 160, 360, 576),
        "climate": torch.rand(1, 2, 160, 360, 576),
        "static": torch.rand(1, 2, 10, 360, 576),
        "y": torch.rand(1, 2, 1, 360, 576),
        "obs": torch.rand(1, 2, 12, 18, 32, 1, 30, 32),
        "obs_meta": torch.rand(1, 2, 12, 18, 32, 8, 30, 32),
        "obs_mask": torch.rand(1, 2, 12, 18, 32, 1, 30, 32),
        "input_time": torch.tensor([3.0]),
        "lead_time": torch.tensor([3.0]),
    }
    pred = model(inpt)
