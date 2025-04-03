"""
Tests for the pytorch_retrieve.models.prithvi_wxc module.
===============================================================
"""
import os
from pathlib import Path

import pytest
import toml

import torch


try:
    import PrithviWxC
    from PrithviWxC.dataloaders.merra2 import (
        input_scalers,
        output_scalers,
        static_input_scalers,
    )

    from pytorch_retrieve.models.prithvi_wxc import (
            PrithviWxCObs,
            PrithviWxCXObs
    )
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
    Compiles a slimmed-down PrithviWxCObs model.
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
        "embed_dim": 256,
        "n_blocks_encoder": 4,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 1,
        "n_heads": 4,
        "dropout": 0.0,
        "drop_path": 0.0,
        "parameter_dropout": 0.0,
        "positional_encoding": "fourier",
        "obs_patch_size": (6, 4),
        "obs_features": 64,
        "decoder_shifting": True,
        "mask_ratio_inputs": 0.0,
        "mask_ratio_targets": 0.0,
        "residual": 'ignore',
        "masking_mode": "both",
        "checkpoint_encoder": (),
        "checkpoint_decoder": (),
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


@pytest.mark.skipif(not HAS_PRITHVI, reason="Needs PrithviWxC package installed.")
def test_prithvi_wxc_obs():
    """
    Test the PrithviWxC obs model.
    """
    mdl = compile_prithvi_wxc_obs()
    batch = {
        "x": torch.rand((1, 2, 160, 360, 576)),
        "static": torch.rand((1, 10, 360, 576)),
        "input_time": torch.tensor(3.0)[None],
        "lead_time": torch.tensor(3.0)[None],
        "climate": torch.rand((1, 160, 360, 576)),
        "obs": torch.rand((1, 2, 12, 18, 32, 1, 30, 32)),
        "obs_meta": torch.rand((1, 2, 12, 18, 32, 8, 30, 32)),
        "obs_mask": torch.rand((1, 2, 12, 18, 32, 30, 32)) > 0.5,
    }
    pred = mdl(batch)


def compile_prithvi_wxc_xobs():
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
        "embed_dim": 256,
        "n_blocks_encoder": 4,
        "n_blocks_decoder": 2,
        "mlp_multiplier": 1,
        "n_heads": 4,
        "dropout": 0.0,
        "drop_path": 0.0,
        "parameter_dropout": 0.0,
        "positional_encoding": "fourier",
        "obs_patch_size": (6, 4),
        "obs_features": 64,
        "decoder_shifting": True,
        "mask_ratio_inputs": 0.0,
        "mask_ratio_targets": 0.0,
        "residual": 'ignore',
        "masking_mode": "both",
        "checkpoint_encoder": (),
        "checkpoint_decoder": (),
    }

    kwargs["input_scalers_mu"] = in_mu
    kwargs["input_scalers_sigma"] = in_sig
    kwargs["static_input_scalers_mu"] = static_mu
    kwargs["static_input_scalers_sigma"] = static_sig
    kwargs["output_scalers"] = output_sig ** 0.5
    kwargs["masking_mode"] = "local"
    kwargs["decoder_shifting"] = False
    kwargs["mask_ratio_inputs"] = 0.0

    model = PrithviWxCXObs(**kwargs)
    return model


NEEDS_PRITHVI = pytest.mark.skipif(not HAS_PRITHVI, reason="Needs PrithviWxC module installed.")
def test_prithvi_wxc_xobs():
    """
    Test the PrithviWxC obs model.
    """
    mdl = compile_prithvi_wxc_xobs()
    batch = {
        "x": torch.rand((1, 2, 160, 360, 576)),
        "static": torch.rand((1, 10, 360, 576)),
        "input_time": torch.tensor(3.0)[None],
        "lead_time": torch.tensor(3.0)[None],
        "climate": torch.rand((1, 160, 360, 576)),
        "obs": torch.rand((1, 2, 12, 18, 32, 1, 30, 32)),
        "obs_meta": torch.rand((1, 2, 12, 18, 32, 8, 30, 32)),
        "obs_mask": torch.rand((1, 2, 12, 18, 32, 30, 32)) > 0.5,
    }
    pred = mdl(batch)
