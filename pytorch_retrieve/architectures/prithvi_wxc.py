"""
pytorch_retrieve.architectures.prithvi_wxc
==========================================

The PrithviWxC model.
"""
"""
pytorch_retrieve.architectures.prithvi_wxc
==========================================

Defines the PrithviWxC architecture for use within pytorch_retrieve.
"""
from dataclasses import dataclass, asdict
import os
from typing import Any, Dict, List, Union
from pathlib import Path

import torch
from torch import nn

from pytorch_retrieve.config import get_config_attr, InputConfig, OutputConfig
from pytorch_retrieve.architectures.model import RetrievalModel
from .encoder_decoder import StemConfig, HeadConfig


@dataclass
class BackboneConfig:
    """
    Configuration of the model backbone.
    """
    in_channels: int = 160
    input_size_time: int = 2
    in_channels_static: int = 8
    input_scalers_epsilon: int = 0.0
    static_input_scalers_epsilon: int = 0.0
    n_lats_px: int = 360
    n_lons_px: int  = 576
    patch_size_px: List[int] = None
    mask_unit_size_px: List[int] = None
    embed_dim: int = 1024
    n_blocks_encoder: int = 8
    n_blocks_decoder: int = 2
    mlp_multiplier: int = 4
    n_heads: int = 16
    dropout: int = 0.0
    drop_path: int = 0.0
    parameter_dropout: int = 0.0
    positional_encoding: str = "fourier"


    @classmethod
    def parse(
        cls,
        backbone_config: Dict[str, Any],
    ) -> "BackboneConfig":
        """
        Parses the backbone config using the configuration of the small PrithviWxC model as default values.
        """
        in_channels = get_config_attr("in_channels", int, backbone_config, "backbone", default=160)
        input_size_time = get_config_attr("input_size_time", int, backbone_config, "backbone", default=2)
        in_channels_static = get_config_attr("in_channels_static", int, backbone_config, "backbone", default=8)
        input_scalers_epsilon = get_config_attr("input_scalers_epsilon", float, backbone_config, "backbone", default=0.0)
        static_input_scalers_epsilon = get_config_attr("static_input_scalers_epsilon", float, backbone_config, "backbone", default=0.0)
        n_lats_px = get_config_attr("n_lats_px", int, backbone_config, "backbone", default=360)
        n_lons_px = get_config_attr("n_lons_px", int, backbone_config, "backbone", default=576)
        patch_size_px = get_config_attr("patch_size_px", list, backbone_config, "backbone", default=[2, 2])
        mask_unit_size_px = get_config_attr("mask_unit_size_px", list, backbone_config, "backbone", default=[30, 32])
        embed_dim = get_config_attr("embed_dim", int, backbone_config, "backbone", default=1024)
        n_blocks_encoder = get_config_attr("n_blocks_encoder", int, backbone_config, "backbone", default=12)
        mlp_multiplier = get_config_attr("mlp_multiplier", int, backbone_config, "backbone", default=4)
        n_heads = get_config_attr("n_heads", int, backbone_config, "backbone", default=16)
        dropout = get_config_attr("dropout", float, backbone_config, "backbone", default=0.0)
        drop_path = get_config_attr("drop_path", float, backbone_config, "backbone", default=0.0)
        parameter_dropout = get_config_attr("parameter_dropout", float, backbone_config, "backbone", default=0.0)
        positional_encoding = get_config_attr("positional_encoding", str, backbone_config, "backbone", default="fourier")

        return BackboneConfig(
            in_channels=in_channels,
            input_size_time=input_size_time,
            in_channels_static=in_channels_static,
            input_scalers_epsilon=input_scalers_epsilon,
            static_input_scalers_epsilon=static_input_scalers_epsilon,
            n_lats_px=n_lats_px,
            n_lons_px=n_lons_px,
            patch_size_px=patch_size_px,
            mask_unit_size_px=mask_unit_size_px,
            embed_dim=embed_dim,
            n_blocks_encoder=n_blocks_encoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            parameter_dropout=parameter_dropout,
            positional_encoding=positional_encoding
        )

    def to_config_dict(self) -> Dict[str, object]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        dct = asdict(self)
        return dct

    def compile(self) -> nn.Module:
        """
        Compile backbone model.
        """
        from PrithviWxC.dataloaders.merra2 import (
            input_scalers,
            output_scalers,
            static_input_scalers,
        )
        from PrithviWxC.model import PrithviWxC

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
            "in_channels": self.in_channels,
            "input_size_time": self.input_size_time,
            "in_channels_static": self.in_channels_static,
            "input_scalers_epsilon": self.input_scalers_epsilon,
            "static_input_scalers_epsilon": self.static_input_scalers_epsilon,
            "n_lats_px": self.n_lats_px,
            "n_lons_px": self.n_lons_px,
            "patch_size_px": self.patch_size_px,
            "mask_unit_size_px": self.mask_unit_size_px,
            "embed_dim": self.embed_dim,
            "n_blocks_encoder": self.n_blocks_encoder,
            "n_blocks_decoder": self.n_blocks_decoder,
            "mlp_multiplier": self.mlp_multiplier,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "drop_path": self.drop_path,
            "parameter_dropout": self.parameter_dropout,
            "positional_encoding": self.positional_encoding,
            "decoder_shifting": True,
            "mask_ratio_inputs": 0.99,
            "residual": 'ignore',
            "masking_mode": "both"
        }

        kwargs["input_scalers_mu"] = in_mu
        kwargs["input_scalers_sigma"] = in_sig
        kwargs["static_input_scalers_mu"] = static_mu
        kwargs["static_input_scalers_sigma"] = static_sig
        kwargs["output_scalers"] = output_sig ** 0.5
        kwargs["masking_mode"] = "local"
        kwargs["decoder_shifting"] = True
        kwargs["mask_ratio_inputs"] = 0.0

        model = PrithviWxC(**kwargs)
        return model


@dataclass
class PrithviWxCConfig:
    """
    Dataclass reprsentation the configuration of a PrithviWxC model.
    """
    return_latent: bool
    stem_configs: Dict[str, StemConfig]
    backbone_config: BackboneConfig
    head_configs: Dict[str, HeadConfig]

    @classmethod
    def parse(
        cls,
        input_configs: Dict[str, InputConfig],
        output_configs: Dict[str, OutputConfig],
        arch_config: Dict[str, object],
    ):
        """
        Parse PrithviWxC model config.

        Args:
            input_configs: Dictionary defining the input configurations.
            output_configs: Dictionary defining the output configurations.
            archi_config: Dictionary defining the architecture configuration
        """
        backbone_config_dict = arch_config.get("backbone", {})
        backbone_config = BackboneConfig.parse(backbone_config_dict)
        return_latent = get_config_attr(
            "return_latent", bool, arch_config, "model config", default=False
        )

        stem_config_dict = arch_config.get("stem", {})
        stem_configs = {}
        for name, input_config in input_configs.items():
            if name in stem_config_dict:
                config_dict = stem_config_dict[name]
                stem_configs[name] = StemConfig.parse(
                    f"stem.{name}", name, input_config, config_dict
                )
            else:
                stem_configs[name] = StemConfig.parse(
                    f"stem.{name}", name, input_config, stem_config_dict
                )

        head_config_dict = arch_config.get("head", {})
        individual = head_config_dict.get("individual", True)
        if individual:
            head_configs = {}
            for name, output_config in output_configs.items():
                if name in head_config_dict:
                    config_dict = head_config_dict[name]
                else:
                    config_dict = head_config_dict.get("default", {})
                head_configs[name] = HeadConfig.parse(
                    backbone_config.in_channels, output_config, name, config_dict
                )
        else:
            head_configs = {
                name: HeadConfig.parse(
                    decoder_config.out_channels, output_config, "head", head_config_dict
                )
                for name, output_config in output_configs.items()
            }

        return PrithviWxCConfig(
            return_latent=return_latent,
            stem_configs=stem_configs,
            backbone_config=backbone_config,
            head_configs=head_configs,
        )

    def to_config_dict(self):
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        stem_configs = {
            name: cfg.to_config_dict() for name, cfg in self.stem_configs.items()
        }
        stem_configs["individual"] = True
        backbone_config = self.backbone_config.to_config_dict()
        head_configs = {
            name: cfg.to_config_dict() for name, cfg in self.head_configs.items()
        }
        head_configs["individual"] = True
        return {
            "name": "PrithviWxC",
            "return_latent": self.return_latent,
            "backbone": backbone_config,
            "stem": stem_configs,
            "head": head_configs,
        }

    @property
    def out_channels(self):
        return self.channels[-1]

    def get_stem_config(self, name: str) -> StemConfig:
        """
        Get stem config for a given input.

        Args:
            name: The name of the input.

        Return:
            A StemConfig object specifying the stem configuration for the
            given input.
        """
        if isinstance(self.stem_config, dict):
            if name in self.stem_config:
                return self.stem_config[name]
            return self.stem_config["default"]
        return self.stem_config

    def get_head_config(self, name: str) -> HeadConfig:
        """
        Get head config for a given input.

        Args:
            name: The name of the input.

        Return:
            A HeadConfig object specifying the head configuration for the
            given input.
        """
        if isinstance(self.head_config, dict):
            if name in self.head_config:
                return self.head_config[name]
            return self.head_config["default"]
        return self.head_config


class PrithviWxCModel(RetrievalModel):

    @classmethod
    def from_config_dict(cls, config_dict):
        """
        Create PrithviWxC model from a configuration dictionary.

        Args:
            config_dict: A configuration dictionary defining the configuration of
                 the encoder-decoder architecture.
        """
        input_config = get_config_attr(
            "input", dict, config_dict, "model config", default={}
        )
        output_config = get_config_attr(
            "output", dict, config_dict, "model config", required=True
        )

        arch_config = get_config_attr("architecture", dict, config_dict, "model config")
        preset = get_config_attr("preset", str, arch_config, "architecture", "none")
        if preset != "none":
            preset_file = Path(__file__).parent / "presets" / f"{preset}.toml"
            if not preset_file.exists():
                raise RuntimeError(f"The preset configuration {preset} does not exist.")
            preset = read_config_file(preset_file)
            arch_config = update_recursive(preset, arch_config)

        input_config = {
            name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
        }
        output_config = {
            name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
        }

        config = PrithviWxCConfig.parse(input_config, output_config, arch_config)
        return cls(
            input_config=input_config, output_config=output_config, arch_config=config
        )


    def __init__(
        self,
        input_config: Dict[str, InputConfig],
        output_config: Dict[str, OutputConfig],
        arch_config: PrithviWxCConfig,
    ):
        """
        Args:
            input_config: A dictionary mapping input names to corresponding
                InputConfig objects.
            output_config: A dictionary mapping output names to corresponding
                OutputConfig objects.
            arch_config: A EncoderDecoder config object describing the encoder-decoder
                architecture.
        """

        super().__init__(
            config_dict={
                "input": {
                    name: cfg.to_config_dict() for name, cfg in input_config.items()
                },
                "output": {
                    name: cfg.to_config_dict() for name, cfg in output_config.items()
                },
                "architecture": arch_config.to_config_dict(),
            }
        )
        self.return_latent = arch_config.return_latent
        self.stems = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.stem_configs.items()}
        )
        self.backbone = arch_config.backbone_config.compile()
        self.heads = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.head_configs.items()}
        )

    @property
    def output_names(self) -> List[str]:
        """
        Names of the outputs from this model.
        """
        return list(self.heads.keys())

    def forward_unroll(
        self, x: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Creates a prediction by unrolling.

        Unrolled forecasts should contain static data and lead_time inputs with an additional
        time dimension.

        Args:
            x: A dictionary containing the expected model inputs.

        Return:
            A dictionary mapping output names to corresponding lists of output tensors.
        """
        assert x["static"].ndim == 5
        n_steps = x["static"].shape[1]

        latent_preds = []

        x_step = x["x"]

        for step in range(n_steps):
            y = self.backbone({
                "x": x_step,
                "static": x["static"][:, step],
                "lead_time": x["lead_time"],
                "input_time": x["input_time"]
            })
            y_out = (
                self.backbone.output_scalers * y + self.backbone.input_scalers_mu.reshape(1, -1, 1, 1)
            )
            x_step = torch.stack([x_step[:, -1], y_out], 1)
            latent_preds.append(y)

        preds = {
            name: [head(y_step) for y_step in latent_preds] for name, head in self.heads.items()
        }
        if self.return_latent:
            preds["y"] = latent_preds

        return preds

    def forward(
        self, x: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward tensor through network.

        Args:
            x: A dictionary containing the expected model inputs.

        Return:
        """
        if x["static"].ndim == 5:
            return self.forward_unroll(x)

        y = self.backbone(x)
        preds = {
            name: head(y) for name, head in self.heads.items()
        }
        if self.return_latent:
            preds["y"] = latent_preds

        return preds
