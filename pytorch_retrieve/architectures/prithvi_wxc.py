"""
pytorch_retrieve.architectures.prithvi_wxc
==========================================

Defines the PrithviWxC architecture for use within pytorch_retrieve.
"""
from dataclasses import dataclass, asdict
import logging
import os
import types
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
import xarray as xr

from pytorch_retrieve.config import get_config_attr, InputConfig, OutputConfig
from pytorch_retrieve.architectures.model import RetrievalModel
from pytorch_retrieve.tensors import MeanTensor
from .encoder_decoder import StemConfig, HeadConfig


LOGGER = logging.getLogger(__name__)


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
    drop_path: Union[float, Tuple[float, float]] = 0.0
    parameter_dropout: int = 0.0
    positional_encoding: str = "fourier"
    obs_patch_size: Optional[Tuple[int, int]] = None,
    obs_features: Optional[int] = None
    drop_dynamic: float = 0.0
    drop_obs: float = 0.0
    conditional_merging: bool = False
    mask_ratio_targets: float = 0.0
    residual: str = "ignore"
    variant: Optional[str] = None
    encoder_shifting: bool = False
    decoder_shifting: bool = True
    checkpoint_encoder: Optional[List[int]] = ()
    checkpoint_decoder: Optional[List[int]] = ()
    scaling_factors: Optional[str] = None
    lora: Optional[bool] = False


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
        input_scalers_epsilon = get_config_attr("input_scalers_epsilon", float, backbone_config, "backbone", default=1e-3)
        static_input_scalers_epsilon = get_config_attr("static_input_scalers_epsilon", float, backbone_config, "backbone", default=1e-3)
        n_lats_px = get_config_attr("n_lats_px", int, backbone_config, "backbone", default=360)
        n_lons_px = get_config_attr("n_lons_px", int, backbone_config, "backbone", default=576)
        patch_size_px = get_config_attr("patch_size_px", list, backbone_config, "backbone", default=[2, 2])
        mask_unit_size_px = get_config_attr("mask_unit_size_px", list, backbone_config, "backbone", default=[30, 32])
        embed_dim = get_config_attr("embed_dim", int, backbone_config, "backbone", default=1024)
        n_blocks_encoder = get_config_attr("n_blocks_encoder", int, backbone_config, "backbone", default=12)
        n_blocks_decoder = get_config_attr("n_blocks_decoder", int, backbone_config, "backbone", default=2)
        mlp_multiplier = get_config_attr("mlp_multiplier", int, backbone_config, "backbone", default=4)
        n_heads = get_config_attr("n_heads", int, backbone_config, "backbone", default=16)
        dropout = get_config_attr("dropout", float, backbone_config, "backbone", default=0.0)
        drop_path = get_config_attr("drop_path", None, backbone_config, "backbone", default=0.0)
        parameter_dropout = get_config_attr("parameter_dropout", float, backbone_config, "backbone", default=0.0)
        positional_encoding = get_config_attr("positional_encoding", str, backbone_config, "backbone", default="fourier")
        obs_patch_size = get_config_attr("obs_patch_size", None, backbone_config, "backbone", required=False)
        if obs_patch_size is not None:
            obs_patch_size = tuple(obs_patch_size)
        drop_dynamic = get_config_attr("drop_dynamic", float, backbone_config, "backbone", default=0.0)
        drop_obs = get_config_attr("drop_obs", float, backbone_config, "backbone", default=0.0)
        obs_features = get_config_attr("obs_features", None, backbone_config, "backbone", required=False)
        conditional_merging = get_config_attr("conditional_merging", None, backbone_config, "backbone", required=False, default=False)
        residual = get_config_attr("residual", str, backbone_config, "backbone", default="ignore", required=False)
        variant = get_config_attr("variant", str, backbone_config, "backbone", default=None, required=False)
        encoder_shifting = get_config_attr("encoder_shifting", bool, backbone_config, "backbone", default=True, required=False)
        decoder_shifting = get_config_attr("decoder_shifting", bool, backbone_config, "backbone", default=True, required=False)
        checkpoint_encoder = get_config_attr("checkpoint_encoder", list, backbone_config, "backbone", default=(), required=False)
        checkpoint_decoder = get_config_attr("checkpoint_decoder", list, backbone_config, "backbone", default=(), required=False)
        scaling_factors = get_config_attr("scaling_factors", str, backbone_config, "backbone", default=None, required=False)
        scaling_factors = scaling_factors
        lora = get_config_attr("lora", bool, backbone_config, "backbone", default=False, required=False)

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
            n_blocks_decoder=n_blocks_decoder,
            mlp_multiplier=mlp_multiplier,
            n_heads=n_heads,
            dropout=dropout,
            drop_path=drop_path,
            parameter_dropout=parameter_dropout,
            positional_encoding=positional_encoding,
            obs_patch_size=obs_patch_size,
            obs_features=obs_features,
            conditional_merging=conditional_merging,
            drop_dynamic=drop_dynamic,
            drop_obs=drop_obs,
            residual=residual,
            variant=variant,
            encoder_shifting=encoder_shifting,
            decoder_shifting=decoder_shifting,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_decoder=checkpoint_decoder,
            scaling_factors=scaling_factors,
            lora=lora
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
        from pytorch_retrieve.models.prithvi_wxc import PrithviWxC, PrithviWxCObs, PrithviWxCXObs, PrithviWxCRegional

        if self.scaling_factors is None:
            self.scaling_factors = Path(os.environ["PRITHVI_DATA_PATH"])
            if not self.scaling_factors.exists():
                raise ValueError(
                    "PRITHVI_DATA_PATH must point to an existing directory and contain the PrithviWxC scaling factors."
                )
        else:
            if not Path(self.scaling_factors).exists():
                self.scaling_factors = Path(os.environ["PRITHVI_DATA_PATH"])
                if not self.scaling_factors.exists():
                    raise ValueError(
                        "'scaling_factors' must point to a directory containing the input and output scaling factors "
                        "'musigma_surface.nc', 'musigma_vertical.nc', 'anomaly_variance_surface.nc', and "
                        "'anomaly_variance_vertical.nc'."
                    )
        scaling_factors = Path(self.scaling_factors)

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
            str(scaling_factors / "musigma_surface.nc"),
            str(scaling_factors / "musigma_vertical.nc"),
        )
        output_sig = output_scalers(
            SURFACE_VARS,
            VERTICAL_VARS,
            LEVELS,
            str(scaling_factors / "anomaly_variance_surface.nc"),
            str(scaling_factors / "anomaly_variance_vertical.nc"),
        )

        static_mu, static_sig = static_input_scalers(
            str(scaling_factors / "musigma_surface.nc"),
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
            "obs_patch_size": self.obs_patch_size,
            "obs_features": self.obs_features,
            "conditional_merging": self.conditional_merging,
            "drop_dynamic": self.drop_dynamic,
            "drop_obs": self.drop_obs,
            "encoder_shifting": self.encoder_shifting,
            "decoder_shifting": self.decoder_shifting,
            "mask_ratio_inputs": 0.99,
            "residual": self.residual,
            "masking_mode": "both",
            "checkpoint_encoder": self.checkpoint_encoder,
            "checkpoint_decoder": self.checkpoint_decoder
        }

        kwargs["input_scalers_mu"] = in_mu
        kwargs["input_scalers_sigma"] = in_sig
        kwargs["static_input_scalers_mu"] = static_mu
        kwargs["static_input_scalers_sigma"] = static_sig
        kwargs["output_scalers"] = output_sig ** 0.5
        kwargs["masking_mode"] = "local"
        kwargs["mask_ratio_inputs"] = 0.0
        kwargs["mask_ratio_targets"] = self.mask_ratio_targets


        if self.variant == "obs":
            model = PrithviWxCObs(**kwargs)
        elif self.variant == "xobs":
            kwargs.pop("drop_dynamic")
            kwargs.pop("drop_obs")
            model = PrithviWxCXObs(**kwargs)
        elif self.variant == "regional":
            kwargs.pop("obs_features")
            kwargs.pop("obs_patch_size")
            kwargs.pop("drop_dynamic")
            kwargs.pop("drop_obs")
            kwargs.pop("conditional_merging")
            model = PrithviWxCRegional(**kwargs)
        else:
            kwargs.pop("obs_features")
            kwargs.pop("obs_patch_size")
            kwargs.pop("drop_dynamic")
            kwargs.pop("drop_obs")
            kwargs.pop("conditional_merging")
            model = PrithviWxC(**kwargs)
            model.forward = types.MethodType(new_forward, model)
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
                    backbone_config.in_channels, output_config, "head", head_config_dict
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
        self.backbone_lora = None
        if arch_config.backbone_config.lora:
            self.apply_lora()

        self.heads = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.head_configs.items()}
        )

    def apply_lora(self) -> None:
        """
        Adds low-rank adaptation to the backbone model.
        """
        try:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=["qkv_layer", "w_layer"],
                task_type="FEATURE_EXTRACTION",
            )
            backbone_lora = get_peft_model(self.backbone, lora_cfg)

            # Call the unwrapped base to preserve the exact forward signature
            self.backbone = backbone_lora.get_base_model()
            self.backbone_lora = backbone_lora
        except ImportError:
            LOGGER.warning(
                "Could not import 'peft' library. LoRA support is disabled."
            )

    @property
    def output_names(self) -> List[str]:
        """
        Names of the outputs from this model.
        """
        return list(self.heads.keys())

    def forward_unroll(
        self,
        x: Dict[str, torch.Tensor],
        **backbone_kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Creates a prediction by unrolling.

        Unrolled forecasts should contain static data and lead_time inputs with an additional
        time dimension.

        Args:
            x: A dictionary containing the expected model inputs.
            backbone_kwargs: Keyword arguments that will be forewarded to the backbone.

        Return:
            A dictionary mapping output names to corresponding lists of output tensors.
        """
        assert x["static"].ndim == 5
        n_steps = x["static"].shape[1]

        x_step = x["x"]
        latent_preds = []
        preds = {}

        forward_kwargs = {}
        if "obs" in x:
            obs_latent = self.backbone.encode_observations(x)
            forward_kwargs["obs_latent"] = obs_latent
        obs_latent = None


        for step in range(n_steps):

            if "climate" in x:
                climate = x["climate"][:, step]
            else:
                climate = None

            inpt =  {
                "x": x_step,
                "static": x["static"][:, step],
                "lead_time": x["lead_time"],
                "input_time": x["input_time"],
                "climate": climate,
            }

            if obs_latent is not None:
                forward_kwargs["total_lead_time"] = (step + 1) * x["lead_time"]

            y = self.backbone(inpt, apply_residual=False, **backbone_kwargs, **forward_kwargs)

            if self.backbone.residual == "temporal":
                raise ValueError(
                    "Temporal residual isn't supported yet."
                )
                y_out = self.backbone.output_scalers * y + x_hat
            elif self.backbone.residual == "climate":
                y_out = self.backbone.output_scalers * y + x["climate"][:, step]
            else:
                y_out = self.backbone.output_scalers * y + self.backbone.input_scalers_mu.reshape(
                    1, -1, 1, 1
                )
            x_step = torch.stack([x_step[:, -1], y_out], 1)

            latent_preds.append(y)
            for name, head in self.heads.items():
                preds.setdefault(name, []).append(head(y))

        if self.return_latent:
            preds["y"] = [MeanTensor(y) for y in latent_preds]

        return preds

    def forward(
            self, x: Dict[str, torch.Tensor],
            **backbone_kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward tensor through network.

        Args:
            x: A dictionary containing the expected model inputs.
            backbone_kwargs: Keyword arguments that will be forewarded to the backbone.

        Return:
        """
        from pytorch_retrieve.models.prithvi_wxc import PrithviWxCObs, PrithviWxCXObs
        if x["static"].ndim == 5:
            return self.forward_unroll(x, **backbone_kwargs)

        y = self.backbone(x, apply_residual=False, **backbone_kwargs)
        #if isinstance(self.backbone, (PrithviWxCObs, PrithviWxCXObs)):
        #    y = self.backbone(x, apply_residual=False)
        #else:
        #    y = self.backbone(x)

        preds = {
            name: head(y) for name, head in self.heads.items()
        }
        if self.return_latent:
            preds["y"] = y

        return preds


def new_forward(
        self,
        batch: dict[str, torch.Tensor],
        apply_residual: bool = True
) -> torch.Tensor:
    """
    Drop-in replacement for the Prithvi-WxC's forward function that adds an option to
    not apply residuals.

    Args:
        batch: Dictionary containing the keys 'x', 'y', 'input_time',
            'lead_time' and 'static'. The associated torch tensors have the
            following shapes:
            x: Tensor of shape [batch, time, parameter, lat, lon]
            y: Tensor of shape [batch, parameter, lat, lon]
            static: Tensor of shape [batch, channel_static, lat, lon]
            climate: Optional tensor of shape [batch, parameter, lat, lon]
            input_time: Tensor of shape [batch]. Or none.
            lead_time: Tensor of shape [batch]. Or none.
    Returns:
        Tensor of shape [batch, parameter, lat, lon].
    """
    assert batch["x"].shape[2] == self.in_channels
    assert batch["x"].shape[3] == self.n_lats_px
    assert batch["x"].shape[4] == self.n_lons_px
    #assert batch["y"].shape[1] == self.in_channels
    #assert batch["y"].shape[2] == self.n_lats_px
    #assert batch["y"].shape[3] == self.n_lons_px
    if self.positional_encoding == 'fourier':
        # the first two features (lat, lon) are encoded separately
        assert batch['static'].shape[1] - 2 == self.in_channels_static, "When setting self.positional_encoding to fourier, the number of static params change in the dataset. So, in the config, reduce num_static_channels (e.g., 4 instead of 7)."
    else:
        assert batch['static'].shape[1] == self.in_channels_static
    assert batch["static"].shape[2] == self.n_lats_px
    assert batch["static"].shape[3] == self.n_lons_px

    dtype = batch["x"].dtype
    x_rescaled = ((batch["x"].to(dtype=torch.float32) - self.input_scalers_mu) / (
        self.input_scalers_sigma + self.input_scalers_epsilon
    )).to(dtype=dtype)
    batch_size = x_rescaled.shape[0]

    if self.positional_encoding == 'fourier':
        x_static_pos = self.fourier_pos_encoding(batch['static']) # B, embed_dim, lat / patch_size, lon / patch_size
        x_static = (batch['static'][:, 2:].to(dtype=torch.float32) - self.static_input_scalers_mu[:, 3:]) / ( # The first two channels in batch['static'] are used in positional encoding
            self.static_input_scalers_sigma[:, 3:] + self.static_input_scalers_epsilon # This translates to the first three channels in 'static_input_scalers_mu'
        )
        x_static = x_static.to(dtype=dtype)
    else:
        x_static = (batch["static"].to(dtype=torch.float32) - self.static_input_scalers_mu) / (
            self.static_input_scalers_sigma + self.static_input_scalers_epsilon
        ).to(dtype=dtype)

    if self.residual == "temporal":
        # We create a residual of same shape as y
        index = torch.where(batch["lead_time"] > 0, batch["x"].shape[1] - 1, 0)
        index = index.view(-1, 1, 1, 1, 1)
        index = index.expand(batch_size, 1, *batch["x"].shape[2:])
        x_hat = torch.gather(batch["x"], dim=1, index=index)
        x_hat = x_hat.squeeze(1)
        assert (
            batch["y"].shape == x_hat.shape
        ), f'Shapes {batch["y"].shape} and {x_hat.shape} do not agree.'
    elif self.residual == "climate":
        climate_scaled = (
            batch["climate"].to(dtype=torch.float32) - self.input_scalers_mu.view(1, -1, 1, 1)
        ) / (
            self.input_scalers_sigma.view(1, -1, 1, 1) + self.input_scalers_epsilon
        )
        climate_scaled = climate_scaled.to(dtype=dtype)

    # [batch, time, parameter, lat, lon] -> [batch, time x parameter, lat, lon]
    x_rescaled = x_rescaled.flatten(1, 2)
    # Parameter dropout
    x_rescaled = self.parameter_dropout(x_rescaled)

    x_embedded = self.patch_embedding(x_rescaled)
    assert x_embedded.shape[1] == self.embed_dim

    if self.residual == "climate":
        static_embedded = self.patch_embedding_static(
            torch.cat((x_static, climate_scaled), dim=1)
        )
    else:
        static_embedded = self.patch_embedding_static(x_static)
    assert static_embedded.shape[1] == self.embed_dim

    if self.positional_encoding == 'fourier':
        static_embedded += x_static_pos

    x_embedded = self.to_patching(x_embedded)
    static_embedded = self.to_patching(static_embedded)

    time_encoding = self.time_encoding(batch['input_time'], batch['lead_time'])

    tokens = x_embedded + static_embedded + time_encoding


    # Now we generate masks based on masking_mode
    indices_masked, indices_unmasked = self.generate_mask(
        (batch_size, self._nglobal_mu)
    )
    indices_masked = indices_masked.to(device=tokens.device)
    indices_unmasked = indices_unmasked.to(device=tokens.device)
    maskdim: int = indices_masked.ndim

    # Unmasking
    unmask_view = (*indices_unmasked.shape, *[1] * (tokens.ndim - maskdim))
    unmasked = torch.gather(
        tokens,
        dim=maskdim - 1,
        index=indices_unmasked.view(*unmask_view).expand(
            *indices_unmasked.shape, *tokens.shape[maskdim:]
        ),
    )

    # Encoder
    lead_time = batch["lead_time"]
    x_encoded = self.encoder(unmasked, lead_time=lead_time)

    # Generate and position encode the mask tokens
    # (1, 1, 1, embed_dim) -> (batch, global_seq_masked, local seq, embed_dim)
    mask_view = (*indices_masked.shape, *[1] * (tokens.ndim - maskdim))
    masking = self.mask_token.repeat(*static_embedded.shape[:3], 1)
    masked = masking + static_embedded
    masked = torch.gather(
        masked,
        dim=maskdim - 1,
        index=indices_masked.view(*mask_view).expand(
            *indices_masked.shape, *tokens.shape[maskdim:]
        ),
    )

    recon, _ = self.reconstruct_batch(
        indices_masked, indices_unmasked, masked, x_encoded
    )

    x_decoded = self.decoder(recon, lead_time=lead_time)

    # Output: (batch, global sequence, local sequence, in_channels * patch_size[0] * patch_size[1])
    x_unembed = self.unembed(x_decoded)

    # Reshape to (batch, global_lat, global_lon, local_lat, local_lon, in_channels * patch_size[0] * patch_size[1])
    assert x_unembed.shape[0] == batch_size
    assert x_unembed.shape[1] == self.global_shape_mu[0] * self.global_shape_mu[1]
    assert x_unembed.shape[2] == self.local_shape_mu[0] * self.local_shape_mu[1]
    assert (
        x_unembed.shape[3]
        == self.in_channels * self.patch_size_px[0] * self.patch_size_px[1]
    )

    x_out = self.from_patching(x_unembed)

    # Pixel shuffle to (batch, in_channels, lat, lon)
    x_out = F.pixel_shuffle(x_out, self.patch_size_px[0])

    if not apply_residual:
        return x_out

    if self.residual == "temporal":
        x_out = self.output_scalers * x_out + x_hat
    elif self.residual == "climate":
        x_out = self.output_scalers * x_out + batch["climate"]
    elif self.residual == "none":
        x_out = self.output_scalers * x_out + self.input_scalers_mu.reshape(
            1, -1, 1, 1
        )

    return x_out
