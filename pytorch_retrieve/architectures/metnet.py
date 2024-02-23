"""
pytorch_retrieve.architectures.metnet
=====================================

Implements a generic form of the MetNet architecture.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from pytorch_retrieve.config import (
    get_config_attr,
    InputConfig,
    OutputConfig,
    read_config_file,
)
from pytorch_retrieve.modules import metnet
from pytorch_retrieve.modules.input import StandardizationLayer
from pytorch_retrieve.modules.activation import get_activation_factory
from pytorch_retrieve.modules.conv import gru, heads
from pytorch_retrieve.modules.normalization import get_normalization_factory
from pytorch_retrieve.modules.output import Mean
from pytorch_retrieve.utils import update_recursive


@dataclass
class StemConfig:
    """
    Dataclass hodling the configuration of a MetNet stem.
    """

    input_name: str
    in_channels: int
    kind: str
    center_crop: bool
    normalize: Optional[str] = None

    @classmethod
    def parse(cls, name, input_name, input_config, config_dict):
        """
        Parse stem config.

        Args:
            name: The name of the section that is being parsed.
            input_name: The name of the input corresponding to this stem.
            config_dict: The dictionary containing the configuration.

        Return:
            A StemConfig object that represents the stem configuration.
        """
        in_channels = input_config.n_features
        kind = get_config_attr(
            f"kind", str, config_dict, "architecture.stem.{name}", "avgpool"
        )
        center_crop = get_config_attr(
            f"center_crop", bool, config_dict, "architecture.stem.{name}", True
        )
        normalize = input_config.normalize
        if normalize == "none":
            normalize = None
        return StemConfig(
            input_name=input_name,
            in_channels=in_channels,
            kind=kind,
            center_crop=center_crop,
            normalize=normalize,
        )

    @property
    def out_channels(self) -> int:
        """
        Calculates the number of output channels in a MetNet stem.
        """
        if self.kind == "avgpool":
            out_channels = self.in_channels
        else:
            out_channels = 4 * self.in_channels
        if self.center_crop:
            out_channels *= 2
        return out_channels

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        return asdict(self)

    def compile(self) -> nn.Module:
        """
        Compile stem to PyTorch module.
        """
        blocks = []
        if self.normalize != None:
            blocks.append(StandardizationLayer(self.input_name, self.in_channels))
        blocks.append(
            metnet.Stem(
                in_channels=self.in_channels,
                first_stage_kind=self.kind,
                center_crop=self.center_crop,
            )
        )
        return nn.Sequential(*blocks)


@dataclass
class TemporalEncoderConfig:
    """
    Dataclass hodling the configutation of the MetNet temporal encoder.
    """

    in_channels: int
    hidden_channels: int
    kernel_size: int
    n_layers: int
    activation_factory: Optional[str] = None
    normalization_factory: Optional[str] = None

    @classmethod
    def parse(
        cls, config_dict, time_steps: int, stem_cfgs: Dict[str, StemConfig]
    ) -> "TemporalEncoderConfig":
        """
        Parse temporal encoder configuration from dictionary.
        """
        in_channels = time_steps + sum(
            [stem_cfg.out_channels for stem_cfg in stem_cfgs.values()]
        )
        hidden_channels = get_config_attr(
            f"hidden_channels",
            int,
            config_dict,
            "architecture.temporal_encoder",
            required=True,
        )
        n_layers = get_config_attr(
            f"n_layers",
            int,
            config_dict,
            "architecture.temporal_encoder",
            required=True,
        )
        kernel_size = get_config_attr(
            f"kernel_size", int, config_dict, "architecture.temporal_encoder", 3
        )

        activation_factory = get_config_attr(
            f"activation_factory",
            str,
            config_dict,
            "architecture.temporal_encoder",
            "Tanh",
        )
        normalization_factory = get_config_attr(
            f"normalization_factory",
            str,
            config_dict,
            "architecture.temporal_encoder",
            None,
        )
        return TemporalEncoderConfig(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
        )

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        return asdict(self)

    def compile(self) -> nn.Module:
        """
        Compile stem to PyTorch module.
        """
        activation_factory = self.activation_factory
        if activation_factory is not None:
            activation_factory = get_activation_factory(activation_factory)

        normalization_factory = self.normalization_factory
        if normalization_factory is not None:
            normalization_factory = get_normalization_factory(normalization_factory)

        return gru.GRUNetwork(
            self.in_channels,
            self.hidden_channels,
            self.kernel_size,
            self.n_layers,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
        )


@dataclass
class SpatialAggregatorConfig:
    """
    Dataclass describing the head of a MetNet architecture.
    """

    input_size: int
    n_channels: int
    depth: int
    n_heads: int

    @classmethod
    def parse(
        cls,
        input_size: int,
        temporal_encoder_config: TemporalEncoderConfig,
        config_dict: Dict[str, Any],
    ) -> "SpatialAggregatorConfig":
        n_channels = temporal_encoder_config.hidden_channels
        depth = get_config_attr(
            "depth", int, config_dict, f"architecture.spatial_aggregator", required=True
        )
        n_heads = get_config_attr(
            "n_heads", int, config_dict, f"architecture.spatial_aggregator", 4
        )
        return SpatialAggregatorConfig(
            input_size=input_size // 4,
            n_channels=n_channels,
            depth=depth,
            n_heads=n_heads,
        )

    @property
    def out_channels(self) -> int:
        return self.n_channels

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        return asdict(self)

    def compile(self) -> nn.Module:
        """
        Compile head.

        Args:
            in_channels: The number of channels in the encoded head input.

        Return:

        """
        return metnet.SpatialAggregator(
            self.input_size, self.n_channels, self.depth, self.n_heads
        )


@dataclass
class HeadConfig:
    """
    Dataclass describing the head of a MetNet architecture.
    """
    output_config: OutputConfig
    in_channels: int
    depth: int = 1
    activation_factory: Optional[Callable[[], nn.Module]] = None
    normalization_factory: Optional[Callable[[int], nn.Module]] = None

    @classmethod
    def parse(cls, spatial_aggregator_config, output_config, name, config_dict):
        in_channels = spatial_aggregator_config.out_channels
        depth = get_config_attr("depth", int, config_dict, f"architecture.head", 1)
        activation_factory = get_config_attr(
            "activation_factory", str, config_dict, f"architecture.head", "GELU"
        )
        normalization_factory = get_config_attr(
            "normalization_factory",
            str,
            config_dict,
            f"architecture.head",
            "LayerNorm",
        )
        return HeadConfig(
            output_config=output_config,
            in_channels=in_channels,
            depth=depth,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
        )

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        config_dict = asdict(self)
        config_dict.pop("output_config")
        return config_dict

    def compile(self) -> nn.Module:
        """
        Compile head.

        Args:
            in_channels: The number of channels in the encoded head input.

        Return:

        """
        activation_factory = get_activation_factory(self.activation_factory)
        normalization_factory = get_normalization_factory(self.normalization_factory)
        head = heads.BasicConv(
            in_channels=self.in_channels,
            out_shape=self.output_config.shape,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
        )
        return nn.Sequential(head, self.output_config.get_output_layer())


@dataclass
class MetNetConfig:
    time_step: np.timedelta64
    forecast_range: np.timedelta64
    stem_configs: Dict[str, StemConfig]
    temporal_encoder_config: TemporalEncoderConfig
    spatial_aggregator_config: SpatialAggregatorConfig
    head_configs: Dict[str, HeadConfig]

    @classmethod
    def parse(
        cls,
        input_configs: Dict[str, InputConfig],
        output_configs: Dict[str, OutputConfig],
        arch_config: Dict[str, object],
    ):
        input_size = get_config_attr(
            "input_size", int, arch_config, "architecture", required=True
        )
        time_step = get_config_attr(
            "time_step", int, arch_config, "architecture", required=True
        )
        forecast_range = get_config_attr(
            "forecast_range", int, arch_config, "architecture", required=True
        )
        time_steps = int(forecast_range / time_step)

        stem_configs = {}
        stems = arch_config["stem"]
        for name in input_configs:
            cfg_dict = stems.get(name, stems)
            stem_configs[name] = StemConfig.parse(
                name if name in stems else "stem", name, input_configs[name], cfg_dict
            )

        temporal_encoder_config = TemporalEncoderConfig.parse(
            arch_config["temporal_encoder"], time_steps, stem_configs
        )

        spatial_aggregator_config = SpatialAggregatorConfig.parse(
            input_size, temporal_encoder_config, arch_config["spatial_aggregator"]
        )

        head_configs = {}
        for name, output_config in output_configs.items():
            head_cfg = arch_config.get("head", {})
            config_dict = head_cfg.get(name, head_cfg)
            head_configs[name] = HeadConfig.parse(
                spatial_aggregator_config, output_config, name, config_dict
            )

        return MetNetConfig(
            time_step=time_step,
            forecast_range=forecast_range,
            stem_configs=stem_configs,
            temporal_encoder_config=temporal_encoder_config,
            spatial_aggregator_config=spatial_aggregator_config,
            head_configs=head_configs,
        )


class MetNet(nn.Module):
    """
    Generic MetNet architecture as described in the respective papers.

    """

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any]) -> "MetNet":
        """
        Create MetNet model from a model configuration dictionary.

        Args:
            config_dict: A dictionary containing the model configuration.
        """
        input_configs = {
            name: InputConfig.parse(name, cfg)
            for name, cfg in config_dict["input"].items()
        }
        output_configs = {
            name: OutputConfig.parse(name, cfg)
            for name, cfg in config_dict["output"].items()
        }

        arch_config = config_dict["architecture"]
        preset = get_config_attr("preset", str, arch_config, "architecture", "none")
        if preset != "none":
            preset_file = Path(__file__).parent / "presets" / f"{preset}.toml"
            if not preset_file.exists():
                raise RuntimeError(f"The preset configuration {preset} does not exist.")
            preset = read_config_file(preset_file)
            arch_config = update_recursive(preset, arch_config)

        metnet_config = MetNetConfig.parse(input_configs, output_configs, arch_config)
        return MetNet(metnet_config)

    def __init__(self, config: MetNetConfig):
        """
        Create a MetNet model.

        Args:
            config: A MetNetConfig object representing the configuration
                of the MetNet model.
        """
        super().__init__()
        self.config = config
        stems = {
            name: stem_cfg.compile() for name, stem_cfg in config.stem_configs.items()
        }
        self.stems = nn.ModuleDict(stems)

        self.temporal_encoder = config.temporal_encoder_config.compile()
        self.spatial_aggregator = config.spatial_aggregator_config.compile()

        heads = {
            name: head_cfg.compile() for name, head_cfg in config.head_configs.items()
        }
        self.heads = nn.ModuleDict(heads)
        self.time_step = config.time_step
        self.forecast_range = config.forecast_range
        self.max_steps = self.forecast_range // self.time_step

    @property
    def output_names(self) -> List[str]:
        """
        Names of the outputs from this model.
        """
        return list(self.heads.keys())

    def encode_timestep(self, x, fstep=1):
        # Preprocess Tensor
        x = self.preprocessor(x)

        # Condition Time
        x = self.ct(x, fstep)

        x = self.image_encoder(x)

        # Temporal Encoder
        _, state = self.temporal_enc(self.drop(x))
        return self.temporal_agg(self.position_embedding(state))

    def forward(self, x: Dict[str, List[torch.Tensor]]) -> List[torch.Tensor]:
        # For each step in input sequence encode inputs and concatenate.
        seq_length = len(next(iter(x.values())))
        encs = []
        for seq_ind in range(seq_length):
            encs_i = []
            for name, stem in self.stems.items():
                x_s = x.get(name, None)
                if x_s is None:
                    raise RuntimeError(
                        f"MetNet model expected input {name} but it is not present in "
                        "inputs."
                    )
                if not isinstance(x_s, list):
                    raise RuntimeError(
                        f"MetNet model expected input {name} to be a list of inputs."
                    )
                encs_i.append(stem(x_s[seq_ind]))
            encs.append(torch.cat(encs_i, 1))

        # Encode times to predict.
        lead_times = x.get("lead_times", None)
        if lead_times is None:
            raise RuntimeError("MetNet model requires lead times as input.")

        preds = {}
        n_times = lead_times.shape[-1]
        for time_ind in range(n_times):
            time_ind = torch.floor(lead_times[:, time_ind] / self.time_step).to(
                torch.int64
            )
            lead_time_1h = nn.functional.one_hot(torch.tensor(time_ind), self.max_steps)
            lead_time_1h = lead_time_1h[..., None, None]

            # Add 1-hot-encoded lead time to all sequence elements and apply
            # temporal encoder.

            shape = list(encs[0].shape)
            shape[1] = self.max_steps
            lead_time_1h = torch.broadcast_to(lead_time_1h, shape)
            enc = self.temporal_encoder(
                [torch.cat([enc, lead_time_1h], 1) for enc in encs]
            )

            pred = self.spatial_aggregator(enc[-1])
            for name, head in self.heads.items():
                preds.setdefault(name, []).append(head(pred))

        return preds
