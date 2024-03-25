"""
pytorch_retrieve.architecture.recurrent_encoder_decoder
=======================================================

Implements a recurrent version of the encoder-decoder architecture.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import torch
from torch import nn

from pytorch_retrieve.config import get_config_attr
from pytorch_retrieve.modules.conv import (
    blocks,
    downsampling,
    aggregation,
    upsampling,
    stems,
    heads,
)
from pytorch_retrieve.modules.input import StandardizationLayer
from pytorch_retrieve.modules.conv.encoders import (
    Encoder,
    MultiInputSharedEncoder,
    MultiInputParallelEncoder,
)
from pytorch_retrieve.modules.conv.utils import Scale
from pytorch_retrieve.modules.conv.decoders import Decoder
from pytorch_retrieve.modules.conv.recurrent import GRU, Assimilator, forward
from pytorch_retrieve.modules.activation import get_activation_factory
from pytorch_retrieve.modules.normalization import get_normalization_factory
from pytorch_retrieve.modules.output import Mean, Quantiles
from pytorch_retrieve.architectures.model import RetrievalModel
from pytorch_retrieve.config import InputConfig, OutputConfig, read_config_file
from pytorch_retrieve.utils import update_recursive
from pytorch_retrieve.modules.conv import recurrent
from .encoder_decoder import (
    StemConfig, HeadConfig, get_block_factory, get_aggregation_factory,
    get_downsampling_factory, get_upsampling_factory
)


@dataclass
class EncoderConfig:
    """
    Configuration for a recurrent encoder.

    Attributes:
        kind: The kind of encoder architecture.
        inputs: Dictionary mapping strings to corresponding channel numbers.
        channels: List of channels in each stage of the encoder.
        stage_depths: The depths of all stages in the encoder.
        downsampling_factors: The downsampling factors applied between the
             stages of the encoder.
        base_scale: The base scale of the encoder.
        block_factory: Name of the block factory to use to create the
             convolution blocks in the encoder-decoder architecture.
        downsampling_factory: Name of the factory class to use to create the
            downsampling modules.
        aggregation_factory: Name of the aggregation factory class to use to
            create the aggregation modules in the encoder.
        shared: Whether or not the encoder is shared between inputs.
        multi_scale: Whether inputs are integrated into the encoder at their
            native resolutions.
    """

    kind: str
    inputs: Dict[str, int]
    channels: List[int]
    stage_depths: List[int]
    downsampling_factors: List[int]
    base_scale: int = 1
    block_factory: str = "basic"
    block_factory_args: dict[str, Any] = None
    downsampling_factory: str = "max_pool"
    aggregation_factory: str = "linear"
    shared: bool = True
    multi_scale: bool = True
    recurrence_factory: str = "Assimilator"
    recurrence_factory_args: Optional[Dict[str, Any]] = None

    @classmethod
    def parse(
        cls, stem_configs: dict[str, StemConfig], config_dict: Dict[str, object]
    ) -> "EncoderConfig":
        """
        Parse encoder config from configuration dict.

        Args:
            stem_configs: Dictionary containing the stems configs for all inputs
                of the encoder-decoder architecture.
            config_dict: The 'encoder' section of the encoder-decoder
                architecture.

        Return:
             An 'EncoderConfig' object reprsenting the encoder configuration
             parsed from the provided encoder-decoder architecture.
        """
        input_scales = [config.out_scale for config in stem_configs.values()]
        inputs = {name: config.out_scale for name, config in stem_configs.items()}
        base_scale = min(input_scales)

        kind = get_config_attr("kind", str, config_dict, "architecture.encoder", "none")
        channels = get_config_attr(
            "channels", list, config_dict, "architecture.encoder"
        )
        stage_depths = get_config_attr(
            "stage_depths", list, config_dict, "architecture.encoder"
        )
        default = [2] * (len(channels) - 1)
        downsampling_factors = get_config_attr(
            "downsampling_factors", list, config_dict, "architecture.encoder", default
        )

        block_factory = get_config_attr(
            "block_factory", str, config_dict, "architecture.encoder", "BasicConv"
        )
        block_factory_args = get_config_attr(
            "block_factory_args", dict, config_dict, "architecture.encoder", {}
        )
        downsampling_factory = get_config_attr(
            "downsampling_factory",
            str,
            config_dict,
            "architecture.encoder",
            "MaxPool2d",
        )
        aggregation_factory = get_config_attr(
            "aggregation_factory", str, config_dict, "architecture.encoder", "Linear"
        )
        shared = get_config_attr(
            "shared", bool, config_dict, "architecture.encoder", True
        )
        multi_scale = get_config_attr(
            "multi_scale", bool, config_dict, "architecture.encoder", True
        )
        recurrence_factory = get_config_attr(
            "recurrence_factory", str, config_dict, "architecture.encoder", "Assimilator"
        )
        recurrence_factory_args = get_config_attr(
            "recurrence_factory_args", dict, config_dict, "architecture.encoder", {}
        )

        return EncoderConfig(
            kind=kind,
            inputs=inputs,
            channels=channels,
            stage_depths=stage_depths,
            downsampling_factors=downsampling_factors,
            base_scale=base_scale,
            block_factory=block_factory,
            block_factory_args=block_factory_args,
            downsampling_factory=downsampling_factory,
            aggregation_factory=aggregation_factory,
            shared=shared,
            multi_scale=multi_scale,
            recurrence_factory=recurrence_factory,
            recurrence_factory_args=recurrence_factory_args
        )

    @property
    def out_channels(self) -> int:
        """
        The channels of the last stage of the encoder.
        """
        return self.channels[-1]

    @property
    def scales(self) -> List[int]:
        """
        List of the scales corresponding to the output of all stages of the
        encoder.
        """
        scale = Scale(self.base_scale)
        scales = [scale]
        for f_d in self.downsampling_factors:
            scale = scale * f_d
            scales.append(scale)
        return scales

    @property
    def skip_connections(self) -> Dict[int, int]:
        """
        Dictionary specifying the number of channels in the skip tensors
        produced by this encoder.
        """
        return {scl: chans for scl, chans in zip(self.scales, self.channels)}

    def to_config_dict(self) -> Dict[str, object]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        return asdict(self)

    def compile(self) -> nn.Module:
        """
        Compile encoder.

        Return:
             The encoder module described by this EncoderConfiguration object.
        """
        recurrence_factory = recurrent.get_recurrence_factory(self.recurrence_factory)
        block_factory = get_block_factory(self.block_factory)(**self.block_factory_args)
        block_factory = recurrence_factory(block_factory)
        downsampling_factory = get_downsampling_factory(self.downsampling_factory)()
        aggregation_factory = get_aggregation_factory(self.aggregation_factory)

        if self.shared:
            encoder_class = MultiInputSharedEncoder
        else:
            encoder_class = MultiInputParallelEncoder
        return encoder_class(
            self.inputs,
            self.channels,
            self.stage_depths,
            downsampling_factors=self.downsampling_factors,
            block_factory=block_factory,
            aggregator_factory=aggregation_factory,
        )


@dataclass
class DecoderConfig:
    """
    Configuration of a decoder in an encoder-decoder architecture.

    Attributes:
        channels: List of the channels in each stage of the decoder.
        stage_depths: List of integers specifying the number of blocks
            in each stage of the decoder.
        upsampling_factors: List specifying the degree of the upsampling
            applied prior to each stage of the decoder.
        skip_connections: Dictionary mapping scales to corresponding channels
            of the skip connections.
        block_factory: Name defining the block factory to use to create the
            convolution blocks of encoder-decoder module.
        aggregator_factory: Name defining the aggregation block factory
            to use to create the aggregation modules in the decoder.
        kind: String defining the kind of the decoder architecture.
    """
    channels: List[int]
    stage_depths: List[int]
    upsampling_factors: List[int]
    skip_connections: Dict[int, int]
    block_factory: "basic"
    block_factory_args: Optional[Dict[str, Any]] = None
    upsampling_factory: str = "bilinear"
    aggregation_factory: str = "linear"
    kind: str = "standard"
    recurrence_factory: str = "LSTM"
    recurrence_factory_args: Dict[str, Any] = None

    @classmethod
    def parse(cls, encoder_config, config_dict):
        """
        Parse decoder config object from configuration dictionary.

        Args:
            decoder_config: The parsed encoder configuration of the
                architecture.
            config_dict: The 'decoder' section of the architecture configuration.

        Return:
            A DecoderConfig object representing the configuration of j
        """
        kind = get_config_attr("kind", str, config_dict, "architecture.decoder", "none")

        channels = get_config_attr(
            "channels", list, config_dict, "architecture.decoder"
        )
        channels.insert(0, encoder_config.out_channels)

        stage_depths = get_config_attr(
            "stage_depths", list, config_dict, "architecture.decoder"
        )
        default = [2] * len(stage_depths)
        upsampling_factors = get_config_attr(
            "upsampling_factors", list, config_dict, "architecture.decoder", default)

        block_factory = get_config_attr(
            "block_factory", str, config_dict, "architecture.decoder", "BasicConv"
        )
        block_factory_args = get_config_attr(
            "block_factory_args", dict, config_dict, "architecture.encoder", {}
        )
        upsampling_factory = get_config_attr(
            "upsampling_factory",
            str,
            config_dict,
            "architecture.decoder",
            "Bilinear",
        )
        aggregation_factory = get_config_attr(
            "aggregation_factory", str, config_dict, "architecture.decoder", "Linear"
        )
        recurrence_factory = get_config_attr(
            "recurrence_factory", str, config_dict, "architecture.decoder", "Assimilator"
        )
        recurrence_factory_args = get_config_attr(
            "recurrence_factory_args", dict, config_dict, "architecture.decoder", {}
        )
        skip_connections = encoder_config.skip_connections
        skip_connections = {
            key.scale: value for key, value in skip_connections.items()
        }
        return DecoderConfig(
            channels=channels,
            stage_depths=stage_depths,
            upsampling_factors=upsampling_factors,
            skip_connections=skip_connections,
            block_factory=block_factory,
            block_factory_args=block_factory_args,
            upsampling_factory=upsampling_factory,
            aggregation_factory=aggregation_factory,
            kind=kind,
            recurrence_factory=recurrence_factory,
            recurrence_factory_args=recurrence_factory_args
        )

    @property
    def out_channels(self):
        return self.channels[-1]

    def to_config_dict(self) -> Dict[str, object]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        config = asdict(self)
        config["channels"] = config["channels"][1:]
        return config

    def compile(self):
        recurrence_factory = recurrent.get_recurrence_factory(self.recurrence_factory)
        block_factory = get_block_factory(self.block_factory)(**self.block_factory_args)
        block_factory = recurrence_factory(block_factory)
        upsampling_factory = get_upsampling_factory(self.upsampling_factory)()
        aggregation_factory = get_aggregation_factory(self.aggregation_factory)

        skip_connections = self.skip_connections
        skip_connections = {
            Scale(key): value for key, value in skip_connections.items()
        }

        return Decoder(
            channels=self.channels,
            stage_depths=self.stage_depths,
            upsampling_factors=self.upsampling_factors,
            block_factory=block_factory,
            skip_connections=skip_connections,
            upsampler_factory=upsampling_factory,
        )


@dataclass
class EncoderDecoderConfig:
    """
    Dataclass reprsentation the configuration of an encoder-decoder model.
    """

    stem_configs: Dict[str, StemConfig]
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig
    head_configs: Dict[str, HeadConfig]

    @classmethod
    def parse(
        cls,
        input_configs: Dict[str, InputConfig],
        output_configs: Dict[str, OutputConfig],
        arch_config: Dict[str, object],
    ):
        stem_config_dict = arch_config.get("stem", {})
        individual_stems = stem_config_dict.get("individual", False)
        if individual_stems:
            stem_configs = {}
            for name, input_config in input_configs.items():
                if name in stem_config_dict:
                    config_dict = stem_config_dict[name]
                else:
                    if not "default" in stem_config_dict:
                        raise ValueError(
                            "Expected a stem config for every input or a "
                            "'default' stem config because the 'individual' "
                            " attribute in 'architecture.stem' is set. However, "
                            " none of these were found."
                        )

                    config_dict = stem_config_dict["default"]
                stem_configs[name] = StemConfig.parse(
                    name, name, input_config, config_dict
                )

        else:
            stem_configs = {
                name: StemConfig.parse("stem", name, input_config, stem_config_dict)
                for name, input_config in input_configs.items()
            }

        encoder_config = get_config_attr("encoder", dict, arch_config, "architecture")
        encoder_config = EncoderConfig.parse(stem_configs, encoder_config)

        decoder_config = get_config_attr("decoder", dict, arch_config, "architecture")
        decoder_config = DecoderConfig.parse(encoder_config, decoder_config)

        head_config_dict = arch_config.get("head", {})
        if "base" in head_config_dict:
            head_configs = {}
            for name, output_config in output_configs.items():
                if name in head_config_dict:
                    config_dict = head_config_dict[name]
                else:
                    config_dict = head_config_dict["base"]
                head_configs[name] = HeadConfig.parse(
                    decoder_config.out_channels, output_config, name, config_dict
                )
        else:
            head_configs = {
                name: HeadConfig.parse(
                    decoder_config.out_channels, output_config, "head", head_config_dict
                )
                for name, output_config in output_configs.items()
            }

        return EncoderDecoderConfig(
            stem_configs,
            encoder_config,
            decoder_config,
            head_configs,
        )

    def to_config_dict(self):
        stem_configs = {
            name: cfg.to_config_dict() for name, cfg in self.stem_configs.items()
        }
        stem_configs["individual"] = True
        head_configs = {
            name: cfg.to_config_dict() for name, cfg in self.head_configs.items()
        }
        head_configs["individual"] = True
        return {
            "name": "EncoderDecoder",
            "stem": stem_configs,
            "encoder": self.encoder_config.to_config_dict(),
            "decoder": self.decoder_config.to_config_dict(),
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
            return self.stem_config["base"]
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
            return self.head_config["base"]
        return self.head_config


class RecurrentEncoderDecoder(RetrievalModel):
    """
    PyTorch module implementing a generic U-Net/encoder-decoder architecture.
    """

    @classmethod
    def from_config_dict(cls, config_dict):
        """
        Create encoder-decoder model from a configuration dictionary.

        Args:
            config_dict: A configuration dictionary defining the configuration of
                 the encoder-decoder architecture.
        """
        input_config = get_config_attr(
            "input", dict, config_dict, "model config", required=True
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

        config = EncoderDecoderConfig.parse(input_config, output_config, arch_config)
        return cls(
            input_config=input_config, output_config=output_config, arch_config=config
        )

    def __init__(
        self,
        input_config: Dict[str, InputConfig],
        output_config: Dict[str, OutputConfig],
        arch_config: EncoderDecoderConfig,
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
        self.stems = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.stem_configs.items()}
        )
        self.encoder = arch_config.encoder_config.compile()
        self.decoder = arch_config.decoder_config.compile()
        self.heads = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.head_configs.items()}
        )

    @property
    def output_names(self) -> List[str]:
        """
        Names of the outputs from this model.
        """
        return list(self.heads.keys())

    def to_config_dict(self) -> Dict[str, object]:
        """
        Return configuration used to construct the EncoderDecoder model.

        Raises:
            RuntimeError if the model was not constructed from a configuration
            dict.
        """
        if self.config_dict is None:
            raise RuntimeError(
                "This EncoderDecoder architecture was not constructed from a config dict "
                "and can thus not be serialized."
            )
        return self.config_dict

    def forward(
        self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward tensor through network.

        Args:
            x: A single tensor or a dict of tensors to propagate through
                the network.

        Return:
            A single output tensor or a dictionary mapping output names to output
            tensors.
        """
        if not isinstance(x, dict):
            if len(self.stems) > 1:
                raise ValueError(
                    "The input is a single tensor but the architecture has more "
                    "than one stem. This requires the input to be a dictionary "
                    " mapping input names to corresponding data tensors."
                )
            name, stem = next(iter(self.stems.items()))
            encs = {name: forward(stem, x)}
        else:
            encs = {inpt: forward(self.stems[inpt], x[inpt]) for inpt in x}

        encs = self.encoder(encs)
        decs = self.decoder(encs)

        output = {name: forward(head, decs) for name, head in self.heads.items()}

        if not isinstance(x, dict):
            return next(iter(output.values()))

        return output
