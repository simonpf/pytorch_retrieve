"""
pytorch_retrieve.architecture.encoder_decoder
=============================================

Implements generic U-Net/encoder-decoder architectures for use as retrieval
backbones. The encoder-decoder architecture is designed to perform instantaneous
retrievals and provide estimates at the same or similar resolution as the
input data. The architecture supports multiple inputs at potentially different
resolutions and multiple outputs.
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
from pytorch_retrieve.modules.activation import get_activation_factory
from pytorch_retrieve.modules.normalization import get_normalization_factory
from pytorch_retrieve.modules.output import Mean
from pytorch_retrieve.architectures.model import RetrievalModel
from pytorch_retrieve.config import InputConfig, OutputConfig, read_config_file
from pytorch_retrieve.utils import update_recursive


def get_block_factory(name: str) -> Callable[[int, int], nn.Module]:
    """
    Retrieve a block factory by its name.

    Args:
        name: The name of the block factory.

    Return:
        A factory object to produce convolution block to be used within an
        encoder-decoder architecture.
    """
    try:
        block_factory = getattr(blocks, name)
    except AttributeError:
        raise ValueError(
            f"The block factory '{name}' is not defined. Please refer "
            " to the documentation of 'pytorch_retrieve.modules.conv.block' for "
            "available block factories."
        )
    return block_factory


def get_downsampling_factory(name) -> Callable[[int], nn.Module]:
    """
    Retrieve a downsampling factory by its name.

    Args:
        name: The name of the downsampling factory.

    Return:
        A factory object to produce downsampling block to be used within an
        encoder-decoder architecture.
    """
    if name is None or name.lower() == "none":
        return lambda: None
    try:
        downsampling_factory = getattr(downsampling, name)
    except AttributeError:
        raise ValueError(
            f"The downsampling factory '{name}' is not defined. Please refer "
            " to the documentation of "
            "'pytorch_retrieve.modules.conv.downsampling' for available "
            " block factories."
        )
    return downsampling_factory


def get_upsampling_factory(name) -> Callable[[int], nn.Module]:
    """
    Retrieve an upsampling factory by its name.

    Args:
        name: The name of the upsampling factory.

    Return:
        A factory object to produce upsampling block to be used within an
        encoder-decoder architecture.
    """
    try:
        upsampling_factory = getattr(upsampling, name)
    except AttributeError:
        raise ValueError(
            f"The upsampling factory '{name}' is not defined. Please refer "
            " to the documentation of "
            "'pytorch_retrieve.modules.conv.upsampling' for available "
            " upsampling factories."
        )
    return upsampling_factory


def get_aggregation_factory(name) -> Callable[[int], nn.Module]:
    """
    Retrieve a aggregation factory by its name.

    Args:
        name: The name of the aggregation factory.

    Return:
        A factory object to produce aggregation block to be used within an
        encoder-decoder architecture.
    """
    try:
        aggregation_factory = getattr(aggregation, name)
    except AttributeError:
        raise ValueError(
            f"The aggregation factory '{name}' is not defined. Please refer "
            " to the documentation of "
            "'pytorch_retrieve.modules.conv.aggregation' for available "
            " aggregation factories."
        )
    return aggregation_factory


def get_stem_factory(name) -> Callable[[int, int], nn.Module]:
    """
    Retrieve a stem factory by its name.

    Args:
        name: The name of the stem factory.

    Return:
        A factory object to produce a stem for an encoder-decoder architecture.
    """
    try:
        stem_factory = getattr(stems, name)
    except AttributeError:
        raise ValueError(
            f"The stem kind '{name}' is not defined. Please refer "
            " to the documentation of "
            "'pytorch_retrieve.modules.conv.stems' for available "
            " stems."
        )
    return stem_factory


def get_head_factory(name) -> Callable[[int, Tuple[int]], nn.Module]:
    """
    Retrieve a head factory by its name.

    Args:
        name: The name of the head factory.

    Return:
        A factory object to produce a head for an encoder-decoder architecture.
    """
    try:
        head_factory = getattr(heads, name)
    except AttributeError:
        raise ValueError(
            f"The head kind '{name}' is not defined. Please refer "
            " to the documentation of "
            "'pytorch_retrieve.modules.conv.heads' for available "
            " stems."
        )
    return head_factory


def get_output(name) -> Callable[[int, Tuple[int]], nn.Module]:
    """
    Retrieve a output factory by its name.

    Args:
        name: The name of the output factory.

    Return:
        A factory object to produce an output layer of a given kind.
    """
    try:
        output_factory = getattr(output, name)
    except AttributeError:
        raise ValueError(
            f"The output kind '{name}' is not defined. Please refer "
            " to the documentation of "
            "'pytorch_retrieve.modules.conv.heads' for available "
            " stems."
        )
    return output_factory

@dataclass
class StemConfig:
    """
    Configuration of a stem.

    All inputs are fed into a stem prior to being fed into the body of
    the encoder-decoder model. This class holds configuration attributes
    a single stem.

    Attributes:
        input_name: The name of the input corresponding to this stem.
        in_channels: The number of channels/features in the input corresponding
            to this stem.
        out_channels: The number of channels in the output from the stem.
        depth: The depth of the stem.
        downsampling: The degree of downsampling applied in the stem.
        normalize: String indicating the kind of normalization applied to
            the inputs.
        normalize_output: String of a normalization factory to use to normalize the stem outputs
            or 'None' to not apply normalization to stem outputs.

    """
    input_name: str
    in_channels: int
    in_scale: Tuple[int]
    kind: str
    out_channels: int
    depth: int = (0,)
    downsampling: Union[int, Tuple[int]] = 1
    upsampling: Union[int, Tuple[int]] = 1
    upsampling_factory: str = "Bilinear"
    normalize: Optional[str] = None
    normalize_output: Optional[str] = None

    @classmethod
    def parse(
        cls,
        name: str,
        input_name: str,
        input_config: InputConfig,
        config_dict: Dict[str, Any],
    ) -> "StemConfig":
        """
        Parse stem configuration from configuration dict.

        Args:
            name: The name of the section of the architecture configuration.
            input_name: The name of the input corresponding to this stem.
            input_config: The input configuration of the input corresponding to
                this stem.
            config_dict: Dictionary hodling the raw configuration of this stem.

        Return:
            A StemConfig object holding the parsed configuration of this stem.
        """
        in_channels = input_config.n_features
        in_scale = Scale(input_config.scale)

        depth = get_config_attr(
            "depth", int, config_dict, f"architecture.stem.{name}", 0
        )
        kind = get_config_attr(
            "kind",
            str,
            config_dict,
            f"architecture.stem.{name}",
            default="BasicConv",
        )
        out_channels = get_config_attr(
            "out_channels",
            int,
            config_dict,
            f"architecture.stem.{name}",
            default=in_channels,
        )
        downsampling = get_config_attr(
            "downsampling", None, config_dict, f"architecture.stem.{name}", 1
        )
        upsampling = get_config_attr(
            "upsampling", None, config_dict, f"architecture.stem.{name}", 1
        )
        upsampling_factory = get_config_attr(
            "upsampling_factory", None, config_dict, f"architecture.stem.{name}", "Bilinear"
        )
        if isinstance(downsampling, list):
            downsampling = tuple(downsampling)

        normalize = input_config.normalize

        normalize_output = get_config_attr(
            "normalize_output", None, config_dict, f"architecture.stem.{name}", None
        )

        return StemConfig(
            input_name,
            in_channels=in_channels,
            in_scale=in_scale,
            kind=kind,
            out_channels=out_channels,
            depth=depth,
            downsampling=downsampling,
            upsampling=upsampling,
            upsampling_factory=upsampling_factory,
            normalize=normalize,
            normalize_output=normalize_output
        )

    @property
    def out_scale(self) -> Scale:
        scale = Scale(self.in_scale)
        if self.upsampling is not None:
            scale = scale // self.upsampling
        if self.downsampling is not None:
            scale = scale * self.downsampling
        return scale

    def to_config_dict(self) -> Dict[str, object]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        dct = asdict(self)
        return dct

    def compile(self) -> nn.Module:
        """
        Compile stem into Pytorch module.

        Return:
            A ``torch.nn`` Module implementing the stem described by the
            object.
        """
        from pytorch_retrieve.modules.normalization import LayerNormFirst
        blocks = []
        if self.normalize != "none":
            blocks.append(StandardizationLayer(self.input_name, self.in_channels))

        if self.upsampling is not None:
            upsampling = Scale(self.upsampling)
            if max(upsampling.scale) > 1:
                upsampling_factory = get_upsampling_factory(self.upsampling_factory)()
                blocks.append(
                    upsampling_factory(
                        self.in_channels, self.in_channels, self.upsampling
                    )
                )

        stem_factory = get_stem_factory(self.kind)
        blocks.append(
            stem_factory(
                self.in_channels,
                self.out_channels,
                depth=self.depth,
                downsampling=self.downsampling,
            )
        )

        if self.normalize_output is not None:
            norm_factory = get_normalization_factory(self.normalize_output)
            blocks.append(norm_factory(self.out_channels))

        return nn.Sequential(*blocks)


@dataclass
class EncoderConfig:
    """
    Configuration for an encoder.

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
        block_factory_args: Dictionary of arguments that will be passed to
            the block factory.
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
    input_channels: Dict[str, int]
    channels: List[int]
    stage_depths: List[int]
    downsampling_factors: List[int]
    base_scale: Tuple[int] = Scale(1)
    block_factory: Union[str, List[str]] = "BasicConv"
    block_factory_args: Union[Dict[str, Any], List[Dict[str, Any]]] = None
    downsampling_factory: Optional[str] = None
    aggregation_factory: str = "Linear"
    shared: bool = True
    multi_scale: bool = True

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
             An 'EncoderConfig' object representing the encoder configuration
             parsed from the provided encoder-decoder architecture.
        """
        input_scales = [config.out_scale for config in stem_configs.values()]
        inputs = {
            name: config.out_scale for name, config in stem_configs.items()
        }
        input_channels = {
            name: config.out_channels for name, config in stem_configs.items()
        }
        tot_scales = np.prod(np.array(input_scales), -1)
        base_scale = min(input_scales)

        kind = get_config_attr("kind", str, config_dict, "architecture.encoder", "none")
        channels = get_config_attr(
            "channels", list, config_dict, "architecture.encoder", required=True
        )
        stage_depths = get_config_attr(
            "stage_depths", list, config_dict, "architecture.encoder", required=True
        )
        default = [2] * (len(channels) - 1)
        downsampling_factors = get_config_attr(
            "downsampling_factors", list, config_dict, "architecture.encoder", default
        )

        block_factory = get_config_attr(
            "block_factory", None, config_dict, "architecture.encoder", "BasicConv"
        )
        block_factory_args = get_config_attr(
            "block_factory_args", None, config_dict, "architecture.encoder", {}
        )
        if isinstance(block_factory, list):
            if not isinstance(block_factory_args, list):
                block_factory_args = [block_factory_args] * len(block_factory)

        downsampling_factory = get_config_attr(
            "downsampling_factory",
            str,
            config_dict,
            "architecture.encoder",
            "none",
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

        return EncoderConfig(
            kind=kind,
            inputs=inputs,
            input_channels=input_channels,
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
        dct = asdict(self)
        if isinstance(dct["block_factory_args"], list):
            args = dct["block_factory_args"]
            args = [dict(arg) for arg in args]
            dct["block_factory_args"] = args
        elif type(dct["block_factory_args"]) != dict:
            args = dct["block_factory_args"]
            dct["block_factory_args"] = dict(args)
        return dct

    def compile(self) -> nn.Module:
        """
        Compile encoder.

        Return:
             The encoder module described by this EncoderConfiguration object.
        """
        if isinstance(self.block_factory, list):
            block_factory = [
                get_block_factory(b_fac)(**b_fac_args)
                for b_fac, b_fac_args in zip(self.block_factory, self.block_factory_args)
            ]
        else:
            block_factory = get_block_factory(self.block_factory)(**self.block_factory_args)
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
            base_scale=self.base_scale,
            input_channels=self.input_channels,
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
    block_factory: Union[str, List[str]] = "basic"
    block_factory_args: Union[Dict[str, Any], List[Dict[str, Any]]] = None
    upsampling_factory: str = "bilinear"
    aggregation_factory: str = "linear"
    kind: str = "standard"

    @classmethod
    def parse(cls, encoder_config, config_dict):
        """
        Parse decoder config object from configuration dictionary.

        Args:
            encoder_config: The parsed decoder configuration of the
                architecture.
            config_dict: The 'decoder' section of the architecture configuration.

        Return:
            A DecoderConfig object representing the given configuration.
        """
        kind = get_config_attr("kind", str, config_dict, "architecture.decoder", "none")

        channels = get_config_attr(
            "channels", list, config_dict, "architecture.decoder"
        )
        channels.insert(0, encoder_config.out_channels)

        stage_depths = get_config_attr(
            "stage_depths", list, config_dict, "architecture.decoder", required=True
        )
        default = [2] * len(stage_depths)
        upsampling_factors = get_config_attr(
            "upsampling_factors", list, config_dict, "architecture.decoder", default
        )

        block_factory = get_config_attr(
            "block_factory", None, config_dict, "architecture.decoder", "BasicConv"
        )
        block_factory_args = get_config_attr(
            "block_factory_args", None, config_dict, "architecture.decoder", {}
        )
        if isinstance(block_factory, list):
            if not isinstance(block_factory_args, list):
                block_factory_args = [block_factory_args] * len(block_factory)
            else:
                if len(block_factory) != len(channels) - 1 or len(block_factory_args) != len(channels) - 1:
                    raise RuntimeError(
                        "If 'block_factory' and 'block_factory_args' are provided as lists, they must "
                        "have the same length as the number of stages in the decoder."
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
        )

    @property
    def out_channels(self):
        return self.channels[-1]

    def to_config_dict(self) -> Dict[str, object]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        dct = asdict(self)
        if isinstance(dct["block_factory_args"], list):
            args = dct["block_factory_args"]
            args = [dict(arg) for arg in args]
            dct["block_factory_args"] = args
        elif type(dct["block_factory_args"]) != dict:
            args = dct["block_factory_args"]
            dct["block_factory_args"] = dict(args)
        dct["channels"] = dct["channels"][1:]
        return dct

    def compile(self) -> nn.Module:
        """
        Compile the decoder module defined by this configuration.
        """
        if isinstance(self.block_factory, list):
            block_factory = [
                get_block_factory(b_fac)(**b_fac_args)
                for b_fac, b_fac_args in zip(self.block_factory, self.block_factory_args)
            ]
        else:
            block_factory = get_block_factory(self.block_factory)(**self.block_factory_args)

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
class HeadConfig:
    """
    Dataclass describing the head of a decoder-encoder architecture.
    """

    output_config: OutputConfig
    in_channels: int
    depth: int = 1
    kind: str = "BasicConv"
    activation_factory: Optional[Callable[[], nn.Module]] = None
    normalization_factory: Optional[Callable[[int], nn.Module]] = None

    @classmethod
    def parse(cls, in_channels, output_config, name, config_dict):
        depth = get_config_attr("depth", int, config_dict, f"architecture.head", 1)
        kind = get_config_attr("kind", str, config_dict, f"architecture.head", "BasicConv")
        activation_factory = get_config_attr(
            "activation_factory", str, config_dict, f"architecture.head", "ReLU"
        )
        normalization_factory = get_config_attr(
            "normalization_factory",
            str,
            config_dict,
            f"architecture.head",
            "BatchNorm2d",
        )
        return HeadConfig(
            output_config=output_config,
            in_channels=in_channels,
            depth=depth,
            kind=kind,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
        )

    def to_config_dict(self) -> Dict[str, object]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        dct = asdict(self)
        dct.pop("output_config")
        return dct

    def compile(self) -> nn.Module:
        """
        Compile head.

        Args:
            in_channels: The number of channels in the encoded head input.

        Return:

        """
        activation_factory = get_activation_factory(self.activation_factory)
        normalization_factory = get_normalization_factory(self.normalization_factory)
        head_factory = get_head_factory(self.kind)
        shape = self.output_config.get_output_shape()
        head = head_factory(
            in_channels=self.in_channels,
            out_shape=shape,
            depth=self.depth,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
        )
        output_layer = self.output_config.get_output_layer()
        return nn.Sequential(head, output_layer)


@dataclass
class EncoderDecoderConfig:
    """
    Dataclass representation the configuration of an encoder-decoder model.
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

        encoder_config = get_config_attr("encoder", dict, arch_config, "architecture")
        encoder_config = EncoderConfig.parse(stem_configs, encoder_config)

        decoder_config = get_config_attr("decoder", dict, arch_config, "architecture")
        decoder_config = DecoderConfig.parse(encoder_config, decoder_config)

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


class EncoderDecoder(RetrievalModel):
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
        is_sequence = False
        if not isinstance(x, dict):
            if len(self.stems) > 1:
                raise ValueError(
                    "The input is a single tensor but the architecture has more "
                    "than one stem. This requires the input to be a dictionary "
                    " mapping input names to corresponding data tensors."
                )
            # Sequence inputs are stacked to support 3D convolutions.
            if isinstance(x, list):
                is_sequence = True
                x = torch.stack(x, -3)

            name, stem = next(iter(self.stems.items()))
            encs = {name: stem(x)}
        else:
            encs = {}
            for name, tensor in x.items():

                if not name in self.stems:
                    continue

                if isinstance(tensor, list):
                    is_sequence = True
                    tensor = torch.stack(tensor, -3)
                encs[name] = self.stems[name](tensor)

        encs = self.encoder(encs)
        decs = self.decoder(encs)

        # Return
        if len(self.heads) == 0:
            return decs

        output = {name: head(decs) for name, head in self.heads.items()}

        # Convert 5D tensors back to lists of outputs.
        for name, tensor in output.items():
            if is_sequence:
                output[name] = list(torch.unbind(tensor, -3))

        if len(output) == 1 and not isinstance(x, dict):
            return next(iter(output.values()))

        return output
