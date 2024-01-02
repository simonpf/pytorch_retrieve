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
from typing import Callable, Dict, List, Optional, Tuple, Union

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
from pytorch_retrieve.modules.conv.decoders import Decoder
from pytorch_retrieve.modules.activation import get_activation_factory
from pytorch_retrieve.modules.normalization import get_normalization_factory
from pytorch_retrieve.modules.output import Mean
from pytorch_retrieve.architectures.model import RetrievalModel
from pytorch_retrieve.config import InputConfig, OutputConfig


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
            " block factories."
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
            " block factories."
        )
    return aggregation_factory


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
    """

    input_name: str
    in_channels: int
    in_scale: int
    out_channels: int
    depth: int = (0,)
    downsampling: int = 1
    normalize: Optional[str] = None

    @classmethod
    def parse(
        cls,
        name: str,
        input_name: str,
        input_config: InputConfig,
        config_dict: Dict[str, Any],
    ) -> StemConfig:
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
        in_scale = input_config.scale
        depth = get_config_attr(
            "depth", int, config_dict, f"architecture.stem.{name}", 0
        )
        out_channels = get_config_attr(
            "out_channels",
            int,
            config_dict,
            f"architecture.stem.{name}",
            default=in_channels,
        )
        downsampling = get_config_attr(
            "downsampling", int, config_dict, f"architecture.stem.{name}", 1
        )
        normalize = input_config.normalize
        if normalize == "none":
            normalize = None
        return StemConfig(
            input_name,
            in_channels=in_channels,
            in_scale=in_scale,
            out_channels=out_channels,
            depth=depth,
            downsampling=downsampling,
            normalize=normalize,
        )

    @property
    def out_scale(self):
        return self.in_scale * self.downsampling

    def to_config_dict(self) -> Dict[str, object]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        return asdict(self)

    def compile(self) -> nn.Module:
        """
        Compile stem into Pytorch module.

        Return:
            A ``torch.nn`` Module implementing the stem described by the
            object.
        """
        blocks = []
        if self.normalize != None:
            blocks.append(StandardizationLayer(self.input_name, self.in_channels))
        blocks.append(
            stems.BasicConv(
                self.in_channels,
                self.out_channels,
                depth=self.depth,
                downsampling=self.downsampling,
            )
        )
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
    downsampling_factory: str = "max_pool"
    aggregation_factory: str = "linear"
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

        return EncoderConfig(
            kind=kind,
            inputs=inputs,
            channels=channels,
            stage_depths=stage_depths,
            downsampling_factors=downsampling_factors,
            base_scale=base_scale,
            block_factory=block_factory,
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
        scale = self.base_scale
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
        block_factory = get_block_factory(self.block_factory)()
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
    upsampling_factory: "bilinear"
    aggregation_factory: str = "linear"
    kind: str = "standard"

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
            "upsampling_factors", list, config_dict, "architecture.encoder", default
        )

        block_factory = get_config_attr(
            "block_factory", str, config_dict, "architecture.encoder", "BasicConv"
        )
        upsampling_factory = get_config_attr(
            "upsampling_factory",
            str,
            config_dict,
            "architecture.encoder",
            "Bilinear",
        )
        aggregation_factory = get_config_attr(
            "aggregation_factory", str, config_dict, "architecture.encoder", "Linear"
        )
        return DecoderConfig(
            channels=channels,
            stage_depths=stage_depths,
            upsampling_factors=upsampling_factors,
            skip_connections=encoder_config.skip_connections,
            block_factory=block_factory,
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
        return asdict(self)

    def compile(self):
        block_factory = get_block_factory(self.block_factory)()
        upsampling_factory = get_upsampling_factory(self.upsampling_factory)()
        aggregation_factory = get_aggregation_factory(self.aggregation_factory)

        return Decoder(
            channels=self.channels,
            stage_depths=self.stage_depths,
            upsampling_factors=self.upsampling_factors,
            block_factory=block_factory,
            skip_connections=self.skip_connections,
            upsampler_factory=upsampling_factory,
        )


@dataclass
class HeadConfig:
    """
    Dataclass describing the head of a decoder-encoder architecture.
    """

    in_channels: int
    shape: Tuple[int]
    depth: int = 1
    activation_factory: Optional[Callable[[], nn.Module]] = None
    normalization_factory: Optional[Callable[[int], nn.Module]] = None

    @classmethod
    def parse(cls, decoder_config, output_config, name, config_dict):
        in_channels = decoder_config.out_channels
        depth = get_config_attr("depth", int, config_dict, f"architecture.head", 1)
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
            in_channels=in_channels,
            shape=output_config.shape,
            depth=depth,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
        )

    def to_config_dict(self) -> Dict[str, object]:
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
        activation_factory = get_activation_factory(self.activation_factory)
        normalization_factory = get_normalization_factory(self.normalization_factory)
        head = heads.BasicConv(
            in_channels=self.in_channels,
            out_shape=self.shape,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
        )
        return nn.Sequential(head, Mean())


@dataclass
class EncoderDecoderConfig:
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
                stem_configs[name] = StemConfig.parse(name, input_config, config_dict)

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
                    decoder_config, output_config, name, config_dict
                )
        else:
            head_configs = {
                name: HeadConfig.parse(
                    decoder_config, output_config, "head", head_config_dict
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
            "architecture": {
                "name": "EncoderDecoder",
                "stem": stem_configs,
                "encoder": self.encoder_config.to_config_dict(),
                "decoder": self.decoder_config.to_config_dict(),
                "head": head_configs,
            }
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

    def compile(self, config_dict=None):
        stems = {name: config.compile() for name, config in self.stem_configs.items()}
        encoder = self.encoder_config.compile()
        decoder = self.decoder_config.compile()
        heads = {name: config.compile() for name, config in self.head_configs.items()}

        return EncoderDecoder(stems, encoder, decoder, heads, config_dict=config_dict)


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
        input_config = get_config_attr("input", dict, config_dict, "model config")
        output_config = get_config_attr("output", dict, config_dict, "model config")
        arch_config = get_config_attr("architecture", dict, config_dict, "model config")
        input_cfgs = {
            name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
        }
        output_cfgs = {
            name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
        }

        config = EncoderDecoderConfig.parse(input_cfgs, output_cfgs, arch_config)
        return config.compile(config_dict=config_dict)

    def __init__(
        self,
        stems: Dict[str, nn.Module],
        encoder: nn.Module,
        decoder: nn.Module,
        heads: Dict[str, nn.Module],
        config_dict: Optional[Dict[str, object]] = None,
    ):
        """
        Args:
            stems: A dictionary mapping input names to corresponding stem
                modules.
            encoder: The encoder part of the model as a PyTorch module.
            decoder: The decoder part of the model as a PyTorch module.
            heads: A dictionary mapping output names to corresponding head
                modules.
            config_dict: An optional config dict describing the configuration
                of the model.
        """
        super().__init__(config_dict=config_dict)
        self.stems = nn.ModuleDict(stems)
        self.encoder = encoder
        self.decoder = decoder
        self.heads = nn.ModuleDict(heads)
        self.config_dict = config_dict

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
            encs = {name: stem(x)}
        else:
            encs = {inpt: self.stems[inpt](x[inpt]) for inpt in x}

        encs = self.encoder(encs)
        decs = self.decoder(encs)

        output = {name: head(decs) for name, head in self.heads.items()}

        if not isinstance(x, dict):
            return next(iter(output.values()))

        return output