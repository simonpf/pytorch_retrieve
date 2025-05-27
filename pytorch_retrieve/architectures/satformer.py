"""
pytorch_retrieve.architecture.satformer
=======================================

The SatFormer is a hybrid convolutional transformer architecture for image to
image mapping of satellite imagery.
"""

from copy import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from .model import RetrievalModel
from pytorch_retrieve.config import (
    InputConfig,
    OutputConfig,
    read_config_file,
    get_config_attr,
)

from pytorch_retrieve.modules.mlp import MLP
from pytorch_retrieve.modules.conv import blocks
from pytorch_retrieve.modules.conv.utils import Scale
from pytorch_retrieve.modules.encodings import FourierEncoding
from pytorch_retrieve.modules.input import StandardizationLayer
from pytorch_retrieve.modules.normalization import get_normalization_factory
from pytorch_retrieve.modules.activation import get_activation_factory
from pytorch_retrieve.tensors import MeanTensor
from pytorch_retrieve.modules.conv.upsampling import BilinearWNorm
from pytorch_retrieve.modules.conv.encoders import Encoder
from pytorch_retrieve.modules.conv.decoders import Decoder
from pytorch_retrieve.modules.conv.stages import SequentialWKeywordsStage
from pytorch_retrieve.modules.normalization import LayerNormFirst
from .encoder_decoder import StemConfig, HeadConfig, get_downsampling_factory


class FlipDims(nn.Module):
    """
    Reshapes a tensor containing sequence of images with dimensions
    [batch, chans, seq, height, width] moving channels to the back so that
    it can be propagated through linear layers expecting 2D tensors of shape
    [batch, chans], applies the base_module, and reshapes the data to its
    original shape.
    """

    def __init__(self, base_module: nn.Module):
        """
        base_module: A torch Module defining the module to apply to the reshaped
            input.
        """
        super().__init__()
        self.base_module = base_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape, apply module, and reshape back.
        """
        n_batch, _, n_seq, n_y, n_x = x.shape
        x = torch.permute(x, (0, 2, 3, 4, 1)).flatten(0, 3)
        y = self.base_module(x)
        n_chans = y.shape[-1]
        y = torch.permute(y.view(n_batch, n_seq, n_y, n_x, n_chans), (0, 4, 1, 2, 3))
        return y


@dataclass
class EncoderConfig:
    """
    Configuration for the encoder in a Satformer.

    Attributes:
        token_length: The length of the token separating different input and outputs>
        channels: List of channels in each stage of the encoder.
        stage_depths: The depths of all stages in the encoder.
        downsampling_factors: The downsampling factors applied between the
             stages of the encoder.
        base_scale: The base scale of the encoder.
        block_factory_args: Dictionary of arguments that will be passed to
            the block factory.
        downsampling_factory: Name of the factory class to use to create the
            downsampling modules.
    """

    token_length: int
    channels: List[int]
    stage_depths: List[int]
    downsampling_factors: List[int]
    base_scale: Tuple[int] = Scale(1)
    block_factory_args: Union[Dict[str, Any], List[Dict[str, Any]]] = None
    downsampling_factory: Optional[str] = None

    @classmethod
    def parse(
        cls,
        stem_configs: dict[str, StemConfig],
        config_dict: Dict[str, object],
        token_length: int,
    ) -> "EncoderConfig":
        """
        Parse encoder config from configuration dict.

        Args:
            stem_configs: Dictionary containing the stems configs for all inputs
                of the encoder-decoder architecture.
            config_dict: The 'encoder' section of the encoder-decoder
                architecture.
            token_length: The number of channel assigned to the input/output token.

        Return:
             An 'EncoderConfig' object reprsenting the encoder configuration
             parsed from the provided encoder-decoder architecture.
        """
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
        block_factory = "Satformer"
        block_factory_args = get_config_attr(
            "block_factory_args", None, config_dict, "architecture.encoder", {}
        )
        if isinstance(block_factory, list):
            if not isinstance(block_factory_args, list):
                block_factory_args = [block_factory_args] * len(block_factory)

        base_scale = get_config_attr(
            "base_scale", int, config_dict, "architecture.encoder", 1
        )

        downsampling_factory = get_config_attr(
            "downsampling_factory",
            str,
            config_dict,
            "architecture.encoder",
            "none",
        )

        return EncoderConfig(
            token_length=token_length,
            channels=channels,
            stage_depths=stage_depths,
            downsampling_factors=downsampling_factors,
            block_factory_args=block_factory_args,
            downsampling_factory=downsampling_factory,
            base_scale=base_scale
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
        scls = {scl: chans for scl, chans in zip(self.scales, self.channels)}
        scls[self.scales[0]] += self.token_length
        return scls

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
        block_factory = blocks.Satformer()
        if isinstance(self.block_factory_args, list):
            block_factory = [
                blocks.Satformer(**args) for args in self.block_factory_args
            ]
            stage_factory = [SequentialWKeywordsStage] * len(block_factory)
        else:
            block_factory = blocks.Satformer(self.block_factory_args)
            stage_factory = SequentialWKeywordsStage

        downsampling_factory = get_downsampling_factory(self.downsampling_factory)()

        channels = copy(self.channels)
        channels[0] += self.token_length

        return Encoder(
            channels,
            self.stage_depths,
            downsampling_factors=self.downsampling_factors,
            block_factory=block_factory,
            stage_factory=stage_factory,
            base_scale=self.base_scale
        )


@dataclass
class DecoderConfig:
    """
    Configuration of a decoder in an encoder-decoder architecture.

    Attributes:
        output_embed_dim: The embedding dimension for output.
        channels: List of the channels in each stage of the decoder.
        stage_depths: List of integers specifying the number of blocks
            in each stage of the decoder.
        upsampling_factors: List specifying the degree of the upsampling
            applied prior to each stage of the decoder.
        skip_connections: Dictionary mapping scales to corresponding channels
            of the skip connections.
        block_factory_args: Dictionary containing the argument passed to
            the block factory.
    """

    output_embed_dim: int
    channels: List[int]
    stage_depths: List[int]
    upsampling_factors: List[int]
    skip_connections: Dict[int, int]
    block_factory_args: Union[Dict[str, Any], List[Dict[str, Any]]] = None

    @classmethod
    def parse(cls, output_embed_dim, encoder_config, config_dict):
        """
        Parse decoder config object from configuration dictionary.

        Args:
            encoder_config: The parsed decoder configuration of the
                architecture.
            config_dict: The 'decoder' section of the architecture configuration.

        Return:
            A DecoderConfig object representing the given configuration.
        """
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

        block_factory = "SatformerBlock"
        block_factory_args = get_config_attr(
            "block_factory_args", None, config_dict, "architecture.decoder", {}
        )
        if isinstance(block_factory, list):
            if not isinstance(block_factory_args, list):
                block_factory_args = [block_factory_args] * len(block_factory)
            else:
                if (
                    len(block_factory) != len(channels) - 1
                    or len(block_factory_args) != len(channels) - 1
                ):
                    raise RuntimeError(
                        "If 'block_factory' and 'block_factory_args' are provided as lists, they must "
                        "have the same length as the number of stages in the decoder."
                    )

        skip_connections = encoder_config.skip_connections
        skip_connections = {key.scale: value for key, value in skip_connections.items()}

        return DecoderConfig(
            output_embed_dim,
            channels=channels,
            stage_depths=stage_depths,
            upsampling_factors=upsampling_factors,
            skip_connections=skip_connections,
            block_factory_args=block_factory_args,
        )

    @property
    def out_channels(self):
        return self.channels[-1]

    def get_scales(self, base_scale) -> List[int]:
        """
        List of the scales corresponding to the output of all stages of the
        encoder.
        """
        scale = base_scale
        scales = [scale]
        for f_d in self.upsampling_factors:
            scale = scale // f_d
            scales.append(scale)
        return scales

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

    def compile(self, token_length: int, skip_connections: bool = True) -> nn.Module:
        """
        Compile the decoder module defined by this configuration.
        """

        block_factory = blocks.Satformer()
        if isinstance(self.block_factory_args, list):
            block_factory = [
                blocks.Satformer(**args) for args in self.block_factory_args
            ]
            stage_factory = [SequentialWKeywordsStage] * len(block_factory)
        else:
            block_factory = blocks.Satformer(self.block_factory_args)
            stage_factory = SequentialWKeywordsStage

        upsampling_factory = BilinearWNorm()

        if skip_connections:
            skip_connections = self.skip_connections
            skip_connections = {
                Scale(key): value
                for key, value in skip_connections.items()
            }
        else:
            skip_connections = None

        channels = copy(self.channels)
        channels[-1] += token_length

        if skip_connections is not None:
            max_scale = max(skip_connections)
            skip_connections[max_scale] = channels[0]

        upsampling_factors = self.upsampling_factors

        return Decoder(
            channels=channels,
            stage_depths=self.stage_depths,
            upsampling_factors=self.upsampling_factors,
            block_factory=block_factory,
            stage_factory=stage_factory,
            skip_connections=skip_connections,
            upsampler_factory=upsampling_factory,
        )


@dataclass
class EncodingConfig:
    """
    Configuration of an encoding layer.
    """

    kind: str
    channels_in: int
    channels_out: int
    depth: int
    normalize: str
    input_name: str
    activation_factory: str
    normalization_factory: str
    residual_connections: Optional[str] = None

    @classmethod
    def parse(cls, name: str, config_dict: Dict[str, object]) -> "EncodingConfig":
        kind = get_config_attr(
            "kind",
            str,
            config_dict,
            f"architecture.encoding.{name}",
            default="mlp",
        )
        channels_in = get_config_attr(
            "channels_in",
            int,
            config_dict,
            f"architecture.encoding.{name}",
            required=True,
        )
        channels_out = get_config_attr(
            "channels_out",
            int,
            config_dict,
            f"architecture.encoding.{name}",
            required=True,
        )
        depth = get_config_attr(
            "depth", int, config_dict, f"architecture.encoding.{name}", default=1
        )
        normalize = get_config_attr(
            "normalize",
            str,
            config_dict,
            f"architecture.encoding.{name}",
            default=None,
        )
        input_name = get_config_attr(
            "input_name",
            str,
            config_dict,
            f"architecture.encoding.{name}",
            default=None,
        )
        if normalize is not None and input_name is None:
            raise ValueError(
                f"'normalize' attribute of encoder '{name}' is set to '{normalize}' but the 'input_name' property"
                "is not set."
            )

        activation_factory = get_config_attr(
            "activation_factory",
            str,
            config_dict,
            f"architecture.encoder.{name}",
            default="GELU",
        )
        normalization_factory = get_config_attr(
            "normalization_factory",
            str,
            config_dict,
            f"architecture.encoder.{name}",
            default="LayerNorm",
        )
        residual_connections = get_config_attr(
            "residual_connections",
            str,
            config_dict,
            f"architecture.encoder.{name}",
            default="simple",
        )
        return EncodingConfig(
            kind=kind,
            channels_in=channels_in,
            channels_out=channels_out,
            depth=depth,
            normalize=normalize,
            input_name=input_name,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
            residual_connections=residual_connections,
        )

    def to_config_dict(self):
        dct = asdict(self)
        return dct

    def compile(self) -> MLP:
        """
        Compile encoding.
        """
        from pytorch_retrieve.modules.normalization import LayerNormFirst

        blocks = []
        if self.normalize is not None and self.normalize != "none":
            blocks.append(
                StandardizationLayer(
                    kind=self.normalize,
                    name=self.input_name,
                    n_features=self.channels_in,
                )
            )

        if self.kind.lower() == "fourier":
            blocks.append(FourierEncoding(self.channels_in, self.channels_out, dim=1))
        else:
            activation_factory = get_activation_factory(self.activation_factory)
            normalization_factory = get_normalization_factory(
                self.normalization_factory
            )

            blocks += [
                FlipDims(
                    MLP(
                        in_channels=self.channels_in,
                        out_channels=self.channels_out,
                        n_layers=self.depth,
                        activation_factory=activation_factory,
                        normalization_factory=normalization_factory,
                        residual_connections=self.residual_connections,
                    )
                ),
                LayerNormFirst(self.channels_out),
            ]

        return nn.Sequential(*blocks)


@dataclass
class SatformerConfig:
    """
    Configuration for a Satformer.

    Attributes:
        token_length: The number of dimensions used to encode different inputs.
        output_embed_dim: The number of channels to use for the output embedding.
        stem_configs: Dictionary holding the configurations for the network stems.
        encoding_configs: Dictionary holding the configurations for the input
            embeddings.
        encoder_config: The configuration for the encoder.
        decoder_config: The configuration for the decoder.
        head_configs: The configuration for the network heads.
    """

    token_length: int
    output_embed_dim: int
    n_heads_perceiver: int
    stem_configs: Dict[str, StemConfig]
    encoding_configs: Dict[str, EncodingConfig]
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
        token_length = len(input_configs)
        output_embed_dim = get_config_attr(
            "output_embed_dim",
            int,
            arch_config,
            "architecture",
            required=False,
            default=32,
        )
        n_heads_perceiver = get_config_attr(
            "n_heads_perceiver",
            int,
            arch_config,
            "architecture",
            required=False,
            default=4,
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
        enc_config_dict = arch_config.get("encoding", {})
        enc_configs = {}
        for name, enc_config in enc_config_dict.items():
            enc_configs[name] = EncodingConfig.parse(name, enc_config)

        encoder_config = get_config_attr(
            "encoder", dict, arch_config, "architecture", required=True
        )
        encoder_config = EncoderConfig.parse(stem_configs, encoder_config, token_length)

        decoder_config = get_config_attr(
            "decoder", dict, arch_config, "architecture", required=True
        )
        decoder_config = DecoderConfig.parse(
            output_embed_dim, encoder_config, decoder_config
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
                    decoder_config.out_channels + encoder_config.token_length,
                    output_config,
                    name,
                    config_dict,
                )
        else:
            head_configs = {
                name: HeadConfig.parse(
                    decoder_config.out_channels + encoder_config.token_length,
                    output_config,
                    "head",
                    head_config_dict,
                )
                for name, output_config in output_configs.items()
            }

        return SatformerConfig(
            token_length,
            output_embed_dim,
            n_heads_perceiver,
            stem_configs,
            enc_configs,
            encoder_config,
            decoder_config,
            head_configs,
        )

    def to_config_dict(self):
        stem_configs = {
            name: cfg.to_config_dict() for name, cfg in self.stem_configs.items()
        }
        stem_configs["individual"] = True
        encoding_configs = {
            name: cfg.to_config_dict() for name, cfg in self.encoding_configs.items()
        }
        head_configs = {
            name: cfg.to_config_dict() for name, cfg in self.head_configs.items()
        }
        head_configs["individual"] = True
        return {
            "name": "Satformer",
            "output_embed_dim": self.output_embed_dim,
            "stem": stem_configs,
            "encoding": encoding_configs,
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


class Perceiver(nn.Module):
    """
    A perceiver module used to extract multi-scale output features from the input sequence
    at the beginning of each decoder stage.
    """
    def __init__(
            self,
            embed_dim: int,
            n_heads: int = 4
    ):
        """
        Args:
            embed_dim: The embedding dim of the perceiver.
            n_heads: The number of heads in the attention layer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Parameter(
            torch.normal(torch.zeros(embed_dim), torch.ones(embed_dim))
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.att = nn.MultiheadAttention(self.embed_dim, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        """
        Propagate input through perceiver.
        """
        n_batch, _, n_seq_in, n_y, n_x = x.shape
        x_att = torch.permute(x, (0, 3, 4, 2, 1)).reshape(-1, n_seq_in, self.embed_dim)
        x_att = self.norm(x_att)
        query = torch.repeat_interleave(self.query[None, None], x_att.shape[0], 0)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat_interleave(n_x * n_y, 0)
        x, _ = self.att(query, x_att, x_att, key_padding_mask=key_padding_mask)
        x = torch.permute(x.unflatten(0, (n_batch, n_y, n_x)), (0, 4, 3, 1, 2))
        return x


class CondPerceiver(nn.Module):
    """
    A conditional perceiver module used to extract multi-scale output features from the input sequence
    at the beginning of each decoder stage.
    """
    def __init__(
            self,
            cond_dim: int,
            embed_dim: int,
            n_heads: int
    ):
        """
        Args:
            cond_dim: The number of features in the conditioning input.
            embed_dim: The number of features in the perceiver input and output.
            n_heads: The number of heads in the attention layer.
        """
        super().__init__()
        self.cond_dim = cond_dim
        self.embed_dim = embed_dim
        self.query = nn.Linear(cond_dim, embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.att = nn.MultiheadAttention(self.embed_dim, num_heads=n_heads, batch_first=True)

    def forward(self, x_cond: torch.Tensor, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        """
        Propagate input through conditional perceiver.
        """
        n_batch, _, n_seq_in, n_y, n_x = x.shape
        rep = x_cond.shape[0] // x.shape[0]
        if rep > 1:
            x = x.repeat_interleave(rep, 0)
        x_att = torch.permute(x, (0, 3, 4, 2, 1)).reshape(-1, n_seq_in, self.embed_dim)
        x_in = torch.permute(x_cond, (0, 2, 3, 1)).reshape(-1, 1, self.cond_dim)
        x_in = self.norm(self.query(x_in))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat_interleave(rep * n_x * n_y, 0)
        x, _ = self.att(x_in, x_att, x_att, key_padding_mask=key_padding_mask)

        n_batch, _, n_y, n_x = x_cond.shape
        x = torch.permute(x.unflatten(0, (n_batch, n_y, n_x)), (0, 4, 3, 1, 2)).select(
            2, 0
        )
        return x


class Satformer(RetrievalModel):
    """
    The Satformer architecture is a convolutional encoder-decoder model that
    treats sensors channels as a sequence.
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
        input_config = {
            name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
        }
        output_config = {
            name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
        }
        config = SatformerConfig.parse(
            input_configs=input_config,
            output_configs=output_config,
            arch_config=arch_config,
        )
        return cls(
            input_config=input_config, output_config=output_config, arch_config=config
        )

    def __init__(
        self,
        input_config: Dict[str, InputConfig],
        output_config: Dict[str, OutputConfig],
        arch_config: SatformerConfig,
    ):
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

        self.meta_data = {name: cfg.meta_data for name, cfg in input_config.items()}
        self.encoding_map = {name: cfg.encoding for name, cfg in input_config.items()}
        self.input_names = list(self.input_config)
        self.masks = {
            name: cfg.mask for name, cfg in input_config.items() if cfg.mask is not None
        }
        self.masks.update(
            {
                name: cfg.mask
                for name, cfg in output_config.items()
                if cfg.mask is not None
            }
        )

        perceivers = {}
        self.outputs = {}
        for output_name, output_cfg in output_config.items():
            self.outputs[output_name] = output_cfg.conditional
            if output_cfg.encoding is not None and output_cfg.encoding != "None":
                self.encoding_map[output_name] = output_cfg.encoding
            if output_cfg.encoding is None or output_cfg.encoding == "None":
                perceivers[output_name] = Perceiver(
                    arch_config.encoder_config.channels[-1],
                    n_heads=arch_config.n_heads_perceiver
                )
            else:
                channels = copy(arch_config.encoder_config.channels)
                channels[0] += len(self.input_names)
                channels = channels[::-1][:len(arch_config.decoder_config.channels) + 1]

                perceivers[output_name] = nn.ModuleList([
                    CondPerceiver(
                        arch_config.output_embed_dim,
                        enc_chans,
                        n_heads=arch_config.n_heads_perceiver
                    ) for enc_chans in channels
                ])

        self.perceivers = nn.ModuleDict(perceivers)
        self.token_length = len(self.input_names)

        n_chans = arch_config.encoder_config.channels[0]

        self.stems = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.stem_configs.items()}
        )
        self.encodings = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.encoding_configs.items()}
        )
        self.encoder = arch_config.encoder_config.compile()
        skip_connections = copy(self.encoder.skip_connections)
        self.enc_norms = nn.ModuleDict(
            {str(scl): LayerNormFirst(chans) for scl, chans in skip_connections.items()}
        )
        self.decoders = nn.ModuleDict(
            {
                name: arch_config.decoder_config.compile(
                    token_length=self.token_length,
                    skip_connections=cfg.conditional is not None
                    and cfg.conditional != "None",
                )
                for name, cfg in output_config.items()
            }
        )

        scales = arch_config.encoder_config.scales
        max_scale = max(scales)
        scales = arch_config.decoder_config.get_scales(max_scale)
        base_scale = min(scales)

        downsamplers = []
        downsampler_scales = []

        for scale in scales:
            f_d = (scale // base_scale).scale[1:]
            downsamplers.append(nn.AvgPool2d(f_d, f_d))
            downsampler_scales.append(scale)
        self.downsamplers = nn.ModuleList(downsamplers)
        self.downsampler_scales = downsampler_scales
        self.output_embed_dim = arch_config.output_embed_dim

        self.heads = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.head_configs.items()}
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:

        # Propagate input tensors through stems and encode meta data.
        sequence_elements = []
        attn_mask = []
        tokens = []
        masks = []
        token_index = 0
        output_slices = []
        seq_start = 0

        for name in self.input_names:
            if name in x:
                inpt = self.stems[name](x[name])
                meta_data = self.meta_data[name]
                # Add meta data encoding
                if meta_data is not None and meta_data != "None":
                    x_meta = x[meta_data]

                    # Downsample meta data if necessary
                    h_x, w_x = inpt.shape[-2:]
                    h_meta, w_meta = x_meta.shape[-2:]
                    if h_x != h_meta or w_x != w_meta:
                        if h_x < h_meta:
                            kernel = (1, h_meta // h_x, w_meta // w_x)
                            x_meta = nn.functional.avg_pool3d(x_meta, kernel_size=kernel, stride=kernel)
                        else:
                            fac = (1, h_x // h_meta, w_x // w_meta)
                            x_meta = nn.functional.upsample(x_meta, scale_factor=fac)

                    enc = self.encodings[self.encoding_map[name]](x_meta)
                    inpt = inpt + enc
                sequence_elements.append(inpt)
                attn_mask += [False] * inpt.shape[2]

                n_batch, n_chans_in, n_seq, n_y, n_x = inpt.shape
                token = token_index * torch.zeros(
                    (n_batch, self.token_length, n_seq, n_y, n_x),
                    device=inpt.device,
                    dtype=inpt.dtype,
                )
                token[:, token_index] = 1.0
                tokens.append(token)

                if name in self.masks:
                    masks.append(x[self.masks[name]])
                else:
                    masks.append(
                        torch.zeros(
                            (n_batch, n_seq), device=inpt.device, dtype=torch.bool
                        )
                    )

                seq_start += n_seq

            token_index += 1

        input_sequence = torch.cat(sequence_elements, 2)
        tokens = torch.cat(tokens, 2)
        mask = torch.cat(masks, 1)
        inpt = torch.cat((tokens, input_sequence), 1)


        # Drop invalid samples
        inds = torch.argsort(mask, dim=1)
        max_valid = (~mask).sum(dim=1).max()
        inds_all = inds[:, None, :, None, None].expand_as(inpt)
        inpt = torch.gather(inpt, 2, inds_all[:, :, :max_valid])
        mask = torch.gather(mask, 1, inds[:, :max_valid])

        encs = self.encoder(inpt, mask=mask)
        encs = {scl: self.enc_norms[str(scl)](enc) for scl, enc in encs.items()}

        outputs = {}
        for name, cond in self.outputs.items():
            if cond is not None and cond != "None":
                output = self.encodings[self.encoding_map[name]](x[cond])
                n_batch, _, n_seq = output.shape[:3]
                output = torch.permute(output, (0, 2, 1, 3, 4)).flatten(0, 1)
                output_scaled = {}
                for perc, downsample, scale in zip(
                        self.perceivers[name], self.downsamplers, self.downsampler_scales
                ):
                    output_scaled[scale] = perc(downsample(output), encs[scale], key_padding_mask=mask)
                #min_scale = max(self.downsampler_scales)
                #output_scaled[min_scale] = self.perceivers[name](
                #    output_scaled[min_scale], encs[min_scale], key_padding_mask=mask
                #)
            else:
                min_scale = max(list(encs.keys()))
                output = self.perceivers[name](encs[min_scale], key_padding_mask=mask)
                n_batch, _, n_seq = output.shape[:3]
                output = torch.permute(output, (0, 2, 1, 3, 4)).flatten(0, 1)
                output_scaled = output

            dec = self.decoders[name]
            head = self.heads[name]
            output = head(
                dec(
                    output_scaled,
                    stage_kwargs={
                        scl: {"x_in": tnsr.repeat_interleave(n_seq, 0)}
                        for scl, tnsr in encs.items()
                    },
                    mask=mask.repeat_interleave(n_seq, 0),
                )
            )
            outputs[name] = list(
                torch.unbind(torch.unflatten(output, 0, (n_batch, n_seq)), 1)
            )

        return outputs

    @property
    def output_names(self) -> List[str]:
        """
        Names of the outputs from this model.
        """
        return list(self.heads.keys())
