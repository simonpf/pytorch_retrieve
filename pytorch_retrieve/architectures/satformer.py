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
    get_config_attr
)

from pytorch_retrieve.modules.mlp import MLP
from pytorch_retrieve.modules.conv import blocks
from pytorch_retrieve.modules.conv.utils import Scale
from pytorch_retrieve.modules.input import StandardizationLayer
from pytorch_retrieve.modules.normalization import get_normalization_factory
from pytorch_retrieve.modules.activation import get_activation_factory
from pytorch_retrieve.tensors import MeanTensor
from pytorch_retrieve.modules.conv.upsampling import StepwiseBilinear
from pytorch_retrieve.modules.conv.encoders import Encoder
from pytorch_retrieve.modules.conv.decoders import Decoder
from pytorch_retrieve.modules.conv.stages import SequentialWKeywordsStage
from .encoder_decoder import StemConfig, HeadConfig, get_downsampling_factory


class FlipDims(nn.Module):
    def __init__(self, base_module: nn.Module):
        super().__init__()
        self.base_module = base_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        block_factory: Name of the block factory to use to create the
             convolution blocks in the encoder-decoder architecture.
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
        scale = Scale(1)
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
            block_factory = [blocks.Satformer(**args) for args in self.block_factory_args]
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
            stage_factory=stage_factory
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
    block_factory_args: Union[Dict[str, Any], List[Dict[str, Any]]] = None

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
                if len(block_factory) != len(channels) - 1 or len(block_factory_args) != len(channels) - 1:
                    raise RuntimeError(
                        "If 'block_factory' and 'block_factory_args' are provided as lists, they must "
                        "have the same length as the number of stages in the decoder."
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
            block_factory_args=block_factory_args,
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

        block_factory = blocks.Satformer()
        if isinstance(self.block_factory_args, list):
            block_factory = [blocks.Satformer(**args) for args in self.block_factory_args]
            stage_factory = [SequentialWKeywordsStage] * len(block_factory)
        else:
            block_factory = blocks.Satformer(self.block_factory_args)
            stage_factory = SequentialWKeywordsStage

        upsampling_factory = StepwiseBilinear()

        skip_connections = self.skip_connections
        skip_connections = {
            Scale(key): value for key, value in skip_connections.items()
        }

        return Decoder(
            channels=self.channels,
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
    channels_in: int
    channels_out: int
    depth: int
    normalize: str
    input_name: str
    activation_factory: str
    normalization_factory: str
    residual_connections: Optional[str] = None

    @classmethod
    def parse(
        cls, name: str, config_dict: Dict[str, object]
    ) -> "EncodingConfig":
        channels_in = get_config_attr(
            "channels_in", int, config_dict, f"architecture.encoding.{name}", required=True
        )
        channels_out = get_config_attr(
            "channels_out", int, config_dict, f"architecture.encoding.{name}", required=True
        )
        depth = get_config_attr(
            "depth", int, config_dict, f"architecture.encoding.{name}", required=True
        )
        normalize = get_config_attr(
            "normalize", str, config_dict, f"architecture.encoding.{name}", default=None,
        )
        input_name = get_config_attr(
            "input_name", str, config_dict, f"architecture.encoding.{name}", default=None,
        )
        if normalize is not None and input_name is None:
            raise ValueError(
                f"'normalize' attribute of encoder '{name}' is set to '{normalize}' but the 'input_name' property"
                "is not set."
            )

        activation_factory = get_config_attr(
            "activation_factory", str, config_dict, f"architecture.encoder.{name}", default="GELU"
        )
        normalization_factory = get_config_attr(
            "normalization_factory", str, config_dict, f"architecture.encoder.{name}", default="LayerNorm"
        )
        residual_connections = get_config_attr(
            "residual_connections", str, config_dict, f"architecture.encoder.{name}", default="simple"
        )
        return EncodingConfig(
            channels_in=channels_in,
            channels_out=channels_out,
            depth=depth,
            normalize=normalize,
            input_name=input_name,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory, residual_connections=residual_connections
        )

    def to_config_dict(self):
        dct = asdict(self)
        return dct

    def compile(self) -> MLP:
        """
        Compile encoding.
        """
        activation_factory = get_activation_factory(self.activation_factory)
        normalization_factory = get_normalization_factory(self.normalization_factory)
        blocks = []
        if self.normalize is not None and self.normalize != "none":
            blocks.append(
                StandardizationLayer(kind=self.normalize, name=self.input_name, n_features=self.channels_in)
            )
        blocks.append(
            FlipDims(MLP(
                in_channels=self.channels_in,
                out_channels=self.channels_out,
                n_layers=self.depth,
                activation_factory=activation_factory,
                normalization_factory=normalization_factory,
                residual_connections=self.residual_connections
            ))
        )
        return nn.Sequential(*blocks)



@dataclass
class SatformerConfig():

    token_length: int
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
        outputs = [name for name, cfg in output_configs.items()]
        token_length = len(input_configs) + len(outputs)

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

        encoder_config = get_config_attr("encoder", dict, arch_config, "architecture", required=True)
        encoder_config = EncoderConfig.parse(stem_configs, encoder_config, token_length)

        decoder_config = get_config_attr("decoder", dict, arch_config, "architecture", required=True)
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

        return SatformerConfig(
            token_length,
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



class Satformer(RetrievalModel):

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
            arch_config=arch_config
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
        self.masks = {name: cfg.mask for name, cfg in input_config.items() if cfg.mask is not None}
        self.masks.update({name: cfg.mask for name, cfg in output_config.items() if cfg.mask is not None})

        conditional_outputs = {}
        unconditional_outputs = []
        for output_name, output_cfg in output_config.items():
            if output_cfg.conditional is not None:
                conditional_outputs[output_name] = output_cfg.conditional
                if output_cfg.encoding is not None:
                    self.encoding_map[output_name] = output_cfg.encoding
            else:
                unconditional_outputs.append(output_name)

        self.conditional_outputs = conditional_outputs
        self.unconditional_outputs = unconditional_outputs
        self.token_length = (
            len(self.input_names) +
            len(conditional_outputs) +
            len(unconditional_outputs)
        )

        self.stems = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.stem_configs.items()}
        )
        self.encodings = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.encoding_configs.items()}
        )
        self.encoder = arch_config.encoder_config.compile()
        self.decoder = arch_config.decoder_config.compile()
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
                if meta_data is not None:
                    enc = self.encodings[self.encoding_map[name]](x[meta_data])
                    inpt = inpt + enc
                sequence_elements.append(inpt)
                attn_mask += [False] * inpt.shape[2]

                n_batch, n_chans_in, n_seq, n_y, n_x = inpt.shape
                token = token_index  * torch.zeros(
                    (n_batch, self.token_length, n_seq, n_y, n_x),
                    device=inpt.device,
                    dtype=inpt.dtype
                )
                token[:, token_index] = 1.0
                tokens.append(token)

                if name in self.masks:
                    masks.append(x[self.masks[name]])
                else:
                    masks.append(torch.zeros((n_batch, n_seq), device=inpt.device, dtype=torch.bool))

                seq_start += n_seq

            token_index += 1

        for name, cond in self.conditional_outputs.items():
            if cond in x:
                inpt = self.encodings[self.encoding_map[name]](x[cond])
                sequence_elements.append(inpt)
                attn_mask += [True] * inpt.shape[2]

                n_batch, _, n_seq, n_y, n_x = inpt.shape
                token = token_index  * torch.zeros(
                    (n_batch, self.token_length, n_seq, n_y, n_x),
                    device=inpt.device,
                    dtype=inpt.dtype
                )
                token[:, token_index] = 1.0
                tokens.append(token)

                if name in self.masks:
                    masks.append(x[self.masks[name]])
                else:
                    masks.append(torch.zeros((n_batch, n_seq), device=inpt.device, dtype=torch.bool))

                output_slices.append(slice(seq_start, seq_start + n_seq))
                seq_start += n_seq

            token_index += 1

        for name in self.unconditional_outputs:
            attn_mask += [True]
            inpt = torch.zeros(
                (n_batch, n_chans_in, 1, n_y, n_x),
                device=inpt.device,
                dtype=inpt.dtype,
            )
            sequence_elements.append(inpt)
            token = token_index  * torch.zeros(
                (n_batch, self.token_length, 1, n_y, n_x),
                device=inpt.device,
                dtype=inpt.dtype
            )
            token[:, token_index] = 1.0
            tokens.append(token)

            masks.append(torch.zeros((n_batch, 1), device=inpt.device, dtype=torch.bool))
            output_slices.append(slice(seq_start, seq_start + n_seq))

            seq_start += 1
            token_index += 1

        input_sequence = torch.cat(sequence_elements, 2)
        attn_mask = torch.tensor(attn_mask, device=input_sequence.device)[None]
        attn_mask = attn_mask.repeat_interleave(input_sequence.shape[2], 0)

        tokens = torch.cat(tokens, 2)
        mask = torch.cat(masks, 1)
        inpt = torch.cat((tokens, input_sequence), 1)

        output = self.decoder(self.encoder(inpt, mask=mask), mask=mask, attn_mask=attn_mask)
        seqs = {}
        for slc, output_name in zip(output_slices[:len(self.conditional_outputs)], self.conditional_outputs):
            seqs[output_name] = [self.heads[output_name](x_i) for x_i in torch.unbind(output[:, :, slc], 2)]

        for slc, output_name in zip(output_slices[len(self.conditional_outputs):], self.unconditional_outputs):
            seqs[output_name] = [self.heads[output_name](x_i) for x_i in torch.unbind(output[:, :, slc], 2)]

        return seqs


    @property
    def output_names(self) -> List[str]:
        """
        Names of the outputs from this model.
        """
        return list(self.heads.keys())
