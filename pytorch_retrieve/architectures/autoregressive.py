"""
pytroch_retrieve.architectures.autoregressive
=============================================

Generic implementation of autoregressive forecast models.
"""
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List

import torch
from torch import nn

from . import encoder_decoder
from . import recurrent_encoder_decoder as recurrent
from .model import RetrievalModel
from pytorch_retrieve.config import (
    get_config_attr,
    InputConfig,
    OutputConfig
)
from pytorch_retrieve.modules.conv import decoders
from pytorch_retrieve.modules.normalization import LayerNormFirst



@dataclass
class EncoderConfig:
    """
    Represents the configuration of the (spatial) encoder of the autoregressive
    forecast model. The encoder takes sequences of input observations and encodes
    them into the latent model space.
    """
    latent_dim: int
    stem_configs: Dict[str, encoder_decoder.StemConfig]
    encoder_config: encoder_decoder.EncoderConfig
    decoder_config: encoder_decoder.DecoderConfig

    @classmethod
    def parse(
        cls,
        latent_dim: int,
        input_configs: Dict[str, InputConfig],
        arch_config: Dict[str, object],
    ) -> "EncoderConfig":
        """
        Parses the encoder config from config dictionaries.

        Args:
            latent_dim: The dimension of the latent model space.
            input_configs: A dictionary mapping input names to corresponding
                InputConfig object describing the corresponding input.
            arch_config: Dictionary holding the configuration of the encoder.

        EncoderConfig:
            The configuration described by the inputs represented as an
            EncoderConfig object.
        """

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
                stem_configs[name] = encoder_decoder.StemConfig.parse(
                    name, name, input_config, config_dict
                )

        else:
            stem_configs = {
                name: encoder_decoder.StemConfig.parse(
                    "stem", name, input_config, stem_config_dict
                )
                for name, input_config in input_configs.items()
            }

        encoder_config = get_config_attr("encoder", dict, arch_config, "architecture")
        encoder_config = encoder_decoder.EncoderConfig.parse(
            stem_configs, encoder_config
        )

        decoder_config = get_config_attr("decoder", dict, arch_config, "architecture")
        if len(decoder_config["channels"]) > 0:
            decoder_config["channels"][-1] = latent_dim
        decoder_config = encoder_decoder.DecoderConfig.parse(
            encoder_config, decoder_config
        )
        return EncoderConfig(
            latent_dim,
            stem_configs,
            encoder_config,
            decoder_config
        )

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Revert this config back into dict representation.
        """
        stem_configs = {
            name: cfg.to_config_dict() for name, cfg in self.stem_configs.items()
        }
        stem_configs["individual"] = True
        config = {
            "stem": stem_configs,
            "encoder": self.encoder_config.to_config_dict(),
            "decoder": self.decoder_config.to_config_dict()
        }
        return config


    def compile(self) -> "Encoder":
        """Compile propagator."""
        stems = {name: cfg.compile() for name, cfg in self.stem_configs.items()}
        encoder = self.encoder_config.compile()
        decoder = self.decoder_config.compile()
        return Encoder(stems, encoder, decoder)


class Encoder(nn.Module):
    """
    The encoder of the autoregressive model takes a sequence of input observations
    and encodes them into a corresponding sequence of model states.
    """
    def __init__(
            self,
            stems: Dict[str, nn.Module],
            encoder: nn.Module,
            decoder: nn.Module
    ):
        """
        Args:
            stems: A dictionary mapping input names to corresponding stems.
            encoder: A nn.Module implementing the encoder part of the encoder-decoder
                 architecture of this encoder.
            encoder: A nn.Module implementing the decoder part of the encoder-decoder
                 architecture of this encoder.
        """
        super().__init__()
        self.stems = nn.ModuleDict(stems)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode inputs.

        Args:
            x: A list of tensors to encoder.

        Return:
            A list of encoded tensors.
        """
        if not isinstance(x, dict):
            if len(self.stems) > 1:
                raise ValueError(
                    "The input is a single tensor but the architecture has more "
                    "than one stem. This requires the input to be a dictionary "
                    " mapping input names to corresponding data tensors."
                )
            name, stem = next(iter(self.stems.items()))
            seq_length = len(x)
            encs = {name: recurrent.forward(stem, x)}
        else:
            encs = {}
            for name, tensor in x.items():
                encs[name] = recurrent.forward(self.stems[name], tensor)
            seq_length = len(next(iter(x.values())))

        encs = [
            {name: tensors[ind] for name, tensors in encs.items()}
            for ind in range(seq_length)
        ]

        encs = recurrent.forward(self.encoder, encs)
        decs = recurrent.forward(self.decoder, encs)
        return decs


@dataclass
class TemporalEncoderConfig:
    """
    Represents the configuration of the temporal encoder of this autoregressive
    model.
    """
    kind: str
    n_inputs: int
    latent_dim: int
    order: int
    encoder_config: encoder_decoder.EncoderConfig
    decoder_config: encoder_decoder.DecoderConfig

    @classmethod
    def parse(cls, latent_dim: int, order: int, config_dict: Dict[str, Any]) -> "TemporalEncoderConfig":
        """
        Parse configuration dictionary into a TemporalEncoderConfig.

        Args:
            latent_dim: The dimensionality of the latent model space.
            config_dict: A dictionary defining the configuration of the
                temporal encoder.
        """
        kind = get_config_attr("kind", None, config_dict, "architecture.temporal_encoder", required=True)
        if kind.lower() == "direct":
            n_inputs = get_config_attr("n_inputs", None, config_dict, "architecture.temporal_encoder", required=True)
        else:
            n_inputs = 0

        stem_config = encoder_decoder.StemConfig(
            "latent", latent_dim, 1, "none", latent_dim, 0, 1, upsampling=1
        )
        encoder_dict = get_config_attr("encoder", None, config_dict, "architecture.temporal_encoder", required=True)
        encoder_config = recurrent.EncoderConfig.parse(
            {"latent": stem_config}, encoder_dict
        )

        decoder_dict = get_config_attr("decoder", None, config_dict, "architecture.temporal_encoder", required=True)
        if len(decoder_dict["channels"]) > 0:
            decoder_dict["channels"][-1] = latent_dim
        decoder_config = recurrent.DecoderConfig.parse(
            encoder_config, decoder_dict
        )
        return TemporalEncoderConfig(
            kind=kind,
            n_inputs=n_inputs,
            latent_dim=latent_dim,
            order=order,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Transform this TemporalEncoderConfig object back into dictionary
        representation.
        """
        config = {
            "kind": self.kind,
            "n_inputs": self.n_inputs,
            "encoder": self.encoder_config.to_config_dict(),
            "decoder": self.decoder_config.to_config_dict()
        }
        return config


    def compile(self) -> "TemporalEncoder":
        """Compile propagator."""
        if self.kind.lower() == "recurrent":
            encoder = self.encoder_config.compile()
            decoder = self.decoder_config.compile()
            return RecurrentTemporalEncoder(encoder, decoder)

        block_factory = encoder_decoder.get_block_factory(self.encoder_config.block_factory)
        block_factory = block_factory(**self.encoder_config.block_factory_args)
        return DirectTemporalEncoder(
            block_factory,
            self.n_inputs,
            self.encoder_config.channels,
            self.order,
        )



class RecurrentTemporalEncoder(nn.Module):
    """
    The temporal encoder takes a sequence of latent model states and encodes
    them into a sequence of updated model states that will be fed into the
    propagator.
    """
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module
    ):
        """
        Args:
            encoder: The encoder part of the temporal encoder.
            decoder: The decoder part of the temporal encoder.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode inputs.

        Args:
            x: A list of encoded input observations.

        Return:
            A list of encoded model states.
        """
        return self.decoder(self.encoder({"latent": x}))


class DirectTemporalEncoder(nn.Module):
    """
    The direct temporal encoder merges a sequence of encoded observations by direct
    application of a convolution block.
    """
    def __init__(
            self,
            block_factory: Callable[[int, int], nn.Module],
            n_inputs: int,
            channels: List[int],
            order: int
    ):
        super().__init__()
        encs = []
        for _ in range(order):
            blocks = []
            ch_in = channels[0]
            for ch_out in channels[1:]:
                blocks.append(
                    block_factory(ch_in, ch_out)
                )
                ch_in = ch_out
            encs.append(nn.Sequential(*blocks))

        self.blocks = nn.ModuleList(encs)

    def forward(self, x):
        x = torch.cat(x, 1)
        return [block(x) for block in self.blocks]



@dataclass
class PropagatorConfig:
    """
    The propagator config describes the propagator that propagates model
    states in time to obtain the model states for the following time step.
    """
    order: int
    latent_dim: int
    encoder_config: encoder_decoder.EncoderConfig
    decoder_config: encoder_decoder.DecoderConfig

    @classmethod
    def parse(cls, order, latent_dim, config_dict) -> "PropagatorConfig":
        """
        Parse a dictionary defining a propagator configuration into a
        PropagatorConfig object.

        Args:
            order: The order of the propagator, which is the number of input
                model states used to predict the following step.
            latent_dim: The dimensionality of the latent model space.
            config_dict: The dictionary defining the propagator configuration.

        Return:
            A PropagatorConfig object representing the provided configuration.
        """

        stem_config = encoder_decoder.StemConfig(
            "latent", latent_dim, (1, 1), "none", order * latent_dim, 0, 1, upsampling=1
        )
        encoder_dict = get_config_attr("encoder", None, config_dict, "architecture.propagator", required=True)
        encoder_dict["channels"][0] = order * latent_dim
        encoder_config = encoder_decoder.EncoderConfig.parse(
            {"latent": stem_config}, encoder_dict
        )

        decoder_dict = get_config_attr("decoder", None, config_dict, "architecture.propagator", required=True)
        decoder_dict["channels"][-1] = latent_dim
        decoder_config = encoder_decoder.DecoderConfig.parse(
            encoder_config, decoder_dict
        )
        return PropagatorConfig(
            order=order,
            latent_dim=latent_dim,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Transform this propagator configuration back into dictionary
        representation.
        """
        config = {
            "encoder": self.encoder_config.to_config_dict(),
            "decoder": self.decoder_config.to_config_dict()
        }
        return config

    def compile(self):
        """Compile propagator."""
        encoder = self.encoder_config.compile()
        decoder = self.decoder_config.compile()
        return Propagator(encoder, decoder)


class Propagator(nn.Module):
    """
    The propagator predicts a future latent state given n previous latent state,
    where we call n the order of the propagator.
    """
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module
    ):
        """
        Args:
            encoder: The encoder part of the propagator.
            decoder: The decoder part of the propagator.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform propagator step.

        Args:
            x: The n last latent model states stacked into a single
                tensor.

        Return:
            A tensor representing the next model step.
        """
        return self.decoder(self.encoder({"latent": x}))



@dataclass
class DecoderConfig:
    """
    The decoder of the autoregressive model transforms the data back
    from the latent model space into the desired outputs.
    """
    head_configs: Dict[str, encoder_decoder.HeadConfig]
    channels: List[int]
    stage_depths: List[int]
    upsampling_factors: List[int]
    block_factory: str = "BasicConv"
    block_factory_args: Dict[str, Any] = None
    upsampling_factory: str = "bilinear"
    aggregation_factory: str = "linear"
    kind: str = "standard"

    @classmethod
    def parse(cls, latent_dim, decoder_config_dict, output_configs):
        """
        Parse decoder config object from configuration dictionary.

        Args:
            latent_dim: The dimensionality of the latent space used by the
                model.
            config_dict: The 'decoder' section of the architecture configuration.

        Return:
            A DecoderConfig object representing the given configuration.
        """
        kind = get_config_attr("kind", str, decoder_config_dict, "architecture.decoder", "none")
        channels = get_config_attr(
            "channels", list, decoder_config_dict, "architecture.decoder"
        )
        channels.insert(0, latent_dim)

        stage_depths = get_config_attr(
            "stage_depths", list, decoder_config_dict, "architecture.decoder", required=True
        )
        default = [2] * len(stage_depths)
        upsampling_factors = get_config_attr(
            "upsampling_factors", list, decoder_config_dict, "architecture.decoder", default
        )

        block_factory = get_config_attr(
            "block_factory", str, decoder_config_dict, "architecture.decoder", "BasicConv"
        )
        block_factory_args = get_config_attr(
            "block_factory_args", dict, decoder_config_dict, "architecture.decoder", {}
        )
        upsampling_factory = get_config_attr(
            "upsampling_factory",
            str,
            decoder_config_dict,
            "architecture.decoder",
            "Bilinear",
        )
        aggregation_factory = get_config_attr(
            "aggregation_factory", str, decoder_config_dict, "architecture.decoder", "Linear"
        )

        head_config_dict = get_config_attr("head", dict, decoder_config_dict, "architecture.decoder", {})
        individual = head_config_dict.get("individual", True)
        if individual:
            head_configs = {}
            for name, output_config in output_configs.items():
                if name in head_config_dict:
                    config_dict = head_config_dict[name]
                else:
                    config_dict = head_config_dict.get("default", {})
                head_configs[name] = encoder_decoder.HeadConfig.parse(
                    channels[-1], output_config, name, config_dict
                )
        else:
            head_configs = {
                name: encoder_decoder.HeadConfig.parse(
                    channels[-1], output_config, "head", head_config_dict
                )
                for name, output_config in output_configs.items()
            }

        return DecoderConfig(
            head_configs=head_configs,
            channels=channels,
            stage_depths=stage_depths,
            upsampling_factors=upsampling_factors,
            block_factory=block_factory,
            block_factory_args=block_factory_args,
            upsampling_factory=upsampling_factory,
            aggregation_factory=aggregation_factory,
            kind=kind,
        )

    def to_config_dict(self) -> Dict[str, object]:
        """
        Convert configuration object to dict representation suitable for
        serialization.
        """
        head_configs = {
            name: cfg.to_config_dict() for name, cfg in self.head_configs.items()
        }
        head_configs["individual"] = True
        config = asdict(self)
        config["head"] = head_configs

        config["channels"] = config["channels"][1:]
        return config

    def compile(self) -> nn.Module:
        """
        Compile the decoder module defined by this configuration.
        """
        block_factory = encoder_decoder.get_block_factory(self.block_factory)(**self.block_factory_args)
        upsampling_factory = encoder_decoder.get_upsampling_factory(self.upsampling_factory)()
        aggregation_factory = encoder_decoder.get_aggregation_factory(self.aggregation_factory)

        decoder = decoders.Decoder(
            channels=self.channels,
            stage_depths=self.stage_depths,
            upsampling_factors=self.upsampling_factors,
            block_factory=block_factory,
            skip_connections=None,
            upsampler_factory=upsampling_factory,
        )
        heads = {name: config.compile() for name, config in self.head_configs.items()}
        return Decoder(decoder, heads)


class Decoder(nn.Module):
    """
    The decoder module converts the latent representation of the model back
    to the desired variables.
    """
    def __init__(
            self,
            decoder: nn.Module,
            heads: Dict[str, nn.Module]
    ):
        """
        Args:
            decoder: A torch.nn.Module implementing the shared part of the
                decoder.
            heads: A dictionary mapping output names to curresponding output
                modules. The modules are applied to the output of 'decoder' to
                produce the actual model output.
        """
        super().__init__()
        self.decoder = decoder
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: A state from the latent model space.

        Return:
            A dictionary mapping output names to corresponding prediction tensors.
        """
        decs = self.decoder(x)
        output = {name: head(decs) for name, head in self.heads.items()}
        return output

@dataclass
class AutoregressiveConfig:
    """
    Represents the configuration of an autoregressive model.
    """
    time_step: int
    latent_dim: int
    order: int
    encoder_config: EncoderConfig
    temporal_encoder_config: TemporalEncoderConfig
    propagator_config: PropagatorConfig
    decoder_config: DecoderConfig

    @classmethod
    def parse(
        cls,
        input_configs: Dict[str, InputConfig],
        output_configs: Dict[str, OutputConfig],
        arch_config: Dict[str, object],
    ):
        """
        Parse a given architecture configuration into an AutoregressiveConfig
        object.

        Args:
            input_configs: A dictionary mapping input names to corresponding
                InputConfig objects describing each input of the model.
            output_configs: A dictionary mapping output names to corresponding
                OutputConfig object describing each model output.

        Return: An AutoregressiveConfig object describing the given configuration. """
        time_step = get_config_attr("time_step", int, arch_config, "architecture", required=True)
        latent_dim = get_config_attr("latent_dim", int, arch_config, "architecture", required=True)
        order = get_config_attr("order", int, arch_config, "architecture", required=True)

        encoder_cfg = get_config_attr("encoder", dict, arch_config, "architecture", required=True)
        encoder_cfg = EncoderConfig.parse(latent_dim, input_configs, encoder_cfg)

        temporal_encoder_cfg = get_config_attr("temporal_encoder", dict, arch_config, "architecture", required=True)
        temporal_encoder_cfg = TemporalEncoderConfig.parse(latent_dim, order,  temporal_encoder_cfg)

        propagator_cfg = get_config_attr("propagator", dict, arch_config, "architecture", required=True)
        propagator_cfg = PropagatorConfig.parse(order, latent_dim,  propagator_cfg)

        decoder_cfg = get_config_attr("decoder", dict, arch_config, "architecture", required=True)
        decoder_cfg = DecoderConfig.parse(latent_dim, decoder_cfg, output_configs)

        return AutoregressiveConfig(
            time_step=time_step,
            latent_dim=latent_dim,
            order=order,
            encoder_config=encoder_cfg,
            temporal_encoder_config=temporal_encoder_cfg,
            propagator_config=propagator_cfg,
            decoder_config=decoder_cfg
        )

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Transform this TemporalEncoderConfig object back into dictionary
        representation.
        """
        config = {
            "name": "Autoregressive",
            "time_step": self.time_step,
            "latent_dim": self.latent_dim,
            "order": self.order,
            "encoder": self.encoder_config.to_config_dict(),
            "temporal_encoder": self.temporal_encoder_config.to_config_dict(),
            "propagator": self.propagator_config.to_config_dict(),
            "decoder": self.decoder_config.to_config_dict()
        }
        return config

    def compile(self) -> "Autoregressive":
        """
        Compile configuration into an 'Autoregressive' model.
        """
        encoder = self.encoder_config.compile()
        temporal_encoder = self.temporal_encoder_config.compile()
        propagator = self.propagator_config.compile()
        decoder = self.decoder_config.compile()
        return Autoregressive(
            self.time_step,
            self.order,
            encoder,
            temporal_encoder,
            propagator,
            decoder
        )

class Autoregressive(RetrievalModel):
    """
    A forecast model that performs forecasts of future outputs from a sequence
    of past and potentially unrelated observations using an autoregressive approach.

    The 'Autoregressive' model maps a sequence of past observations
    [x_{t=-i}, x_{t=0}] to a sequence of future outputs [y_{t=1}, ..., y_{t=m}].
    """

    @classmethod
    def from_config_dict(cls, config_dict):
        input_config = get_config_attr(
            "input", dict, config_dict, "model config", required=True
        )
        input_config = {
            name: InputConfig.parse(name, cfg) for name, cfg in input_config.items()
        }
        output_config = get_config_attr(
            "output", dict, config_dict, "model config", required=True
        )
        output_config = {
            name: OutputConfig.parse(name, cfg) for name, cfg in output_config.items()
        }
        arch_config = get_config_attr("architecture", dict, config_dict, "model config")
        config = AutoregressiveConfig.parse(input_config, output_config, arch_config)
        return Autoregressive(input_config=input_config,
            output_config=output_config,
            arch_config=config
        )
        return config.compile()


    def __init__(
            self,
            input_config: Dict[str, InputConfig],
            output_config: Dict[str, OutputConfig],
            arch_config: Dict[str, AutoregressiveConfig]
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
        self.time_step = arch_config.time_step
        self.order = arch_config.order

        encoder = arch_config.encoder_config.compile()
        temporal_encoder = arch_config.temporal_encoder_config.compile()
        propagator = arch_config.propagator_config.compile()
        decoder = arch_config.decoder_config.compile()
        self.encoder = encoder
        self.temporal_encoder = temporal_encoder
        self.propagator = propagator
        self.decoder = decoder

    @property
    def output_names(self) -> List[str]:
        """
        Names of the outputs from this model.
        """
        return list(self.decoder.heads.keys())

    def forward(self, x):

        lead_times = x.get("lead_time")[0]
        x = {name: tensors for name, tensors in x.items() if name != "lead_time"}
        encs = self.encoder(x)
        encs = self.temporal_encoder(encs)

        steps = [lead_time // self.time_step for lead_time in lead_times]
        max_steps = max(steps)

        encs = encs[-self.order:]
        preds = {}
        for step in range(max_steps):
            propd = self.propagator(torch.cat(encs, 1))
            if (step + 1) in steps:
                pred = self.decoder(propd)
                for key, tensor in pred.items():
                    preds.setdefault(key, []).append(tensor)
            encs = encs[1:] + [propd]

        return preds
