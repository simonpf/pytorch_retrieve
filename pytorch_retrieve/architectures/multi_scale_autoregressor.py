"""
pytorch_retrieve.architecture.multi_scale_autoregressor
=======================================================

Implements the MultiScaleRegressor architecture for the simultaneous retrieval and
forecasts of physical quantities from satellite observations.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Union
from math import ceil

import torch
from torch import nn

from .model import RetrievalModel
from .encoder_decoder import (
    get_block_factory,
    EncoderConfig,
    Encoder,
    DecoderConfig,
    Decoder,
    HeadConfig,
    StemConfig
)
from pytorch_retrieve.config import get_config_attr
from pytorch_retrieve.config import InputConfig, OutputConfig, read_config_file
from pytorch_retrieve.modules.conv.decoders import MultiScalePropagator
from pytorch_retrieve.utils import update_recursive


@dataclass
class PropagatorConfig:
    stage_depths: List[int]
    block_factory: str
    block_factory_args: Dict[str, Any]
    order: int = 2
    residual: bool = True

    @classmethod
    def parse(
        cls,
        config_dict: Dict[str, object]
    ) -> "PropagatorConfig":
        stage_depths = get_config_attr(
            "stage_depths",
            list,
            config_dict,
            "architecture.propagator",
        )
        block_factory = get_config_attr(
            "block_factory",
            None,
            config_dict,
            "architecture.propagator",
            default="BasicConv3d"
        )
        block_factory_args = get_config_attr(
            "block_factory_args",
            None,
            config_dict,
            "architecture.propagator",
            default={}
        )
        order = get_config_attr(
            "order",
            int,
            config_dict,
            "architecture.propagator",
            default=2
        )
        residual = get_config_attr(
            "residual",
            bool,
            config_dict,
            "architecture.propagator",
            default=True
        )
        return PropagatorConfig(
            stage_depths=stage_depths,
            block_factory=block_factory,
            block_factory_args=block_factory_args,
            order=order,
            residual=residual
        )

    def compile(self, encoder: Encoder, decoder: Decoder) -> MultiScalePropagator:
        """
        Compile propagator.

        Args:
            encoder: The encoder of the multi-scale autoregressor model.
            decoder: The decoder  of the multi-scale autoregressor model.

        Return:
            The multi-scale propagator for the autoregressor model.
        """

        skip_connections = encoder.skip_connections
        inputs = decoder.multi_scale_outputs
        inputs[decoder.base_scale] = skip_connections[decoder.base_scale]

        if isinstance(self.block_factory, list):
            block_factory_args = self.block_factory_args
            if not isinstance(block_factory_args, list):
                block_factory_args = [block_factory_args] * len(self.block_factory)
            block_factory = [
                get_block_factory(b_fac)(**b_fac_args)
                for b_fac, b_fac_args in zip(self.block_factory, block_factory_args)
            ]
        else:
            block_factory = get_block_factory(self.block_factory)(**self.block_factory_args)

        return MultiScalePropagator(
            inputs,
            self.stage_depths,
            block_factory,
            base_scale=decoder.base_scale,
            residual=self.residual,
            order=self.order
        )

    def to_config_dict(self) -> Dict[str, Any]:
        dct = asdict(self)
        if isinstance(dct["block_factory_args"], list):
            args = dct["block_factory_args"]
            args = [dict(arg) for arg in args]
            dct["block_factory_args"] = args
        elif type(dct["block_factory_args"]) != dict:
            args = dct["block_factory_args"]
            dct["block_factory_args"] = dict(args)
        return dct


@dataclass
class MultiScaleAutoregressorConfig:
    """
    Dataclass representing the configuration of a multi-scale autoregressor.
    """
    time_step: int
    retrieval: bool
    stem_configs: Dict[str, StemConfig]
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig
    propagator_config: PropagatorConfig
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

        propagator_config = get_config_attr("propagator", dict, arch_config, "architecture")
        propagator_config = PropagatorConfig.parse(propagator_config)

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

        time_step = get_config_attr("time_step", int, arch_config, "architecture", required=True)
        retrieval = get_config_attr("retrieval", bool, arch_config, "architecture", default=True)

        return MultiScaleAutoregressorConfig(
            time_step=time_step,
            retrieval=retrieval,
            stem_configs=stem_configs,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            propagator_config=propagator_config,
            head_configs=head_configs,
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

        dct = {
            "name": "MultiScaleAutoregressor",
            "time_step": self.time_step,
            "retrieval": self.retrieval,
            "stem": stem_configs,
            "encoder": self.encoder_config.to_config_dict(),
            "decoder": self.decoder_config.to_config_dict(),
            "propagator": self.propagator_config.to_config_dict(),
            "head": head_configs,
        }
        return dct


class MultiScaleAutoregressor(RetrievalModel):

    @classmethod
    def from_config_dict(cls, config_dict):
        """
        Create multi-scale autoregressor model from a configuration dictionary.

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

        config = MultiScaleAutoregressorConfig.parse(input_config, output_config, arch_config)
        return cls(
            input_config=input_config, output_config=output_config, arch_config=config
        )


    def __init__(
            self,
            input_config: Dict[str, InputConfig],
            output_config: Dict[str, OutputConfig],
            arch_config: MultiScaleAutoregressorConfig
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

        self.stems = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.stem_configs.items()}
        )
        self.encoder = arch_config.encoder_config.compile()
        self.decoder = arch_config.decoder_config.compile()
        self.output_decoder = arch_config.decoder_config.compile()
        self.propagator = arch_config.propagator_config.compile(self.encoder, self.decoder)
        self.heads = nn.ModuleDict(
            {name: cfg.compile() for name, cfg in arch_config.head_configs.items()}
        )
        self.scales = [max(list(self.encoder.skip_connections.keys()))] + list(self.decoder.multi_scale_outputs.keys())
        self.base_scale = max(self.scales)

        self.time_step = arch_config.time_step
        self.retrieval = arch_config.retrieval

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
                "This MultiScaleAutoregressor architecture was not constructed from a config dict "
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
        decoded = self.decoder.forward_multi_scale_output(encs)

        if "lead_time" in x:
            lead_times = x["lead_time"][0]
            steps = ceil(lead_times.max() / self.time_step)
        else:
            lead_times =[]
            steps = 0

        decoded[self.decoder.base_scale] = encs[self.decoder.base_scale]
        predicted = self.propagator(decoded, steps)
        predicted = {
            scl: torch.stack(tensors, 2) for scl, tensors in predicted.items()
        }

        if self.retrieval:
            inpt = {
                scl: torch.cat((decoded[scl], predicted[scl]), 2) for scl in predicted
            }
            results = self.output_decoder(inpt)
        else:
            inpt = {}
            base_scale = max(self.scales)
            for scl in self.scales:
                extra = base_scale[0] // scl[0]
                inpt[scl] = torch.cat((decoded[scl][:, :, -extra:], predicted[scl]), 2)
            results = self.output_decoder(inpt)

        lead_time = self.time_step

        if not self.retrieval:
            results = results[:, :, extra:]

        results = torch.unbind(results, 2)

        outputs = {}
        for result in results:
            for name, head in self.heads.items():
                outputs.setdefault(name, []).append(head(result))
        return outputs
