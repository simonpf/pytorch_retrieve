"""
pytorch_retrieve.mlp
====================

Defines the 'MLP' architecture for retrievals based on multi-layer perceptrons
(MLPs). These are typically retrievals that operate on single pixels
 independently taking only a single multi-spectral observation from one or
multiple sensors as input.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from pytorch_retrieve.config import get_config_attr
from pytorch_retrieve.modules.activation import get_activation_factory
from pytorch_retrieve.modules.normalization import get_normalization_factory
from pytorch_retrieve.modules import mlp
from pytorch_retrieve.modules.utils import ParamCount
from pytorch_retrieve.modules.stems import MLPStem
from pytorch_retrieve.modules.aggregation import MLPAggregator
from pytorch_retrieve.modules import output


@dataclass
class StemConfig:
    """
    Configuration attributes of the stems of an MLP architecture.
    """

    in_channels: int
    hidden_channels: int
    n_layers: int
    residual_connections: Optional[str] = None
    activation_factory: Callable[[], nn.Module] = nn.ReLU
    normalization_factory: Optional[Callable[[int], nn.Module]] = None
    masked: bool = False

    @staticmethod
    def parse(in_channels: int, config_dict: dict, exhaustive=False) -> "StemConfig":
        hidden_channels = get_config_attr(
            "hidden_channels",
            int,
            config_dict,
            "Stem",
            -1,
        )
        n_layers = get_config_attr("n_layers", int, config_dict, "Stem", 1)
        residual_connections = get_config_attr(
            "residual_connections", str, config_dict, "Stem", "none"
        )
        if residual_connections == "none":
            residual_connections = None

        activation_factory = get_config_attr(
            "activation_factory", str, config_dict, "Stem", "ReLU"
        )
        activation_factory = get_activation_factory(activation_factory)
        normalization_factory = get_config_attr(
            "normaliation_factory", str, config_dict, "Stem", "none"
        )
        normalization_factory = get_normalization_factory(normalization_factory)
        masked = get_config_attr("masked", bool, config_dict, "Stem", False)

        return StemConfig(
            in_channels,
            hidden_channels,
            n_layers,
            residual_connections=residual_connections,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
            masked=masked,
        )

    def compile(self, out_channels):
        if self.hidden_channels < 0:
            hidden_channels = out_channels
        else:
            hidden_channels = self.hidden_channels
        return MLPStem(
            self.in_channels,
            out_channels,
            self.n_layers,
            hidden_channels=hidden_channels,
            residual_connections=self.residual_connections,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            masked=self.masked,
        )


@dataclass
class BodyConfig:
    """
    Attributes of the body of an MLP
    """

    hidden_channels: int
    n_layers: int
    residual_connections: Optional[str] = None
    activation_factory: Callable[[], nn.Module] = nn.ReLU
    normalization_factory: Optional[Callable[[int], nn.Module]] = None
    masked: bool = False

    @staticmethod
    def parse(config_dict: dict, exhaustive=False) -> "Body":
        hidden_channels = get_config_attr(
            "hidden_channels", int, config_dict, "Body", None, required=True
        )
        n_layers = get_config_attr(
            "n_layers", int, config_dict, "Body", None, required=True
        )
        residual_connections = get_config_attr(
            "residual_connections", str, config_dict, "Body", "none"
        )
        if residual_connections == "none":
            residual_connections = None

        activation_factory = get_config_attr(
            "activation_factory", str, config_dict, "Body", "ReLU"
        )
        activation_factory = get_activation_factory(activation_factory)
        normalization_factory = get_config_attr(
            "normaliation_factory", str, config_dict, "Body", "none"
        )
        normalization_factory = get_normalization_factory(normalization_factory)
        masked = get_config_attr("masked", bool, config_dict, "Body", False)

        return BodyConfig(
            hidden_channels,
            n_layers,
            residual_connections=residual_connections,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
            masked=masked,
        )

    def compile(self):
        return mlp.MLP(
            self.hidden_channels,
            self.hidden_channels,
            self.n_layers,
            hidden_channels=self.hidden_channels,
            residual_connections=self.residual_connections,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            masked=self.masked,
            internal=True,
        )


@dataclass
class AggregatorConfig:
    """
    Dataclass for representing attributes of the aggregator module.
    """

    hidden_channels: int = -1
    n_layers: int = 1
    residual_connections: Optional[str] = None
    activation_factory: Callable[[], nn.Module] = nn.ReLU
    normalization_factory: Optional[Callable[[int], nn.Module]] = None
    masked: bool = False

    @staticmethod
    def parse(config_dict: dict, exhaustive=False) -> "AggregatorConfig":
        hidden_channels = get_config_attr(
            "hidden_channels", int, config_dict, "architecture.aggregator", -1
        )
        n_layers = get_config_attr(
            "n_layers", int, config_dict, "architecture.aggregator", 1
        )
        residual_connections = get_config_attr(
            "residual_connections", str, config_dict, "architecture.aggregator", "none"
        )
        if residual_connections == "none":
            residual_connections = None
        activation_factory = get_config_attr(
            "activation_factory", str, config_dict, "architecture.aggregator", "ReLU"
        )
        activation_factory = get_activation_factory(activation_factory)
        normalization_factory = get_config_attr(
            "normaliation_factory", str, config_dict, "architecture.aggregator", "none"
        )
        normalization_factory = get_normalization_factory(normalization_factory)
        masked = get_config_attr(
            "masked", bool, config_dict, "architecture.aggregator", False
        )

        return AggregatorConfig(
            hidden_channels,
            n_layers,
            residual_connections=residual_connections,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
            masked=masked,
        )

    def compile(self, inputs, out_channels):
        if self.hidden_channels < 0:
            hidden_channels = out_channels
        else:
            hidden_channels = self.hidden_channels

        return MLPAggregator(
            inputs,
            out_channels,
            self.n_layers,
            hidden_channels=hidden_channels,
            residual_connections=self.residual_connections,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            internal=True,
        )


@dataclass
class OutputConfig:
    """
    Dataclass for representing attributes of model heads.
    """

    out_channels: int
    n_layers: int
    shape: Tuple[int]
    residual_connections: Optional[str] = None
    activation_factory: Callable[[], nn.Module] = nn.ReLU
    normalization_factory: Optional[Callable[[int], nn.Module]] = None
    masked: bool = False

    @staticmethod
    def parse(config_dict: dict, exhaustive=False) -> "StemConfig":
        shape = get_config_attr("shape", list, config_dict, "Output", None)
        out_channels = np.prod(shape)

        n_layers = get_config_attr("n_layers", int, config_dict, "Stem", 1)
        residual_connections = get_config_attr(
            "residual_connections", str, config_dict, "Stem", "none"
        )
        if residual_connections == "none":
            residual_connections = None

        activation_factory = get_config_attr(
            "activation_factory", str, config_dict, "Stem", "ReLU"
        )
        activation_factory = get_activation_factory(activation_factory)
        normalization_factory = get_config_attr(
            "normaliation_factory", str, config_dict, "Stem", "none"
        )
        normalization_factory = get_normalization_factory(normalization_factory)
        masked = get_config_attr("masked", bool, config_dict, "Stem", False)

        return OutputConfig(
            out_channels,
            n_layers,
            shape,
            residual_connections=residual_connections,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
            masked=masked,
        )

    def compile(self, hidden_channels: int) -> nn.Module:
        """
        Compile output config into MLP head.

        The MLP head takes the output from the MLP body and transforms it
        into an output.
        """
        head = mlp.MLP(
            hidden_channels,
            self.out_channels,
            self.n_layers,
            hidden_channels=hidden_channels,
            residual_connections=self.residual_connections,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            masked=self.masked,
            output_shape=tuple(self.shape),
            internal=False,
        )
        return nn.Sequential(head, output.Mean())


class MLP(ParamCount, nn.Module):
    @classmethod
    def from_config_dict(cls, config):
        """
        Compile MLP architecture from configuration dictionary.

        Args:
            config: A configuration dictionary containing an 'architecture'
                 as well as 'input' and 'output' tables.

        Return:
            The compiled MLP architecture.
        """
        if not "architecture" in config:
            raise RuntimeError(
                "Model configuration needs to have an 'architecture' table "
                "at the top level."
            )
        arch_cfg = config["architecture"]

        body_cfg = get_config_attr("body", dict, arch_cfg, "architecture")
        body_cfg = BodyConfig.parse(body_cfg)
        hidden_channels = body_cfg.hidden_channels
        inpt_cfgs = config["input"]

        stems = None
        aggregator = None
        in_channels = None

        stem_cfgs = {}
        for key, inpt_cfg in inpt_cfgs.items():
            in_channels = get_config_attr("in_channels", int, inpt_cfg, f"input.{key}")
            if "stem" in inpt_cfg:
                stem_cfg = StemConfig.parse(in_channels, inpt_cfg["stem"])
            else:
                stem_cfg_dct = arch_cfg.get("stem", {})
                stem_cfg = StemConfig.parse(in_channels, stem_cfg_dct)
            stem_cfgs[key] = stem_cfg
        if len(stem_cfgs) == 1:
            stem_cfgs = next(iter(stem_cfgs.values()))

        agg_cfg = arch_cfg.get("aggregator", {})
        aggregator_cfg = AggregatorConfig.parse(agg_cfg)

        outputs = config["output"]
        output_cfgs = {}
        for key in outputs:
            output_cfgs[key] = OutputConfig.parse(outputs[key])

        return cls(stem_cfgs, body_cfg, output_cfgs, aggregator_cfg, config_dict=config)

    def __init__(
        self,
        stem_cfgs: Union[StemConfig, dict[str, StemConfig]],
        body_cfg: BodyConfig,
        output_cfgs: Union[OutputConfig, dict[str, OutputConfig]],
        aggregator_cfg: Optional[dict[str, AggregatorConfig]],
        config_dict: Optional[dict[str, object]] = None,
    ):
        """
        Compile an MLP retrieval model for a given configuration.

        Args:
            stem_cfgs: A single StemConfig data object or a dict mapping input
                names to corresponding StemConfig objects specifying the
                configurations of the stem modules every input.
            body_cfg: A BodyConfig data object specifying the configuration
                of the MLP body.
            output_cfgs: A single OutputConfig object or a dict mapping outputs
                names to outputs.
            aggregator_cfg: An AggregatorConfig object specifying the
                configuration of the aggregation module.
        """
        super().__init__()

        self.aggregator = None
        hidden_channels = body_cfg.hidden_channels
        if isinstance(stem_cfgs, dict):
            inputs = {name: cfg.in_channels for name, cfg in stem_cfgs.items()}
            self.stems = nn.ModuleDict(
                {name: cfg.compile() for name, cfg in stem_cfgs.items()}
            )
            if aggregator_cfg is None:
                aggregator_cfg = AggregatorConfig()
            self.aggregator = aggregator_cfg.compile(inputs, hidden_channels)
        else:
            self.stems = stem_cfgs.compile(hidden_channels)

        self.body = body_cfg.compile()
        if isinstance(output_cfgs, dict):
            self.outputs = nn.ModuleDict(
                {
                    key: output_cfg.compile(hidden_channels)
                    for key, output_cfg in output_cfgs.items()
                }
            )
        else:
            self.outputs = output_cfgs.compile()

        self.config_dict = config_dict

    @property
    def output_names(self) -> List[str]:
        """
        Names of the outputs from this model.
        """
        return list(self.outputs.keys())

    @property
    def default_optimizer(self):
        return AdamW(self.params())

    @property
    def default_lr_scheduler(self):
        return None

    def to_config_dict(self) -> Dict[str, object]:
        """
        Return configuration used to construct the EncoderDecoder model.

        Raises:
            RuntimeError if the model was not constructed from a configuration
            dict.
        """
        if self.config_dict is None:
            raise RuntimeError(
                "This EncoderDecoder was not constructed from a config dict "
                "and can thus not be serialized."
            )
        return self.config_dict

    def forward(self, inputs: Union[torch.Tensor, dict[str, torch.Tensor]]):
        if isinstance(self.stems, nn.ModuleDict):
            inputs = {
                key: self.aggregators[key](tensor) for key, tensor in inputs.items()
            }
            inputs = self.aggregator(inputs)
        else:
            inputs = self.stems(inputs)

        outputs = self.body(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if isinstance(self.outputs, nn.ModuleDict):
            return {key: head(outputs) for key, head in self.outputs.items()}
        return self.outputs(outputs)
