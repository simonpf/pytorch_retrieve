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

from pytorch_retrieve.config import get_config_attr, InputConfig, OutputConfig
from pytorch_retrieve.modules.activation import get_activation_factory
from pytorch_retrieve.modules.normalization import get_normalization_factory
from pytorch_retrieve.modules.input import StandardizationLayer
from pytorch_retrieve.modules import mlp
from pytorch_retrieve.modules.utils import ParamCount
from pytorch_retrieve.modules.stems import MLPStem
from pytorch_retrieve.modules.aggregation import MLPAggregator
from pytorch_retrieve.modules import output
from pytorch_retrieve.architectures.model import RetrievalModel


@dataclass
class StemConfig:
    """
    Configuration attributes of the stems of an MLP architecture.
    """

    input_name: str
    in_channels: int
    hidden_channels: int
    n_layers: int
    residual_connections: Optional[str] = None
    activation_factory: Callable[[], nn.Module] = nn.ReLU
    normalization_factory: Optional[Callable[[int], nn.Module]] = None
    masked: bool = False
    normalize: Optional[str] = None

    @classmethod
    def parse(
        cls,
        name: str,
        input_name: str,
        input_config: InputConfig,
        config_dict: dict,
        exhaustive=False,
    ) -> "StemConfig":
        """
        Parse configuration of MLP stem.

        Args:
            name: Name of the stem.
            input_name: Name of the input variable.
            input_config: The configuration of the input corresponding to this stem.
            config_dict: The configuration dictionary specifying the configuration of this stem.
        """
        in_channels = input_config.n_features
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
        normalize = input_config.normalize

        return StemConfig(
            input_name,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            residual_connections=residual_connections,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
            masked=masked,
            normalize=normalize,
        )

    def compile(self, out_channels) -> nn.Module:
        """
        Compile stem into a network module.

        Args:
             out_channels: The number of channels in the stem output.

        Return:
             A torch.nn.Module implementing the stem specified by the configuration object.
        """
        blocks = []
        if self.normalize != "none":
            blocks.append(
                StandardizationLayer(
                    self.input_name, self.in_channels, kind=self.normalize
                )
            )

        if self.hidden_channels < 0:
            hidden_channels = out_channels
        else:
            hidden_channels = self.hidden_channels

        blocks.append(
            MLPStem(
                self.in_channels,
                out_channels,
                self.n_layers,
                hidden_channels=hidden_channels,
                residual_connections=self.residual_connections,
                activation_factory=self.activation_factory,
                normalization_factory=self.normalization_factory,
                masked=self.masked,
            )
        )
        return nn.Sequential(*blocks)


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
class HeadConfig:
    """
    Dataclass for representing attributes of model heads.
    """

    output_config: OutputConfig
    in_channels: int
    n_layers: int
    residual_connections: Optional[str] = None
    activation_factory: Callable[[], nn.Module] = nn.ReLU
    normalization_factory: Optional[Callable[[int], nn.Module]] = None
    masked: bool = False

    @classmethod
    def parse(cls, in_channels, output_config, name, config_dict) -> "StemConfig":

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

        return HeadConfig(
            output_config=output_config,
            in_channels=in_channels,
            n_layers=n_layers,
            residual_connections=residual_connections,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
            masked=masked,
        )

    def compile(self) -> nn.Module:
        """
        Compile output config into MLP head.

        The MLP head takes the output from the MLP body and transforms it
        into an output.
        """
        output_shape = self.output_config.get_output_shape()
        out_channels = np.prod(output_shape)
        head = mlp.MLP(
            self.in_channels,
            out_channels,
            self.n_layers,
            hidden_channels=self.in_channels,
            residual_connections=self.residual_connections,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            masked=self.masked,
            output_shape=output_shape,
            internal=False,
        )
        output_layer = self.output_config.get_output_layer()
        return nn.Sequential(head, output_layer)


class MLP(RetrievalModel):
    @classmethod
    def from_config_dict(cls, config_dict):
        """
        Compile MLP architecture from configuration dictionary.

        Args:
            config_dict: A configuration dictionary containing an 'architecture'
                 as well as 'input' and 'output' tables.

        Return:
            The compiled MLP architecture.
        """
        if not "architecture" in config_dict:
            raise RuntimeError(
                "Model configuration needs to have an 'architecture' table "
                "at the top level."
            )
        arch_config = config_dict["architecture"]

        input_configs = get_config_attr(
            "input", dict, config_dict, "model config", required=True
        )
        input_configs = {
            name: InputConfig.parse(name, config)
            for name, config in input_configs.items()
        }
        output_configs = get_config_attr(
            "output", dict, config_dict, "model config", required=True
        )
        output_configs = {
            name: OutputConfig.parse(name, config)
            for name, config in output_configs.items()
        }

        body_config = get_config_attr(
            "body", dict, arch_config, "architecture", required=True
        )
        body_config = BodyConfig.parse(body_config)
        hidden_channels = body_config.hidden_channels

        stems = None
        aggregator = None
        in_channels = None

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

        head_config_dict = arch_config.get("head", {})
        if "base" in head_config_dict:
            head_configs = {}
            for name, output_config in output_configs.items():
                if name in head_config_dict:
                    config_dict = head_config_dict[name]
                else:
                    config_dict = head_config_dict["base"]
                head_configs[name] = HeadConfig.parse(
                    hidden_channels, output_config, name, config_dict
                )
        else:
            head_configs = {
                name: HeadConfig.parse(
                    hidden_channels, output_config, "head", head_config_dict
                )
                for name, output_config in output_configs.items()
            }

        agg_config = arch_config.get("aggregator", {})
        aggregator_config = AggregatorConfig.parse(agg_config)

        config_dict={
            "input": {
                name: cfg.to_config_dict() for name, cfg in input_configs.items()
            },
            "output": {
                name: cfg.to_config_dict() for name, cfg in output_configs.items()
            },
            "architecture": arch_config
        }

        return cls(
            stem_configs,
            body_config,
            head_configs,
            aggregator_config,
            config_dict=config_dict,
        )

    def __init__(
        self,
        stem_cfgs: Union[StemConfig, dict[str, StemConfig]],
        body_cfg: BodyConfig,
        head_cfgs: Union[HeadConfig, dict[str, HeadConfig]],
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
        super().__init__(config_dict=config_dict)

        self.aggregator = None
        hidden_channels = body_cfg.hidden_channels
        if isinstance(stem_cfgs, dict):
            inputs = {name: hidden_channels for name, cfg in stem_cfgs.items()}
            self.stems = nn.ModuleDict(
                {name: cfg.compile(hidden_channels) for name, cfg in stem_cfgs.items()}
            )
            if aggregator_cfg is None:
                aggregator_cfg = AggregatorConfig()
            self.aggregator = aggregator_cfg.compile(inputs, hidden_channels)
        else:
            self.stems = stem_cfgs.compile()

        self.body = body_cfg.compile()
        if isinstance(head_cfgs, dict):
            self.outputs = nn.ModuleDict(
                {key: output_cfg.compile() for key, output_cfg in head_cfgs.items()}
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

        if not isinstance(inputs, dict):
            if len(self.stems) > 1:
                raise ValueError(
                    "The input is a single tensor but the architecture has more "
                    "than one stem. This requires the input to be a dictionary "
                    " mapping input names to corresponding data tensors."
                )
            name, stem = next(iter(self.stems.items()))
            inputs = stem(inputs)
        else:
            inputs = {key: stem(inputs[key]) for key, stem in self.stems.items()}
            inputs = self.aggregator(inputs)

        outputs = self.body(inputs)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if isinstance(self.outputs, nn.ModuleDict):
            return {key: head(outputs) for key, head in self.outputs.items()}
        return self.outputs(outputs)
