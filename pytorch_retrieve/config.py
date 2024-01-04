"""
pytorch_retrieve.config
=======================

The 'pytorch_retrieve.config' module implements functionality for
 reading configuration files.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union

import lightning as L
from lightning.pytorch import strategies
import torch

import yaml
import toml


def read_config_file(path: Path) -> dict:
    """
    Read a configuration file and returns its contents as a 'dict'.

    Args:
        path: Path of the configuration file to read.

    Return:
        A dictionary containing the parsed configuration file.

    Raises:
        RuntimeError if the given path points to a non-existing file or
        if the file suffix is neither '.toml' nor '.yaml'.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            "The given path '{path}' does not point to an existing file."
        )

    suffix = path.suffix
    if not suffix in [".yaml", ".toml"]:
        raise RuntimeError(
            "Config files must be in '.yaml' or '.toml' format and the file "
            "suffix should be either '.yaml' or '.toml'."
        )

    if suffix == ".yaml":
        config = yaml.safe_load(open(path).read())
        return config

    config = toml.loads(open(path).read())
    return config


def get_config_attr(name, constr, config, what, default=None, required=False):
    """
    Get attribute from config dict or raise appropriate runtime error.

    Args:
        name: The name of the attribute to read from the config.
        constr: A constructor functional to convert the original attribute
            value the expected type.
        config: A dictionary containing the configuration.
        what: The name of the instance that is being configured.
        default: If default is not 'None', it will be returned if 'config' does
            not contain the key 'name' instead of raising a RuntimeError.
        required: If 'True', a RuntimeError will be raised if 'config' does not
            contain an entry 'name'.

    Return:
        If the key 'name' is present in 'config', returns the
        ``constr(config[name])``. Otherwise, 'default' is returned.
    """
    if required and name not in config:
        raise RuntimeError(
            f"Expected entry '{name}' in config '{what}' "
            f" but no such attribute is present."
        )
    if name not in config:
        return default
    try:
        attr = constr(config.get(name))
    except ValueError:
        raise RuntimeError(
            f"Error during parsing of attribute '{name}' of the config for "
            f"'{what}'. Could not create {constr.__name__} from "
            f" given value '{config[name]}."
        )
    return attr


@dataclass
class InputConfig:
    """
    Represents input data to the retrieval.

    Attributes:
        n_features: The number of features or spectral channels in the input
            data.
        scale: The scale of the input data. In multi-scale settings the scale
            defines the relative sizes of the inputs.
        normalize: String indicating whether and what kind of normalization
            the model should perform. Set to 'None' if input data is normalized
            by data loader or no normalization is required.
    """

    n_features: int
    scale: int = 1
    normalize: Optional[str] = None

    @classmethod
    def parse(cls, name, cfg):
        n_features = get_config_attr("n_features", int, cfg, f"input.{name}")
        scale = get_config_attr("scale", int, cfg, f"input.{name}", 1)
        normalize = get_config_attr("normalize", str, cfg, f"input.{name}", "none")
        if normalize == "none":
            normalize = None
        return InputConfig(n_features, scale=scale, normalize=normalize)

    def to_config_dict(self):
        """
        Return input represented as dictionary.
        """
        return asdict(self)


@dataclass
class OutputConfig:
    """
    Represents a retrieval output.
    """

    target: str
    kind: str
    shape: List[int]

    @classmethod
    def parse(cls, name, cfg):
        target = get_config_attr("target", str, cfg, f"output.{name}", name)
        kind = get_config_attr("kind", str, cfg, f"output.{name}")

        shape = cfg.get("shape", None)
        if shape is None:
            raise ValueError(
                f"Output {name} is missing the required 'shape' attribute."
            )
        if isinstance(shape, int):
            shape = (shape,)
        else:
            try:
                shape = tuple(shape)
            except ValueError:
                raise ValueError(
                    "Expected an array of integer values for the 'shape' "
                    f"of output {name} but got a value of type '{type(shape)}'"
                    "."
                )
        return OutputConfig(target, kind, shape)

    def to_config_dict(self):
        """
        Return input represented as dictionary.
        """
        return asdict(self)


@dataclass
class ComputeConfig:
    """
    A description of a training regime.
    """

    precision: str = "16-mixed"
    accelerator: str = "cuda"
    devices: Union[List[int]] = -1
    strategy: str = "ddp"

    @classmethod
    def parse(cls, cfg):
        precision = get_config_attr(
            "precision", str, cfg, f"compute config", "16-mixed"
        )
        accelerator = get_config_attr(
            "accelerator", str, cfg, f"compute config", "cuda"
        )
        devices = get_config_attr("devices", list, cfg, f"compute config", None)
        strategy = get_config_attr("strategy", str, cfg, f"compute config", "ddp")
        return ComputeConfig(
            precision=precision,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
        )

    def __init__(
        self, precision="16-mixed", accelerator=None, devices=None, strategy="auto"
    ):
        self.precision = precision

        if accelerator is None:
            if torch.cuda.is_available():
                accelerator = "cuda"
            else:
                accelerator = "cpu"
        self.accelerator = accelerator

        if devices is None:
            if self.accelerator == "cuda":
                devices = list(range(torch.cuda.device_count()))
            else:
                devices = torch.cpu.device_count()
        self.devices = devices

        if strategy is None:
            if self.accelerator == "cuda" and len(self.devices) > 1:
                strategy = "ddp"
        self.strategy = strategy

    def get_strategy(self):
        if self.strategy == "ddp":
            return strategies.DDPStrategy(find_unused_parameters=True)
        return self.strategy
