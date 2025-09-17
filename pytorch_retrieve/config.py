"""
pytorch_retrieve.config
=======================

The 'pytorch_retrieve.config' module implements functionality for
 reading configuration files.
"""
from copy import copy
from dataclasses import dataclass, asdict
from functools import partial
import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
from lightning.pytorch import strategies
import numpy as np
import toml
from torch import nn
import torch
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import yaml

import pytorch_retrieve.modules.transformations
from pytorch_retrieve.modules import output


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


def replace_environment_variables(attr: Any) -> Any:
    """
    Replaces environment variables in string if string starts with 'ENV::'.

    Args:
        attr: Any attribute extracted from a configuration file.

    Return:
        If 'attr' is a string and startswith "ENV::" then the remainder from
        the string is extracted and formatted using current environment variables
        as keys.
    """
    if isinstance(attr, str) and attr.startswith("ENV::"):
        attr = attr[5:].format(**os.environ)
    return attr


def get_config_attr(name, constr, config, what, default=None, required=False):
    """
    Get attribute from config dict or raise appropriate runtime error.

    This function also provides special treatment of strings. If an attribute
    is a string and starts with 'ENV::', the remainder of the string is
    extracted and formatted using currently set environment variables.
    For example, 'ENV::{PATH}' will be replaced to '/home/user' if the
    'PATH' environment variable is set to '/home/user'.

    Args:
        name: The name of the attribute to read from the config.
        constr: A constructor functional to convert the original attribute
            value to the expected type.
        config: A dictionary containing the configuration.
        what: The name of the instance that is being configured. Will be used
            in the error message if the attribute isn't present or can't be
            constructed.
        default: If default is not 'None', it will be returned if 'config' does
            not contain the key 'name' instead of raising a RuntimeError.
        required: If 'True', a RuntimeError will be raised if 'config' does not
            contain an entry 'name'.

    Return:
        If the key 'name' is present in 'config', returns the
        ``constr(config[name])``. Otherwise, 'default' is returned.

    Raises:
        - RuntimeError if 'required' is True but the attribute is not
          present in config.
        - RuntimeError if the application of 'constr' fails.

    """
    if required and name not in config:
        raise RuntimeError(
            f"Expected entry '{name}' in config '{what}' "
            f" but no such attribute is present."
        )
    if name not in config:
        return default
    try:
        attr = replace_environment_variables(config.get(name))
        if constr is not None:
            attr = constr(attr)
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
        meta_data: Optional name of the input that encodes meta data corresponding to this
            input.
        encoding: Optional name of an encoding module to use to encode the meta data.
        mask: Name of an optional mask input that should be used to mask invalid values.
    """

    n_features: int
    scale: Tuple[int] = (1, 1)
    normalize: str = "none"
    meta_data: Optional[str] = None
    encoding: Optional[str] = None
    mask: Optional[str] = None

    @classmethod
    def parse(cls, name, cfg):
        n_features = get_config_attr("n_features", int, cfg, f"input.{name}")
        scale = get_config_attr("scale", None, cfg, f"input.{name}", 1)
        if isinstance(scale, int):
            scale = (scale, scale)
        normalize = get_config_attr("normalize", str, cfg, f"input.{name}", "none")
        meta_data = get_config_attr("meta_data", str, cfg, f"input.{name}", None)
        encoding = get_config_attr("encoding", str, cfg, f"input.{name}", None)
        mask = get_config_attr("mask", str, cfg, f"input.{name}", None)

        return InputConfig(
            n_features,
            scale=scale,
            normalize=normalize,
            meta_data=meta_data,
            encoding=encoding,
            mask=mask,
        )

    def to_config_dict(self):
        """
        Return input represented as dictionary.
        """
        return asdict(self)


@dataclass
class OutputConfig:
    """
    Represents outputs from a retrieval model.

    Attributes:
        target: The name of the output.
        kind: The type of output (mean, quantiles, bins, classification)
        transformation: Optional transformation to apply to the outputs.
        transformation_args: Dictionary containing the parameters of the transformation.
        dimensions: The dimensions of the output.
        n_classes: The number of classes for a classification output.
        conditional: Name of a conditional input that this output is conditioned on.
        encoding: Name of an encoding module to use to encode the conditional input.
        mask: Name of a mask input that identifies output that should be ignored.
    """

    target: str
    kind: str
    shape: List[int]
    quantiles: Optional[Union[int, List[float]]] = None
    transformation: Optional[str] = None
    transformation_args: Dict[str, Any] = None
    dimensions: Optional[List[str]] = None
    n_classes: Optional[int] = None
    conditional: Optional[str] = None
    encoding: Optional[str] = None
    mask: Optional[str] = None

    @classmethod
    def parse(cls, name, cfg):
        target = get_config_attr("target", str, cfg, f"output.{name}", name)
        kind = get_config_attr("kind", str, cfg, f"output.{name}", required=True)
        shape = cfg.get("shape", 1)

        if isinstance(shape, int):
            scalar = True
            shape = (shape,)
        else:
            scalar = False

        if not isinstance(shape, int):
            try:
                shape = tuple(shape)
            except ValueError:
                raise ValueError(
                    "Expected an array of integer values for the 'shape' "
                    f"of output {name} but got a value of type '{type(shape)}'"
                    "."
                )

        transformation = get_config_attr("transformation", None, cfg, f"output.{name}")
        transformation_args = get_config_attr("transformation_args", dict, cfg, f"output.{name}", {})
        transformation_args = dict(transformation_args)
        dimensions = get_config_attr(
            "dimensions", None, cfg, f"output.{name}", default=None
        )
        if dimensions is None:
            if scalar:
                dimensions = []
            else:
                dimensions = [f"{name}_dim_{ind + 1}" for ind in range(len(shape))]

        quantiles = get_config_attr("quantiles", None, cfg, f"output.{name}")
        if kind == "Quantiles":
            if quantiles is None:
                raise ValueError(
                    f"Output {name} has kind 'Quantiles' but quantiles attribute is not set"
                )

        n_classes = get_config_attr("n_classes", None, cfg, f"output.{name}")
        if kind == "Classification":
            if n_classes is None:
                raise ValueError(
                    f"Output {name} has kind 'Classification' but n_classes attribute is not set"
                )
            n_classes = int(n_classes)

        conditional = get_config_attr("conditional", str, cfg, f"output.{name}", None)
        encoding = get_config_attr("encoding", str, cfg, f"output.{name}", None)
        mask = get_config_attr("mask", str, cfg, f"output.{name}", None)

        return OutputConfig(
            target=target,
            kind=kind,
            shape=shape,
            quantiles=quantiles,
            transformation=transformation,
            transformation_args=transformation_args,
            dimensions=dimensions,
            n_classes=n_classes,
            conditional=conditional,
            encoding=encoding,
            mask=mask,
        )

    @property
    def extra_dimensions(self) -> List[str]:
        """
        Return list of the names of 'extra' dimensions included due to the kind of output.
        """
        kind = self.kind
        if kind == "Mean":
            return []
        if kind == "Quantiles":
            return [f"tau_{self.target}"]
        if kind == "Detection":
            return []
        if kind == "RandomSample":
            return ["samples"]
        if kind == "Classification":
            return [f"{self.target}_probability"]


        raise RuntimeError(
            f"The output kind '{kind}' is currently not supported. Refer to "
            "the documentation of the pytorch_retrieve.modules.output module "
            "for available outputs."
        )

    def get_output_shape(self) -> Tuple[int]:
        """
        Determine shape of the network output.
        """
        kind = self.kind

        shape = self.shape
        if isinstance(shape, int):
            if shape == 1:
                shape = ()
            else:
                shape = (shape,)

        if kind == "Mean":
            return shape

        if kind == "Quantiles":
            quantiles = self.quantiles
            if isinstance(quantiles, int):
                return (quantiles,) + shape
            return (len(quantiles),) + shape

        if kind == "Detection":
            return shape

        if kind == "Classification":
            if self.n_classes is None:
                raise RuntimeError(
                    f"The output for target {self.target} has kind 'Classification' "
                    "but the 'n_classes' attribute is not set."
                )
            if sum(shape) == 1:
                return (self.n_classes,)
            return (self.n_classes,) + shape

        raise RuntimeError(
            f"The output kind '{kind}' is currently not supported. Refer to "
            "the documentation of the pytorch_retrieve.modules.output module "
            "for available outputs."
        )

    def get_output_dimensions(self) -> List[str]:
        """
        Return the dimensions of a tensor of the output excluding batch and spatial
        dimensions.
        """
        dims = self.dimensions
        if dims is None:
            dims = []
        return self.extra_dimensions + dims

    def get_output_coordinates(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary of output dimension names and corresponding coordinates.
        """
        if self.kind == "Mean":
            return {}
        if self.kind == "Quantiles":
            quantiles = self.quantiles
            if isinstance(quantiles, int):
                quantiles = np.linspace(0, 1, quantiles + 2)[1:-1]
            elif isinstance(quantiles, list):
                quantiles = np.array(quantiles)
            return {f"tau_{self.target}": quantiles}
        if self.kind == "RandomSample":
            return {}
        if self.kind == "Detection":
            return {}
        if self.kind == "Classification":
            return {
                f"{self.target}_classes":
                np.array([f"class_{i}" for i in range(self.n_classes)])
            }
        raise ValueError(f"Output kind {self.kind} is not supported.")

    def get_output_layer(self) -> nn.Module:
        """
        Get output layer for output.
        """
        transformation = self.transformation
        if transformation is not None:
            try:
                transformation = getattr(
                    pytorch_retrieve.modules.transformations, transformation
                )(**self.transformation_args, output_config=self)
            except AttributeError:
                raise ValueError(
                    f"The transformation {transformation} is not known. Please refere to the "
                    "'pytorch_retrieve.modules.transformations' module for available transformations."
                )

        kind = self.kind
        if kind == "Mean":
            return output.Mean(self.target, self.shape, transformation=transformation)
        if kind == "Quantiles":
            quantiles = self.quantiles
            if isinstance(quantiles, int):
                quantiles = np.linspace(0, 1, quantiles + 2)[1:-1]
            elif isinstance(quantiles, list):
                quantiles = np.array(quantiles)
            return output.Quantiles(
                self.target, self.shape, tau=quantiles, transformation=transformation
            )
        if kind == "Detection":
            return output.Detection(self.target, self.shape)
        if kind == "Classification":
            if self.n_classes is None:
                raise RuntimeError(
                    f"The output for target {self.target} has kind 'Classification' "
                    "but the 'n_classes' attribute is not set."
                )
            return output.Classification(self.target, self.shape)
        raise RuntimeError(
            f"The output kind '{kind}' is currently not supported. Refer to "
            "the documentation of the pytorch_retrieve.modules.output module "
            "for available outputs."
        )

    def to_config_dict(self):
        """
        Return input represented as dictionary.
        """
        dct = asdict(self)
        return dct


@dataclass
class ComputeConfig:
    """
    A description of a training regime.
    """

    precision: str = "16-mixed"
    accelerator: str = "cuda"
    devices: Union[List[int]] = -1
    n_nodes: int = 1
    strategy: str = "auto"
    use_distributed_sampler: bool = True
    sharding_strategy: str = "FULL_SHARD",
    device_mesh: Optional[List[int]] = None,
    transformer_layer_classes: Optional[List[str]] = None

    @classmethod
    def parse(cls, cfg):
        precision = get_config_attr(
            "precision", str, cfg, f"compute config", "16-mixed"
        )
        accelerator = get_config_attr("accelerator", str, cfg, f"compute config", None)
        devices = get_config_attr("devices", None, cfg, f"compute config", None)
        n_nodes = get_config_attr("n_nodes", int, cfg, f"compute config", 1)
        strategy = get_config_attr("strategy", str, cfg, f"compute config", "auto")
        use_distributed_sampler = get_config_attr("use_distributed_sampler", bool, cfg, f"compute config", True)
        sharding_strategy = get_config_attr("sharding_strategy", list, cfg, f"compute config", "FULL_SHARD")
        device_mesh = get_config_attr("device_mesh", list, cfg, f"compute config", None)
        transformer_layer_classes = get_config_attr("transformer_layer_classes", list, cfg, f"compute config", [])

        return ComputeConfig(
            precision=precision,
            accelerator=accelerator,
            devices=devices,
            n_nodes=n_nodes,
            strategy=strategy,
            use_distributed_sampler=use_distributed_sampler,
            sharding_strategy=sharding_strategy,
            device_mesh=device_mesh,
            transformer_layer_classes=transformer_layer_classes
        )

    def __init__(
        self,
        precision: str = "16-mixed",
        accelerator: Optional[str] = None,
        devices: Optional[List[int]] = None,
        n_nodes: int = 1,
        strategy: str = "auto",
        use_distributed_sampler: bool = True,
        sharding_strategy: str = "FULL_SHARD",
        device_mesh: Optional[List[int]] = None,
        transformer_layer_classes: Optional[List[str]] = None
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
                devices = 1
        self.devices = devices

        self.n_nodes = n_nodes

        if strategy is None:
            if self.accelerator == "cuda" and len(self.devices) > 1:
                strategy = "ddp"
        self.strategy = strategy

        self.use_distributed_sampler = use_distributed_sampler
        self.sharding_strategy = sharding_strategy
        self.device_mesh = device_mesh
        self.transformer_layer_classes = transformer_layer_classes


    def get_strategy(self) -> Any:
        """
        Get the compute strategy to pass to lightning.
        """
        if self.strategy == "ddp":
            return strategies.DDPStrategy(find_unused_parameters=True)
        elif self.strategy == "fsdp":
            import pytorch_retrieve

            all_modules = lambda module, recurse, nonwrapped_numel: True

            transformer_classes = []
            for name in self.transformer_layer_classes:
                *parts, cls = name.split(".")
                module = importlib.import_module(".".join(parts))
                transformer_classes.append(getattr(module, cls))

            auto_wrap = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=set(transformer_classes)
            )
            mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)

            return strategies.FSDPStrategy(
                sharding_strategy=self.sharding_strategy,
                auto_wrap_policy=auto_wrap,
                mixed_precision=mp,
                activation_checkpointing_policy=pytorch_retrieve.modules.conv.blocks.ALL,
                limit_all_gathers=True,
                forward_prefetch=True,
                use_orig_params=True,
                device_mesh=self.device_mesh
            )
        return self.strategy


@dataclass
class RetrievalOutputConfig:
    """
    Describes inference output to be calculated from the model predictions.
    """

    output_config: OutputConfig
    retrieval_output: str
    parameters: Optional[Dict[str, Any]]

    def __init__(
        self,
        output_config: OutputConfig,
        retrieval_output: str,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.output_config = output_config
        self.retrieval_output = retrieval_output
        self.parameters = parameters

        try:
            import pytorch_retrieve.retrieval_output

            output_class = getattr(pytorch_retrieve.retrieval_output, retrieval_output)
        except AttributeError:
            raise RuntimeError(
                f"Could not find a retrieval output class matching the name '{retrieval_output}'. Please "
                "refer to the 'pytorch_retrieve.retrieval_output' module for available output classes."
            )

        try:
            if parameters is None:
                self.output = output_class(output_config)
            else:
                self.output = output_class(output_config, **parameters)
        except ValueError:
            raise RuntimeError(
                f"Coud not instantiate retrieval output class '{retrieval_output}' with given parameters "
                f"{parameters}. Plase refer to the 'pytorch_retrieve.retrieval_output' module for available "
                " output classes."
            )

    @staticmethod
    def parse(
        name: str, output_cfg: OutputConfig, config_dict: Union[str, Dict[str, Any]]
    ) -> "RetrievalOuputConfig":
        """
        Parse retrieval output config.

        Args:
            name: The name of the retrieval output.
            output_cfg: The OutputConfig object describing the model output from which the retrieval
                output is computed.
            config_dict: The dictionary from which to parse the RetrievalOutputConfig.
        """
        config_dict = copy(config_dict)
        if isinstance(config_dict, str):
            return RetrievalOutputConfig(output_cfg, config_dict, None)

        retrieval_output = config_dict.pop("retrieval_output", None)
        if retrieval_output is None:
            raise RuntimeError(
                f"Retrieval output entry for output '{name}' lacks 'retrieval_output' attribute."
            )
        return RetrievalOutputConfig(
            output_config=output_cfg,
            retrieval_output=retrieval_output,
            parameters=config_dict,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Return dictionary representation of retrieval output.
        """
        dct = {"retrieval_output": self.retrieval_output}
        if self.parameters is not None:
            dct.update(self.parameters)
        return dct


@dataclass
class InferenceConfig:
    """
    Defines which output quantities to compute for a given output.
    """

    batch_size: int = 8
    tile_size: Optional[Tuple[int, int]] = None
    spatial_overlap: Optional[Tuple[int, int]] = None
    temporal_overlap: Optional[int] = None
    input_loader: Optional[str] = None
    input_loader_args: Optional[Dict[str, Any]] = None
    retrieval_output: Optional[Dict[str, Dict[str, RetrievalOutputConfig]]] = None
    exclude_from_tiling: Optional[List[str]] = None

    @staticmethod
    def parse(
        output_cfg: Dict[str, OutputConfig],
        config_dict: Dict[str, Any],
    ) -> "InferenceConfig":
        """
        Parse inference config.

        Args:
            config_dict: A dictionary describing the inference settings.
            output_cfg: A dictionary mapping model output names to corresponding OutputConfig
                objects.

        Return:
            An InferenceConfig object holding the inference configuration.
        """
        batch_size = get_config_attr(
            "batch_size", int, config_dict, "inference config", default=8
        )

        tile_size = get_config_attr(
            "tile_size", None, config_dict, "inference config", default=None
        )
        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_size = (tile_size, tile_size)
            else:
                tile_size = tuple(tile_size)

        spatial_overlap = get_config_attr(
            "spatial_overlap", None, config_dict, "inference config", default=None
        )

        temporal_overlap = get_config_attr(
            "temporal_overlap", None, config_dict, "inference config", default=None
        )

        input_loader = get_config_attr(
            "input_loader", None, config_dict, "inference config", default=None
        )
        input_loader_args = get_config_attr(
            "input_loader_args", None, config_dict, "inference config", default=None
        )
        if input_loader_args is not None:
            input_loader_args = dict(input_loader_args)

        retrieval_output_dct = config_dict.get("retrieval_output", {})
        retrieval_output = {}
        for model_output, outputs in retrieval_output_dct.items():

            if model_output not in output_cfg:
                raise ValueError(
                    f"Found output name '{model_output}' in inference config but model has no "
                    f"corresponding output."
                )

            outputs = {
                output_name: RetrievalOutputConfig.parse(
                    f"{model_output}.{output_name}",
                    output_cfg[model_output],
                    cfg_dict,
                )
                for output_name, cfg_dict in outputs.items()
            }
            retrieval_output[model_output] = outputs

        exclude_from_tiling = get_config_attr(
            "exclude_from_tiling", None, config_dict, "inference config", default=None
        )
        if exclude_from_tiling is not None:
            exclude_from_tiling = [elem for elem in exclude_from_tiling]

        return InferenceConfig(
            batch_size=batch_size,
            tile_size=tile_size,
            spatial_overlap=spatial_overlap,
            temporal_overlap=temporal_overlap,
            input_loader=input_loader,
            input_loader_args=input_loader_args,
            retrieval_output=retrieval_output,
            exclude_from_tiling=exclude_from_tiling
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert InferenceConfig to dict for serialization.
        """
        dct = asdict(self)
        retrieval_output = {}
        for name, target_outputs in self.retrieval_output.items():
            retrieval_output[name] = {
                name: output.to_dict() for name, output in target_outputs.items()
            }
        dct["retrieval_output"] = retrieval_output
        return dct
