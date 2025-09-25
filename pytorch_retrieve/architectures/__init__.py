"""
pytorch_retrieve.architectures
==============================

This module defines the Architecture classes.
"""
from copy import copy
import importlib
from pathlib import Path
from types import MethodType
from typing import Any, Dict

import torch
from torch import nn

from ..config import read_config_file
from .mlp import MLP
from .encoder_decoder import EncoderDecoder
from .satformer import Satformer
from .recurrent_encoder_decoder import RecurrentEncoderDecoder
from .metnet import MetNet
from .autoregressive import Autoregressive
from .multi_scale_autoregressor import MultiScaleAutoregressor
from .direct_forecast import DirectForecast
from .model import RetrievalModel


def compile_architecture(config_dict) -> nn.Module:
    """
    Compile retrieval model from configuration dict.

    Args:
        config_dict: A dictionary containing the configuration of the model to compile.
    """
    from ..inference import InferenceConfig

    arch = config_dict.get("architecture", None)
    if arch is None:
        raise RuntimeError("The model configuration lacks a 'architecture' section.")

    # TODO: Would be great to handle output_names more consistently if it is required by the training interface.
    #arch["model_class"] = "rtem.models.HeatingRateUNet3D"
    if "model_class" in arch:
        arch = copy(arch)
        model_class = arch.pop("model_class")
        if model_class is not None:
            *module,  model_class = model_class.split(".")
            module = importlib.import_module(".".join(module))
            model_class = getattr(module, model_class)
            model = model_class(**arch)
            model.config_dict = config_dict
            model._inference_config = None
            model.__class__ = type(
                model.__class__.__name__ + "Mixin",
                (RetrievalModel, model.__class__),  # order matters for MRO
                {}
            )
            model.output_names = list(config_dict.get("output", {}).keys())
            return model
    else:
        arch_name = arch.get("name", None)
        if arch_name is None:
            raise RuntimeError(
                "Architecture section needs 'name' field to identify the architecture "
                "to instantiate."
            )

        model = None
        if arch_name == "MLP":
            model = MLP.from_config_dict(config_dict)
        elif arch_name == "EncoderDecoder":
            model = EncoderDecoder.from_config_dict(config_dict)
        elif arch_name == "RecurrentEncoderDecoder":
            model = RecurrentEncoderDecoder.from_config_dict(config_dict)
        elif arch_name == "Autoregressive":
            model = Autoregressive.from_config_dict(config_dict)
        elif arch_name == "MultiScaleAutoregressor":
            model = MultiScaleAutoregressor.from_config_dict(config_dict)
        elif arch_name == "DirectForecast":
            model = DirectForecast.from_config_dict(config_dict)
        elif arch_name == "MetNet":
            model = MetNet.from_config_dict(config_dict)
        elif arch_name == "Satformer":
            model = Satformer.from_config_dict(config_dict)
        elif arch_name == "PrithviWxC":
            from . import prithvi_wxc
            model = prithvi_wxc.PrithviWxCModel.from_config_dict(config_dict)
        else:
            raise RuntimeError(
                f"The architecture '{arch_name}' is currently not supported."
            )

    if "name" in config_dict:
        model.config_dict["name"] = config_dict["name"]

    inference_config = config_dict.get("inference", None)
    if inference_config is not None:
        inference_config = InferenceConfig.parse(model.output_config, inference_config)
        model.inference_config = inference_config

    return model


def compile_preset(
    arch_name: str,
    preset_name: str,
    input_configs: Dict[str, Any],
    output_configs: Dict[set, Any],
) -> RetrievalModel:
    """
    Compile preset.

    Args:
        arch_name: The name of the architecture.
        preset_name: The name of the preset.
        input_configs: Configuration dicts describing the retrieval inputs.
        output_configs: Configuration dicts describing the retrieval outputs.

    Return:
        A RetrievalModel implementing the preset.
    """
    config_dict = {
        "input": input_configs,
        "output": output_configs,
        "architecture": {"name": arch_name, "preset": preset_name},
    }
    return compile_architecture(config_dict)


def load_and_compile_model(path: Path) -> nn.Module:
    """
    Load configuration from file and compile model.

    Args:
        path: Path object pointing to a model configuration to compile and load.

    Return:
        A compile pytorch model representing the architecture defined in the
        the provided model configuration file.
    """
    path = Path(path)
    config_dict = read_config_file(path)
    return compile_architecture(config_dict)


def load_model(path: Path) -> nn.Module:
    """
    Load a model from file.

    Args:
        path: A path pointing a saved 'pytorch_retrieve' model.

    Return:
        The loaded model.
    """
    return RetrievalModel.load(path)
