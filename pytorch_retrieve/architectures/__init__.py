"""
pytorch_retrieve.architectures
==============================

This module defines the Architecture classes.
"""
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from ..config import read_config_file
from .mlp import MLP
from .encoder_decoder import EncoderDecoder
from .recurrent_encoder_decoder import RecurrentEncoderDecoder
from .metnet import MetNet
from .model import RetrievalModel


def compile_architecture(config_dict) -> nn.Module:
    arch = config_dict.get("architecture", None)
    if arch is None:
        raise RuntimeError("The model configuration lacks a 'architecture' section.")

    arch_name = arch.get("name", None)
    if arch_name is None:
        raise RuntimeError(
            "Architecture section needs 'name' field to identify the architecture "
            "to instantiate."
        )

    if arch_name == "MLP":
        return MLP.from_config_dict(config_dict)
    elif arch_name == "EncoderDecoder":
        return EncoderDecoder.from_config_dict(config_dict)
    elif arch_name == "RecurrentEncoderDecoder":
        return RecurrentEncoderDecoder.from_config_dict(config_dict)
    elif arch_name == "MetNet":
        return MetNet.from_config_dict(config_dict)

    raise RuntimeError(f"The architecture '{arch_name}' is currently not supported.")


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
