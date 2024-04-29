"""
pytorch_retrieve.architecture.model
===================================

Defines the RetrievalModel class, which is the base class for all retrieval
models. A RetrievalModel is a 'torch.nn.Module' that is produced by instantiating
an architecture.
"""
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from pytorch_retrieve.modules.utils import ParamCount


class RetrievalModel(ParamCount, nn.Module):
    """
    The retrieval model class is the base class for all neural network
    retrieval models provided by 'pytorch_retrieve'.
    """

    def __init__(self, config_dict: Dict[str, object]):
        """
        Args:
            config_dict: The config dict that the retrieval model was instantiated
                from.
        """
        super().__init__()
        self.config_dict = config_dict
        self.inference_config = None

    @classmethod
    def load(cls, path: Path) -> nn.Module:
        """
        Load a model from file.

        Args:
            path: A path pointing a saved 'pytorch_retrieve' model.

        Return:
            The loaded model.
        """
        from pytorch_retrieve.inference import InferenceConfig
        from . import compile_architecture

        path = Path(path)
        loaded = torch.load(path)
        model = compile_architecture(loaded["model_config"])
        state = loaded["state_dict"]
        if path.suffix == ".ckpt":
            state = {key[6:]: val for key, val in state.items()}
        model.load_state_dict(state)

        inference_config = loaded.get("inference_config", None)
        if inference_config is not None:
            inference_config = InferenceConfig.parse(inference_config)
        model.inference_config = inference_config

        return model


    def save(self, path: Path) -> Path:
        """
        Save retrieval model.

        Args:
            path: Path to which to write the saved model.
        """
        state = self.state_dict()
        model_config = self.config_dict
        if hasattr(self, "inference_config"):
            inference_config = self.inference_config
            if inference_config is not None:
                inference_config = inference_config.to_dict()
        torch.save(
            {
                "state_dict": state,
                "model_config": model_config,
                "inference_config": inference_config
            },
            path
        )

    def to_config_dict(self) -> Dict[str, object]:
        """
        Return configuration used to construct the model.

        """
        return self.config_dict
