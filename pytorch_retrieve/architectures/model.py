"""
pytorch_retrieve.architecture.model
===================================

Defines the RetrievalModel class, which is the base class for all retrieval
models. A RetrievalModel is a 'torch.nn.Module' that is produced by instantiating
an architecture.
"""
from pathlib import Path
from typing import Dict, Union

import torch
from torch import nn

from pytorch_retrieve.config import InputConfig, OutputConfig, InferenceConfig
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
        self._inference_config = None


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
        loaded = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
        model = compile_architecture(loaded["model_config"])
        state = loaded["state_dict"]
        if path.suffix == ".ckpt":
            state = {key[6:]: val for key, val in state.items()}
        model.load_state_dict(state)

        inference_config = loaded.get("inference_config", None)
        model._inference_config = inference_config

        return model


    def save(self, path: Path) -> Path:
        """
        Save retrieval model.

        Args:
            path: Path to which to write the saved model.
        """
        state = self.state_dict()
        model_config = self.config_dict
        torch.save(
            {
                "state_dict": state,
                "model_config": model_config,
                "inference_config": self._inference_config
            },
            path
        )

    @property
    def inference_config(self) -> Union[InferenceConfig, None]:
        if self._inference_config is None:
            return None
        return InferenceConfig.parse(self.output_config, self._inference_config)

    @inference_config.setter
    def inference_config(self, cfg: InferenceConfig) -> None:
        self._inference_config = cfg.to_dict()

    @property
    def input_config(self) -> Dict[str, InputConfig]:
        """
        A dictionary mapping names of the model inputs to corresponding InputConfig object.
        """
        return {
            name: InputConfig.parse(name, cfg) for name, cfg in self.config_dict["input"].items()
        }

    @property
    def output_config(self) -> Dict[str, OutputConfig]:
        """
        A dictionary mapping names of the model output to corresponding OutputConfig object.
        """
        return {
            name: OutputConfig.parse(name, cfg) for name, cfg in self.config_dict["output"].items()
        }


    def to_config_dict(self) -> Dict[str, object]:
        """
        Return configuration used to construct the model.

        """
        return self.config_dict
