"""
pytorch_retrieve.modules.stems
==============================

Defines stem modules, which are responsible for adapting input data to
to the input expected by the retrieval body.
"""
from typing import Callable, Optional

from torch import nn

from pytorch_retrieve.modules.mlp import MLP


class MLPStem(MLP):
    """
    An MLPStem is just an MLP without an output layer.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_layers: int,
        hidden_channels: Optional[int] = None,
        residual_connections: Optional[str] = None,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = None,
        internal=True,
        masked=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            n_layers,
            hidden_channels=hidden_channels,
            residual_connections=residual_connections,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
            internal=internal,
            masked=masked,
        )
