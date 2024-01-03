"""
pytorch_retrieve.modules.aggregation
====================================

Defines aggregator modules that merge inputs from multiple streams.
"""
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn


from pytorch_retrieve.modules.mlp import MLP


class MLPAggregator(MLP):
    """
    Aggregation block consisting of a multi-layer perceptron (MLP).

    The block concatenates all inputs along the last dimension and feeds
    the result into the MLP.
    """

    def __init__(
        self,
        inputs: Dict[str, int],
        out_channels: int,
        n_layers: int,
        hidden_channels: Optional[int] = None,
        residual_connections: Optional[str] = None,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = None,
        internal: bool = False,
        output_shape: Tuple[int] = None,
        masked: bool = False,
    ):
        """
        Create MLP aggregator.

        Args:
            inputs: A dictionary mapping input keys to the corresponding
                 number of channels.
            out_channels: Number of features of the output.
            n_layers: The number of layers.
            hidden_channels: Number of features of the hidden layers.
            residual_connections: The type of residual connections in the MLP:
                None, 'simple', or 'hyper'.
            activation_factory: Factory functional to instantiate the activation
                functions to use in the MLP.
            normalization_factory: Factory functional to instantiate the normalization
                layers to use in the MLP.
            internal: If the module is not an 'internal' module no
                 normalization or activation function are applied to the
                 output.
            output_shape: If provided, the channel dimension of the output will
                 be reshaped to the given shape.
            masked: Whether or not the module should support masked inputs.
        """
        self.inputs = inputs
        in_channels = sum(inputs.values())
        super().__init__(
            in_channels,
            out_channels,
            n_layers,
            hidden_channels=hidden_channels,
            residual_connections=residual_connections,
            activation_factory=activation_factory,
            normalization_factory=normalization_factory,
            internal=internal,
            output_shape=output_shape,
            masked=masked,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([inputs[key] for key in self.inputs], -1)
        return super().forward(x)


class MLPAggregatorFactory:
    """
    Ths MLPAggregatorFactory produces aggregation modules that uses a
    multi-layer perceptron (MLP) to aggrate multiple input streams into
    a single stream with a given number of output channels.
    """

    def __init__(
        self,
        hidden_channels: int,
        n_layers: int,
        residual_connections: Optional[str] = None,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = None,
        internal: bool = True,
        output_shape: Tuple[int] = None,
        masked: bool = False,
    ):
        """
        Create MLP aggregator factory.

        Args:
            hidden_channels: Number of features of the hidden layers.
            n_layers: The number of layers.
            residual_connections: The type of residual connections in the MLP:
                None, 'simple', or 'hyper'.
            activation_factory: Factory functional to instantiate the activation
                functions to use in the MLP.
            normalization_factory: Factory functional to instantiate the normalization
                layers to use in the MLP.
            internal: If the module is not an 'internal' module no
                 normalization or activation function are applied to the
                 output.
            output_shape: If provided, the channel dimension of the output will
                 be reshaped to the given shape.
            masked: Whether or not the module should support masked inputs.
        """
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.residual_connections = residual_connections
        self.activation_factory = activation_factory
        self.normalization_factory = normalization_factory
        self.internal = internal
        self.output_shape = output_shape
        self.masked = masked

    def __call__(self, in_channels: Dict[str, int], out_channels: int):
        """
        Create a MLP aggregation module that aggregates multiple inputs into
        a single tensor.

        Args:
            inputs: A dictionary mapping input keys to the corresponding
                 number of channels.
            out_channels: A tuple specifying the number of channels in the merged
                tensor.

        Return:
            A PyTorch module that aggregates multiple input tensors into a single
            tensor.
        """
        return MLPAggregator(
            in_channels,
            self.hidden_channels,
            out_channels,
            self.n_layers,
            residual_connections=self.residual_connections,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            internal=self.internal,
            output_shape=self.output_shape,
            masked=self.masked,
        )
