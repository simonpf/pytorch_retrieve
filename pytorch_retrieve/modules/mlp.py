"""
pytorch_retrieve.modules.mlp
============================

Implements modules based on fully-connected layers that can be used for
both tabular and image data.
"""
from typing import Optional, Callable, Tuple

from torch import nn

from pytorch_retrieve.config import get_config_attr
from . import activation
from . import normalization
from .utils import ParamCount


class MLPBlock(nn.Module):
    """
    A building block for a fully-connected (MLP) network.

    This block expects the channels/features to be oriented along the
    last dimension of the tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_factory: Callable[[], nn.Module],
        normalization_factory: Callable[[int], nn.Module],
        residual_connections: Optional[str] = None,
        masked: bool = False,
    ):
        """
        Args:
            in_channels: The number of channels/features in the input tensor.
            out_channels: The number of features of the block output.
            activation_factory: A factory functional to create the activation
                layers in the block.
            normalization_factory: A factory functional to create the normalization
                layers used in the block.
            residual_connections: The type of residual connections to apply.
        """
        super().__init__()
        if masked:
            mod = nm
        else:
            mod = nn

        if residual_connections is not None:
            residual_connections = residual_connections.lower()
        self.residual_connections = residual_connections

        if normalization_factory is not None:
            modules = [
                mod.Linear(in_channels, out_channels, bias=False),
                normalization_factory(out_channels),
                activation_factory(),
            ]
        else:
            modules = [
                mod.Linear(in_channels, out_channels, bias=True),
                activation_factory(),
            ]
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        """
        Propagate input through the block.

        Args:
            x: For residual connections 'None' or 'simple', the input tensor.
                for residual connections 'hyper', a tuple
                ``(x, acc, n_acc)`` containing the input ``x``, the
                accumulation buffer ``acc`` and the accumulation counter
                ``n_acc``.

        Return:
            If 'residual_connections' is 'None' or 'simple', the output is
            just the output tensor ``y``. If residual_connections is `hyper`,
            the output is a tuple ``(y, acc, n_acc)`` containing the block
            output ``y``, the accumulation buffer ``acc`` and the accumulation
            counter ``n_acc``.
        """
        if self.residual_connections is None:
            return self.body(x)

        if self.residual_connections == "simple":
            y = self.body(x)
            diff = y.shape[-1] - x.shape[-1]
            if diff <= 0:
                y = y + x[..., : y.shape[-1]]
            else:
                y = y + nn.functional.pad(x, (0, diff))
            return y

        if isinstance(x, tuple):
            x, acc, n_acc = x
            acc = acc.clone()

            diff = acc.shape[-1] - x.shape[-1]
            if diff <= 0:
                acc = acc + x[..., : acc.shape[-1]]
            else:
                acc = acc + nn.functional.pad(x, (0, diff))
            n_acc += 1
        else:
            acc = x.clone()
            n_acc = 1
            n = x.shape[-1]

        y = self.body(x)
        diff = y.shape[-1] - acc.shape[-1]
        if diff <= 0:
            y = y + acc[..., : y.shape[-1]] / n_acc
        else:
            y = y + nn.functional.pad(acc / n_acc, (0, diff))
        return y, acc, n_acc


class MLP(ParamCount, nn.Module):
    """
    A fully-connected feed-forward neural network.

    The MLP can be used both as a fully-connected on 2D data as well
    as a module in a CNN. When used with 4D output the input is
    automatically permuted so that features are oriented along the last
    dimension of the input tensor.
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
        internal: bool = False,
        output_shape: Tuple[int] = None,
        masked: bool = False,
    ):
        """
        Create MLP module.

        Args:
            in_channels: Number of features in the input.
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
        """
        super().__init__()
        if masked:
            mod = nm
        else:
            mod = nn

        if hidden_channels is None:
            hidden_channels = out_channels

        self.n_layers = n_layers
        if residual_connections is not None:
            residual_connections = residual_connections.lower()
        self.residual_connections = residual_connections

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.layers.append(
                MLPBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    activation_factory=activation_factory,
                    normalization_factory=normalization_factory,
                    residual_connections=self.residual_connections,
                    masked=masked,
                )
            )
            in_channels = hidden_channels

        if n_layers > 0:
            if internal:
                self.output_layer = MLPBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    activation_factory=activation_factory,
                    normalization_factory=normalization_factory,
                    residual_connections=self.residual_connections,
                    masked=masked,
                )
            else:
                self.output_layer = mod.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        if isinstance(output_shape, int):
            output_shape = ()
        self.output_shape = output_shape

    def forward(self, x):
        """
        Forward input through network.

        Args:
            x: The 4D or 2D input tensor to propagate through
                the network.

        Return:
            The output tensor.
        """
        needs_reshape = False
        input_shape = x.shape
        if x.ndim == 4:
            needs_reshape = True
            x = torch.permute(x, (0, 2, 3, 1))
            old_shape = x.shape
            x = x.reshape((-1, old_shape[-1]))

        if self.n_layers == 0:
            return x, None

        y = x
        for l in self.layers:
            y = l(y)
        if self.residual_connections == "hyper":
            y = y[0]
        y = self.output_layer(y)

        if needs_reshape:
            y = y.view(old_shape[:-1] + (self.out_channels,))
            y = torch.permute(y, (0, 3, 1, 2))

        # If required, reshape channel dimension
        if self.output_shape is not None:
            if needs_reshape:
                shape = input_shape[:1] + self.output_shape + input_shape[2:]
            else:
                shape = input_shape[:1] + self.output_shape
            y = y.view(shape)

        return y
