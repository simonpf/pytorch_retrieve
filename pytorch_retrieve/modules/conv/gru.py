"""
pytorch_retrieve.modules.conv.gru
=================================

Implements convolutional gated recurrent units.
"""
from typing import Callable, List, Optional, Union, Tuple

import torch
from torch import nn

from pytorch_retrieve.modules.utils import NoNorm


class GRUCell(nn.Module):
    """
    Convolutional version of a gated recurrent unit.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        bias: bool = True,
        activation_factory: Callable[[], nn.Module] = nn.Tanh,
        normalization_factory: Optional[Callable[[int], nn.Module]] = None,
    ):
        """
        Args:
            in_channels: The number of incoming channels.
            hidden_channels: The number of hidden channels.
            kernel_size: The size of the convolution kernels.
            bias: Whether or not the convolution layers should include bias
                terms.
            activation_factory: Factory to produce the activation layer.
            activation_factory: Factory to produce the activation layers.
        """
        super().__init__()
        self.hidden_channels = hidden_channels

        if normalization_factory is None:
            normalization_factory = NoNorm

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv_zr = nn.Conv2d(
            in_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        self.act = activation_factory()
        self.norm_1 = normalization_factory(hidden_channels)
        self.norm_2 = normalization_factory(hidden_channels)

        self.conv_h = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        self.conv_x = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward_single(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward single tensor through cell.

        Args:
           x: The torch tensor to propagate through the cell.
           hidden: The hidden state of the cell.

        Return:
           The new hidden state.
        """
        if hidden is None:
            shape = x.shape
            shape = shape[:1] + (self.hidden_channels,) + shape[2:]
            hidden = x.new_zeros(shape)

        zr = torch.sigmoid(self.norm_1(self.conv_zr(torch.cat([x, hidden], 1))))
        z, r = torch.split(zr, self.hidden_channels, dim=1)

        h_int = self.act(self.norm_2(self.conv_x(x) + self.conv_h(r * hidden)))
        h_new = (1.0 - z) * hidden + z * h_int
        return h_new

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        hidden: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Propagate tensor or sequence of tensors through GRU cell.

        Args:
            x: A list of input tensors or a 5D tensor with sequence elements
               along the first dimenions.
            hidden: The previous hidden state or None.

        Return:
            If the input is a list of tensor a list or resulting hidden states
            is returned. If the input is a 5D tensor, the hidden states are
            returned combined into a single tensor with the sequence elements
            oriented along the first dimension.
        """
        is_tensor = False
        if isinstance(x, torch.Tensor):
            if x.dim() < 5:
                raise RuntimeError(
                    "Input to GRU cell must be a 5-dimensional tensor or a list "
                    " of tensors."
                )
            is_tensor = True

        results = []
        for x_i in x:
            hidden = self.forward_single(x_i, hidden=hidden)
            results.append(hidden)

        if is_tensor:
            return torch.stack(results, 0)

        return results


class GRUNetwork(nn.Module):
    """
    A network of gated recurrent units.

    A GRUNetwork consists of multiple GRU cells that process the input sequence
    consecutively.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: Union[List[int], int],
        n_layers: int,
        **cell_kwargs,
    ):
        """
        Args:
            in_channels: The number of channels in each element of the input
                sequence.
            hidden_channels: The number of hidden channels in each of the GRU
                cells.
            kernel_size: The kernel size used in each GRU cell.
            n_layers: The number of cells in the GRU network.
        """
        super().__init__()
        self.n_layers = n_layers
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * n_layers
        if isinstance(kernel_size, (int, tuple)):
            kernel_size = [kernel_size] * n_layers

        cells = nn.ModuleList()
        for hidden_chans, k_size in zip(hidden_channels, kernel_size):
            cells.append(
                GRUCell(in_channels, hidden_chans, kernel_size=k_size, **cell_kwargs)
            )
            in_channels = hidden_chans
        self.cells = cells

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward sequence of inputs through GRU network.
        """
        if isinstance(x, torch.Tensor):
            if not x.dim() == 5:
                raise RuntimeError(
                    "Input to a GRU network must be 5-dimensional tensor or a list "
                    " of tensors."
                )
        if hidden is None:
            hidden = [None] * self.n_layers

        y = x
        for cell in self.cells:
            y = cell(y)

        return y
