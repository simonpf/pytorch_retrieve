"""
pytorch_retrieve.modules.blocks
===============================

Implements block factories to build recurrent networks.
"""
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from pytorch_retrieve.modules.utils import NoNorm
from pytorch_retrieve.modules.normalization import LayerNormFirst


def forward(
    module: nn.Module, tensor: Union[torch.Tensor, List[torch.Tensor]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Apply module to single tensor or list of tensors.

    Args:
        tensor: A single tensor or a list of tensors.

    Return:
        If tensor is a single tensor, simply the result of applying the given
        torch.nn.Module to the input tensor is returned. Otherwise a list of
        the module applied to all tensors separately is returned.
    """
    if isinstance(tensor, list):
        return [module(tnsr) for tnsr in tensor]
    return module(tensor)


class AssimilatorBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_factory: Callable[[int, int], nn.Module],
        bidirectional: bool = False,
        **factory_kwargs,
    ):
        super().__init__()

        self.reversed = False
        if "block_index" in factory_kwargs:
            if bidirectional:
                self.reversed = factory_kwargs["block_index"] % 2 == 0

        self.out_channels = out_channels
        self.encoder = block_factory(in_channels, out_channels, **factory_kwargs)

        factory_kwargs["downsample"] = 1
        self.propagator = block_factory(out_channels, out_channels, **factory_kwargs)
        self.assimilator = block_factory(
            2 * out_channels, out_channels, **factory_kwargs
        )
        self.output = block_factory(2 * out_channels, out_channels, **factory_kwargs)
        self.act = nn.Tanh()
        self.norm_state = LayerNormFirst(out_channels)
        self.norm_output = LayerNormFirst(out_channels)

    def init_state(self, x: torch.Tensor):
        """
        Initialize the hidden state of the block.

        Args:
            x: The input tensor defining the size of the inputs.

        """
        shape = x.shape[:1] + (self.out_channels,) + x.shape[2:]
        state = torch.normal(0.0, 1.0, shape)
        return state.to(device=x.device, dtype=x.dtype)

    def forward_step(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        if state is None:
            prop = enc
            #state = self.init_state(enc)
        else:
            prop = self.propagator(state)
        encprop = torch.cat([enc, prop], 1)
        corr_state = self.assimilator(encprop)
        output = self.norm_output(self.output(encprop))
        return output, self.norm_state(prop + corr_state)

    def forward(
        self, inputs: List[torch.Tensor], state: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Forward input sequence through assimilation block.

        Args:
            inputs: A list of tensors containing the model inputs.

        Return:
            A list containign the updated states for all inputs in the sequence.
        """
        y = []
        if self.reversed:
            inputs = inputs[::-1]
        for x in inputs:
            x, state = self.forward_step(x, state)
            y.append(x)

        if self.reversed:
            y = y[::-1]
        return y


class Assimilator:
    def __init__(self, block_factory: Callable[[int, int], nn.Module], bidirectional: bool = False):
        self.block_factory = block_factory
        self.bidirectional = bidirectional

    def __call__(
        self, in_channels: int, out_channels: int, downsample: int = 1, **kwargs
    ):
        return AssimilatorBlock(
            in_channels,
            out_channels,
            downsample=downsample,
            block_factory=self.block_factory,
            bidirectional=self.bidirectional,
            **kwargs
        )


class GRUBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: int = 1,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        bias: bool = True,
        activation_factory: Callable[[], nn.Module] = nn.Tanh,
        normalization_factory: Optional[Callable[[int], nn.Module]] = None,
        **factory_kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels

        if isinstance(downsample, int):
            downsample = (downsample,) * 2
        if downsample is not None and np.prod((downsample)) > 1:
            self.projection = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=downsample,
                stride=downsample
            )
        else:
            self.projection = None

        if normalization_factory is None:
            normalization_factory = NoNorm

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv_zr = nn.Conv2d(
            in_channels + out_channels,
            2 * out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        self.act = activation_factory()
        self.norm_1 = normalization_factory(out_channels)
        self.norm_2 = normalization_factory(out_channels)

        self.conv_h = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        self.conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def init_state(self, x: torch.Tensor):
        """
        Initialize the out state of the block.

        Args:
            x: The input tensor defining the size of the inputs.

        """
        shape = x.shape[:1] + (self.out_channels,) + x.shape[2:]
        return x.new_zeros(shape)

    def forward_step(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:

        if self.projection is not None:
            x = self.projection(x)

        if state is None:
            shape = x.shape
            shape = shape[:1] + (self.out_channels,) + shape[2:]
            state = x.new_zeros(shape)

        zr = torch.sigmoid(self.norm_1(self.conv_zr(torch.cat([x, state], 1))))
        z, r = torch.split(zr, self.out_channels, dim=1)

        h_int = self.act(self.norm_2(self.conv_x(x) + self.conv_h(r * state)))
        h_new = (1.0 - z) * state + z * h_int
        return h_new

    def forward(
        self, inputs: List[torch.Tensor], state: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Forward input sequence through assimilation block.

        Args:
            inputs: A list of tensors containing the model inputs.

        Return:
            A list containign the updated states for all inputs in the sequence.
        """
        y = []
        for x in inputs:
            state = self.forward_step(x, state)
            y.append(state)
        return y


class GRU:
    def __init__(self, block_factory: Callable[[int, int], nn.Module]):
        self.block_factory = block_factory

    def __call__(
        self, in_channels: int, out_channels: int, downsample: int = 1, **kwargs
    ):
        return GRUBlock(
            in_channels,
            out_channels,
            downsample=downsample,
            block_factory=self.block_factory,
        )
