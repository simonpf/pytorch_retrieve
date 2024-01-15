"""
pytorch_retrieve.modules.blocks
===============================

Implements block factories to build recurrent networks.
"""
from typing import Callable, List, Optional, Union

import torch
from torch import nn


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
        **factory_kwargs,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.encoder = block_factory(in_channels, out_channels, **factory_kwargs)
        factory_kwargs["downsample"] = 1
        self.propagator = block_factory(out_channels, out_channels, **factory_kwargs)
        self.assimilator = block_factory(
            2 * out_channels, out_channels, **factory_kwargs
        )
        self.attention = block_factory(2 * out_channels, out_channels, **factory_kwargs)
        self.norm = nn.BatchNorm2d(out_channels)

    def init_state(self, x: torch.Tensor):
        """
        Initialize the hidden state of the block.

        Args:
            x: The input tensor defining the size of the inputs.

        """
        shape = x.shape[:1] + (self.out_channels,) + x.shape[2:]
        return x.new_zeros(shape)

    def forward_step(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        if state is None:
            state = self.init_state(enc)
        prop = self.propagator(state.detach())
        encprop = torch.cat([enc, prop], 1)
        updated = self.assimilator(encprop)
        att = torch.sigmoid(self.attention(encprop))
        return self.norm(att * enc + (1.0 - att) * updated)

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


class Assimilator:
    def __init__(self, block_factory: Callable[[int, int], nn.Module]):
        self.block_factory = block_factory

    def __call__(
        self, in_channels: int, out_channels: int, downsample: int = 1, **kwargs
    ):
        return AssimilatorBlock(
            in_channels,
            out_channels,
            downsample=downsample,
            block_factory=self.block_factory,
        )
