"""
pytorch_retrieve.modules.conv.aggregation
=========================================

Aggregator modules for use in convolutional neural network architectures.
"""
from typing import Dict

import torch
from torch import nn


class Linear(nn.Module):
    """
    Aggregates input tensors by concatenation along dimension 1 and the
    linearly projection the concatenated tensor onto the desired number
    of output channels.
    """

    def __init__(self, inputs: Dict[str, int], out_channels: int):
        """
        Args:
            inputs: A dictionary mapping input names to number of channels.
            out_channels: The number of channels to project the aggregated input
                onto.
        """
        super().__init__()
        in_channels = sum(inputs.values())
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        inpts = list(x.values())
        inpt_0 = inpts[0]
        if isinstance(inpt_0, list):
            return [
                self.body(torch.cat([inpt[ind] for inpt in inpts], 1))
                for ind in range(len(inpt_0))
            ]
        inpts = torch.cat(list(x.values()), 1)
        return self.body(inpts)


class Linear3d(nn.Module):
    """
    Three-dimensional version of linear aggregation.
    """

    def __init__(self, inputs: Dict[str, int], out_channels: int):
        """
        Args:
            inputs: A dictionary mapping input names to number of channels.
            out_channels: The number of channels to project the aggregated input
                onto.
        """
        super().__init__()
        in_channels = sum(inputs.values())
        self.body = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        inpts = torch.cat(list(x.values()), 1)
        return self.body(inpts)
