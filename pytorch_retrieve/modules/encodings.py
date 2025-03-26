"""
pytorch_retrieve.modules.encodings
==================================

Provides encoding modules typically used to encode information in transformer-based networks.
"""

import torch
from torch import nn


class FourierEncoding(nn.Module):
    """
    Encodes scalar features using sine and cosine waves of increasing frequency.
    Assumes input features to be normalized into the range [-1, 1]. If the input has more
    than one channel, the output channels used to encode each input feature are split
    evenly between the input and output features.
    """

    def __init__(self, in_channels: int, out_channels, dim: int = -3):
        """
        Args:
            in_channels: The number of continuous features to encode.
            out_channels: The number of channels to use to encode the features.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if (self.out_channels % 2) != 0:
            raise ValueError("'out_channels' must be an even number 'in_channels'.")
        if (self.out_channels // 2) % self.in_channels != 0:
            raise ValueError(
                "The number of 'in_channels' must evenly divice half of 'out_channels'."
            )
        self.channels_per_feature = self.out_channels // 2 // self.in_channels
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the scala information in x using interleaved cosine and sine wave of increasing frequency.

        Args:
            x: The input tensor containing the features to encode.

        Return:
            A tensor containing interleaved cosine and sine signals of the values in x.
        """
        dim = self.dim if self.dim > 0 else x.dim() + self.dim
        dims_front = dim
        dims_back = x.dim() - dim - 1
        shape = (1,) * dims_front + (-1,) + (1,) * dims_back
        freq = (
            torch.arange(1, self.channels_per_feature + 1, dtype=x.dtype, device=x.device)
            .view(shape)
            .repeat_interleave(self.in_channels, dim)
        )
        x = freq * (torch.pi * x).repeat_interleave(self.channels_per_feature, dim)
        x_cos = torch.cos(x)
        x_sin = torch.sin(x)
        x_enc = torch.stack((x_cos, x_sin), dim=dim + 1).flatten(dim, dim + 1)
        return x_enc
