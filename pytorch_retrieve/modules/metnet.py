"""
pytorch_retrieve.modules.metnet
===============================

Implements custom modules for the MetNet architecture.
"""

import torch
from torch import nn
from torchvision.transforms.functional import center_crop

from pytorch_retrieve.modules.conv.downsampling import AvgPool2d, Space2Depth


class Stem(nn.Module):
    """
    Stem module that implements the Metnet preprocessing.

    The MetNet preprocessing downsamples radar and satellite inputs by a factor
    of 4. The first downsampling typically differs between radar and satellite
    data. The radar inputs are downsampled using a space-to-depth transformation,
    whereas the satellite data is average pooled.

    The second downsampling is average pooling for both input modalities. In
    addition to the second downsampling a center crop of the half image size
    is extracted and appended to the outputs.
    """

    def __init__(
        self,
        in_channels: int,
        first_stage_kind: str = "avgpool",
        center_crop: bool = True,
    ):
        """
        Args:
            in_channels: The number of incoming channels.
            first_stage_kind: String specifying the type of the first downsampling
                applied to the input. Should be 'avgpool' for average pooling or
                'space2depth' for space-to-depth downsampling.
            center_crop: Whether or not to append a center crop to the down-scaled
                inputs.

        """
        super().__init__()
        if first_stage_kind == "avgpool":
            self.down_1 = AvgPool2d()(in_channels, in_channels, 2)
        else:
            self.down_1 = Space2Depth()(in_channels, 4 * in_channels, 2)
        self.down_2 = AvgPool2d()(in_channels, in_channels, 2)
        self.center_crop = center_crop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess tensor.
        """
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        if not self.center_crop:
            return down_2

        size = down_1.shape[-1]
        return torch.cat([down_2, center_crop(down_1, size // 2)], 1)


class SpatialAggregator(nn.Module):
    """
    The spatial aggergator is a sequence of axial self-attention modules.
    """

    def __init__(
        self,
        input_size,
        n_channels,
        depth: int,
        n_heads: int,
    ):
        """
        Args:
            input_size: The width/height of the input images.
            n_channels: The number of features in the spatial aggregator.
            depth: The number of axial attention modules.
            n_heads: The number of heads in the self attention modules.
        """
        super().__init__()
        try:
            from axial_attention import AxialAttention, AxialPositionalEmbedding
        except ImportError:
            raise RuntimeError(
                "The MetNet spatial aggregator requires the axial attention module "
                "to be installed."
            )

        if isinstance(input_size, int):
            input_size = (input_size,) * 2

        self.position_embedding = AxialPositionalEmbedding(
            dim=n_channels, shape=input_size
        )
        self.body = nn.Sequential(
            *[
                AxialAttention(
                    dim=n_channels, dim_index=1, heads=n_heads, num_dimensions=2
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate tensor through aggregator.
        """
        return self.body(self.position_embedding(x))
