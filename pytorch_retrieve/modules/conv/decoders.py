"""
pytorch_retrieve.modules.conv.encoders
======================================

Defines decoder modules for use within encoder-decoder architectures.
"""
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn

from pytorch_retrieve.modules.utils import ParamCount
from pytorch_retrieve.modules.conv.encoders import DEFAULT_BLOCK_FACTORY
from pytorch_retrieve.modules.conv.stages import SequentialStage
from pytorch_retrieve.modules.conv.upsampling import Bilinear


class Decoder(nn.Module, ParamCount):
    """
    A decoder for spatial information.

    The decoder takes a 4D input (batch x channel x height x width),
    and decodes channel information input spatial information.

    The decoder consists of multiple stages each preceded by an
    upsampling layer. Features from skip connections are merged
    after the upsamling before the convolutional block of each stage
    are applied.
    """

    def __init__(
        self,
        channels: List[int],
        stage_depths: List[int],
        upsampling_factors: List[int] = None,
        block_factory: Optional[Callable[[int, int], nn.Module]] = None,
        skip_connections: Optional[Dict[int, int]] = None,
        stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
        upsampler_factory: Callable[[int, int, int], nn.Module] = Bilinear(),
        base_scale: Optional[int] = None,
    ):
        """
        Args:
            channels: A list of integers specifying the number of channels/features
                before the first stage and after every stage of the decoder.
            stage_depths: A list integers specifying the number of block in each
                stage of the decoder.
            upsample_factory: An optional list of integers specifying the factor
                by which inputs are upsampled prior to each stage of the decoder.
            block_factory: Factory functional to use to create the blocks
                in the encoders' stages.
            skip_connections: If the decoder expects skip connections, this should
                by a dict mapping scales of the decoder stages to the corresponding
                incoming numbers of channels. If it is 'None', the decoder does not
                expect skip connections.
            stage_factory: Factory functional to use to create the stages in
                the decoder.
            upsampler_factory: Factory functional to use to create the
                upsampling modules in the decoder.
            upsampling_factors: The upsampling factors for each decoder
                stage.
            base_scale: Integer specifying the scale of the lowest-resolution
                input to the decoder. This scale is used as reference to calculate
                the scale corresponding to each stage of the decoder, if skip
                connections is not provided as a 'dict'.
        """
        super().__init__()
        n_stages = len(stage_depths)
        self.n_stages = n_stages
        self.upsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()

        if len(channels) != self.n_stages + 1:
            raise ValueError(
                "The list of given channel numbers must exceed the number "
                "of stages in the decoder by one."
            )

        if block_factory is None:
            block_factory = DEFAULT_BLOCK_FACTORY

        if stage_factory is None:
            stage_factory = SequentialStage(block_factory)

        if upsampling_factors is None:
            upsampling_factors = [2] * n_stages
        if self.n_stages != len(upsampling_factors):
            raise ValueError(
                "The number of upsampling factors  must equal to the "
                "number of stages."
            )
        self.upsampling_factors = upsampling_factors

        if base_scale is None:
            if isinstance(skip_connections, dict):
                base_scale = max(skip_connections.keys())
            else:
                base_scale = np.prod(upsampling_factors)
        self.base_scale = base_scale

        self.has_skips = True
        if skip_connections is None:
            skip_connections = {}
            self.has_skips = False
        self.skip_connections = skip_connections

        if self.base_scale in self.skip_connections:
            in_channels = skip_connections[self.base_scale]
        else:
            in_channels = channels[0]

        scale = self.base_scale
        for index, (n_blocks, out_channels) in enumerate(
            zip(stage_depths, channels[1:])
        ):
            scale /= upsampling_factors[index]

            self.upsamplers.append(
                upsampler_factory(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    factor=upsampling_factors[index],
                )
            )

            channels_combined = out_channels + self.skip_connections.get(scale, 0)

            self.stages.append(
                stage_factory(channels_combined, out_channels, n_blocks, scale=scale)
            )
            in_channels = out_channels

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            x: The output from the encoder. This should be a single tensor
               if no skip connections are used. If skip connections are used,
               x should be list containing the outputs from each stage in the
               encoder.

        Return:
            The output tensor from the last decoder stage.
        """
        if self.has_skips:
            if not isinstance(x, dict):
                raise ValueError(
                    f"For a decoder with skip connections the input must "
                    f"be a dictionary mapping stage indices to inputs."
                )
        else:
            if isinstance(x, dict):
                x = x[self.base_scale]

        if isinstance(x, dict):
            scale = self.base_scale
            y = x[scale]
            stages = self.stages

            for ind, (up, stage) in enumerate(zip(self.upsamplers, stages)):
                scale /= self.upsampling_factors[ind]
                if scale in self.skip_connections:
                    y = stage(torch.cat([x[scale], up(y)], dim=1))
                else:
                    y = stage(up(y))
        else:
            y = x
            stages = self.stages
            for up, stage in zip(self.upsamplers, stages):
                y = stage(up(y))
        return y