"""
pytorch_retrieve.modules.conv.encoders
======================================

Defines decoder modules for use within encoder-decoder architectures.
"""
from copy import copy
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn

from pytorch_retrieve.modules.utils import ParamCount
from pytorch_retrieve.modules.conv.encoders import DEFAULT_BLOCK_FACTORY
from pytorch_retrieve.modules.conv.stages import SequentialStage
from pytorch_retrieve.modules.conv.upsampling import Bilinear


def cat(
    tensor_1: Union[torch.Tensor, List[torch.Tensor]],
    tensor_2: Union[torch.Tensor, List[torch.Tensor]],
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Generic concatenation two tensors or two lists of tensors along first axis.

    Args:
        tensor_1: A single tensor or a list of tensors.
        tensor_2: A single tensor or a list of tensors.

    Return:
        A single tensor if 'tensor_1' and 'tensor_2' are lists. Otherwise
        a list of tensors.
    """
    if isinstance(tensor_1, List):
        return [torch.cat([t_1, t_2], 1) for t_1, t_2 in zip(tensor_1, tensor_2)]
    return torch.cat([tensor_1, tensor_2], 1)


def forward(
    module: nn.Module, tensor: Union[torch.Tensor, List[torch.Tensor]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Generic forward function for single tensors and lists of tensors.

    Args:
        module: The torch.nn.Module through which to propagate the provided tensors.
        tensor: The tensor or tensors to propagate through the module.

    Return:
        A single tensor or list of tensors obtained by propagating the input
        'tensor' through the given module.
    """
    if isinstance(tensor, list):
        return [module(tnsr) for tnsr in tensor]
    return module(tensor)


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

        # Handle block factories and match to stage factories.
        if block_factory is None:
            block_factory = DEFAULT_BLOCK_FACTORY
        if stage_factory is None:
            if isinstance(block_factory, list):
                if len(block_factory) < n_stages:
                    raise RuntimeError(
                        "If a list of block factories is provided, its length must match "
                        "the number of stages in the encoder."
                    )
                stage_factories = [SequentialStage(b_fac) for b_fac in block_factory]
            else:
                stage_factories = [SequentialStage(block_factory) for _ in range(n_stages)]
        else:
            if isinstance(stage_factory, list):
                raise RuntimeError(
                    "If a list of stage factories is provided, its length must match "
                    "the number of stages in the encoder."
                )
                if isinstance(block_factory, list):
                    if len(block_factory) < n_stages:
                        raise RuntimeError(
                            "If a list of block factories is provided, its length must match "
                            "the number of stages in the encoder."
                        )
                    stage_factories = [s_fac(b_fac) for s_fac, b_fac in zip(stage_factory, block_factory)]
                else:
                    stage_factories = [s_fac(block_factory) for s_fac in stage_factory]


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
                base_scale = np.prod(
                    [fac if isinstance(fac, int) else max(fac) for fac in upsampling_factors]
                )
        self.base_scale = base_scale

        self.has_skips = True
        if skip_connections is None:
            skip_connections = {}
            self.has_skips = False
        else:
            skip_connections = copy(skip_connections)
        self.skip_connections = copy(skip_connections)

        if self.base_scale in self.skip_connections:
            in_channels = skip_connections[self.base_scale]
        else:
            in_channels = channels[0]

        scale = self.base_scale
        for index, (n_blocks, out_channels) in enumerate(
            zip(stage_depths, channels[1:])
        ):
            f_up = upsampling_factors[index]
            if isinstance(f_up, (list, tuple)):
                f_up = max(f_up)
            print("FUP :: ", f_up, self.base_scale, upsampling_factors)
            scale /= f_up

            self.upsamplers.append(
                upsampler_factory(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    factor=upsampling_factors[index],
                )
            )

            skip_chans = skip_connections.pop(scale, 0)
            channels_combined = out_channels + skip_chans

            self.stages.append(
                stage_factories[index](channels_combined, out_channels, n_blocks, scale=scale)
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

        prev_scale = -1

        if isinstance(x, dict):
            scale = self.base_scale
            y = x[scale]
            stages = self.stages

            for ind, (up, stage) in enumerate(zip(self.upsamplers, stages)):
                f_up = self.upsampling_factors[ind]
                if isinstance(f_up, (list, tuple)):
                    f_up = max(f_up)
                scale /= f_up
                if scale in self.skip_connections and scale != prev_scale:
                    y = stage(cat(x[scale], forward(up, y)))
                else:
                    y = stage(forward(up, y))
                prev_scale = scale
        else:
            y = x
            stages = self.stages
            for up, stage in zip(self.upsamplers, stages):
                y = stage(forward(up, y))
        return y
