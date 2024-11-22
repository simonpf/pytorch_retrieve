"""
pytorch_retrieve.modules.conv.encoders
======================================

Defines decoder modules for use within encoder-decoder architectures.
"""
from copy import copy
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn

from .utils import Scale
from pytorch_retrieve.modules.utils import ParamCount
from pytorch_retrieve.modules.conv.encoders import DEFAULT_BLOCK_FACTORY
from pytorch_retrieve.modules.conv.stages import SequentialStage
from pytorch_retrieve.modules.conv.upsampling import Bilinear, Trilinear


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
        module: nn.Module, tensor: Union[torch.Tensor, List[torch.Tensor]],
        **kwargs
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
        return [module(tnsr, **kwargs) for tnsr in tensor]
    return module(tensor, **kwargs)


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
        self.channels = channels

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
                        "the number of stages in the decoder."
                    )
                stage_factories = [SequentialStage(b_fac) for b_fac in block_factory]
            else:
                stage_factories = [SequentialStage(block_factory) for _ in range(n_stages)]
        else:
            if isinstance(stage_factory, list):
                if isinstance(block_factory, list):
                    if len(block_factory) < n_stages:
                        raise RuntimeError(
                            "If a list of block factories is provided, its length must match "
                            "the number of stages in the decoder."
                        )
                    stage_factories = [s_fac(b_fac) for s_fac, b_fac in zip(stage_factory, block_factory)]
                else:
                    stage_factories = [s_fac(block_factory) for s_fac in stage_factory]


        if upsampling_factors is None:
            upsampling_factors = [(2, 2)] * n_stages
        if self.n_stages != len(upsampling_factors):
            raise ValueError(
                "The number of upsampling factors  must equal to the "
                "number of stages."
            )

        upsampling_factors = [
            f_u if isinstance(f_u, (tuple, list)) else (f_u, f_u)
            for f_u in upsampling_factors
        ]
        self.upsampling_factors = upsampling_factors

        if base_scale is None:
            if isinstance(skip_connections, dict):
                base_scale = max(skip_connections.keys())
            else:
                base_scale = Scale(tuple(np.prod(np.array(upsampling_factors), 0)))
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
            scale //= f_up

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
                stage_factories[index](
                    channels_combined,
                    out_channels,
                    n_blocks,
                    scale=scale
                )
            )
            in_channels = out_channels


    @property
    def multi_scale_outputs(self) -> Dict[Scale, int]:
        """
        Return a dictionary specifying the number of channels for each
        stage of the multi-scale outputs of the decoder.
        """
        scale = self.base_scale
        stages = self.stages
        outputs = {}
        for f_up, chans in zip(self.upsampling_factors, self.channels[1:]):
            scale //= f_up
            outputs[scale] = chans
        return outputs


    def forward_multi_scale_output(
            self,
            x: Union[torch.Tensor, Dict[Scale, torch.Tensor]],
            **kwargs

    ) -> Dict[Scale, torch.Tensor]:
        """
        Forward input through decoder and return outputs at all decoder scales.

        Args:
            x: A single tensor corresponding to the decoder input at the
                 decoder's base scale or a dictionary mapping scale to
                 corresponding skip-connection input.
        Return:
            A dictionary mapping the scales of the decoder's stages to
                corresponding outputs.
        """
        outputs = {}
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
                scale //= f_up
                if scale in self.skip_connections and scale != prev_scale:
                    y = stage(cat(x[scale], forward(up, y)), **kwargs)
                else:
                    y = stage(forward(up, y), **kwargs)
                outputs[scale] = y
                prev_scale = scale
        else:
            y = x
            scale = self.base_scale
            stages = self.stages
            for ind, (up, stage) in enumerate(zip(self.upsamplers, stages)):
                f_up = self.upsampling_factors[ind]
                scale //= f_up
                y = stage(forward(up, y), **kwargs)
                outputs[scale] = y

        return outputs


    def forward(
            self,
            x: Union[torch.Tensor, Dict[Scale, torch.Tensor]],
            stage_kwargs: Optional[Dict[Scale, Any]] = None,
            **kwargs
    ) -> torch.Tensor:
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
                scale //= f_up
                if stage_kwargs is not None and scale in stage_kwargs:
                    kwargs_s = stage_kwargs[scale]
                else:
                    kwargs_s = {}
                if scale in self.skip_connections and scale != prev_scale:
                    y = stage(cat(x[scale], forward(up, y)), **kwargs_s, **kwargs)
                else:
                    y = stage(forward(up, y), **kwargs_s, **kwargs)
                prev_scale = scale
        else:
            y = x
            stages = self.stages
            scale = self.base_scale
            for ind, (up, stage) in enumerate(zip(self.upsamplers, stages)):
                f_up = self.upsampling_factors[ind]
                scale //= f_up
                if stage_kwargs is not None and scale in stage_kwargs:
                    kwargs_s = stage_kwargs[scale]
                else:
                    kwargs_s = {}
                y = stage(forward(up, y), **kwargs_s, **kwargs)
        return y


class MultiScalePropagator(nn.Module, ParamCount):
    def __init__(
        self,
        inputs: Dict[Scale, int],
        stage_depths: List[int],
        block_factory: Optional[Callable[[int, int], nn.Module]] = None,
        stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
        upsampler_factory: Callable[[int, int, int], nn.Module] = Trilinear(),
        base_scale: Optional[int] = None,
        residual: bool = True,
        order: int = 2
    ):
        super().__init__()

        self.scales = sorted(inputs.keys(), reverse=True)

        n_stages = len(inputs)

        if block_factory is None:
            block_factory = DEFAULT_BLOCK_FACTORY
        if stage_factory is None:
            if isinstance(block_factory, list):
                if len(block_factory) < n_stages:
                    raise RuntimeError(
                        "If a list of block factories is provided, its length must match "
                        "the number of stages in the propagator."
                    )
                stage_factories = [SequentialStage(b_fac) for b_fac in block_factory]
            else:
                stage_factories = [SequentialStage(block_factory) for _ in range(n_stages)]
        else:
            if isinstance(stage_factory, list):
                raise RuntimeError(
                    "If a list of stage factories is provided, its length must match "
                    "the number of stages in the propagator."
                )
                if isinstance(block_factory, list):
                    if len(block_factory) < n_stages:
                        raise RuntimeError(
                            "If a list of block factories is provided, its length must match "
                            "the number of stages in the propagator."
                        )
                    stage_factories = [s_fac(b_fac) for s_fac, b_fac in zip(stage_factory, block_factory)]
                else:
                    stage_factories = [s_fac(block_factory) for s_fac in stage_factory]
        self.residual = residual

        self.upsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.order = order

        prev_scale = None
        prev_chans = None

        for ind, scale in enumerate(self.scales):

            stage_channels = inputs[scale]
            input_channels = (order + (prev_scale is not None)) * stage_channels
            n_blocks = stage_depths[ind]

            self.stages.append(
                stage_factories[ind](
                    input_channels,
                    stage_channels,
                    n_blocks,
                    scale=scale
                )
            )

            if ind < len(self.scales) - 1:
                next_scale = self.scales[ind + 1]
                next_channels = inputs[next_scale]
                self.upsamplers.append(
                    upsampler_factory(
                        in_channels=stage_channels,
                        out_channels=next_channels,
                        factor=(scale // next_scale).scale
                    )
                )
            else:
                self.upsamplers.append(None)

            prev_scale = scale

    def forward(
            self,
            inputs: Dict[Scale, torch.Tensor],
            n_steps: int
    ) -> List[torch.Tensor]:

        """
        Make multi-scale prediction for n steps.

        Args:
            inputs: A dictionary containing the multi-scale inputs for
                to propagate.
            n_steps: The number of prediction steps at the finest temporal scale.

        Return:
            A list containing the predicted tensors for each forecast step.
        """
        min_scale = min(self.scales)
        prev_preds = None

        pred_ms = {}

        for scale, stage, upsampler in zip(
                self.scales,
                self.stages,
                self.upsamplers,
        ):

            t_scale = scale.scale[0] // min_scale.scale[0]
            scale_steps = ceil(n_steps / t_scale)
            preds = list(torch.unbind(inputs[scale][:, :, -self.order:], 2))

            for step in range(scale_steps):
                if prev_preds is None:
                    inpt = torch.cat(preds[-self.order:], 1)
                else:
                    inpt = torch.cat(preds[-self.order:] + [prev_preds[step]], 1)

                pred_step = stage(inpt)

                if self.residual and prev_preds is not None:
                    pred_step = pred_step + prev_preds[step]

                preds.append(pred_step)

            preds = preds[self.order:]
            pred_ms[scale] = preds

            if upsampler is not None:
                preds = torch.stack(preds, 2)
                preds = torch.unbind(upsampler(preds), 2)

            prev_preds = preds

        return pred_ms
