"""
pytorch_retrieve.modules.conv.encoders
======================================

Generic encoder modules.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from .utils import Scale
from pytorch_retrieve.modules.utils import ParamCount
from pytorch_retrieve.modules.conv.blocks import BasicConv
from pytorch_retrieve.modules.conv.stages import SequentialStage
from pytorch_retrieve.modules.conv.aggregation import Linear


DEFAULT_BLOCK_FACTORY = BasicConv(
    kernel_size=3, normalization_factory=nn.BatchNorm2d, activation_factory=nn.ReLU
)

DEFAULT_AGGREGATOR_FACTORY = Linear



def _calculate_output_scales(base_scale, downsampling_factors):
    """
    Calculate output scales for skip connections.
    """
    base_scale = Scale(base_scale)
    scl = base_scale
    scales = []
    for f_d in downsampling_factors:
        scl = f_d * scl
        scales.append(scl)
    return scales


class Encoder(nn.Module, ParamCount):
    """
    An encoder for spatial information.

    The encoder takes a 4D input (batch x channel x height x width),
    and encodes the input  into one or more feature maps with reduced
    spatial extent but typically higher number of features.

    Due to the downsampling applied between stages of the encoder, each stage
    produces feature maps at different spatial scales. These intermediate
    feature maps are commonly forwarded to a decoder through skip connection.
    The SpatialEncoder class supports both extracting only the encoding at the
    highest spatial scale, i.e. the output from the last encoder stage, or
    extracting features at all intermediate and the final scale of the encoder.
    """

    def __init__(
        self,
        channels: List[int],
        stage_depths: List[int],
        in_channels: Optional[int] = None,
        downsampling_factors: Optional[Union[List[int], int]] = None,
        block_factory: Optional[Callable[[int, int], nn.Module]] = None,
        stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
        downsampler_factory: Callable[[int, int], nn.Module] = None,
        base_scale: Union[int, np.ndarray] = Scale(1),
        skip_connections: bool = True,
    ):
        """
        Args:
            channels: A list specifying the number of features (or channels)
                in each stage of the encoder.
            stage_depths: A list containing the number of block in each stage
                of the encoder.
            in_channels: The number of channels in the encoder input in case
                it deviates from the number of channels in the first stage
                of the encoder.
            downsampling_factors: A list of downsampling factors specifying
                the degree of spatial downsampling applied between all stages
                in the encoder. For a constant downsampling factor
                between all layers this can be set to a single 'int'. Otherwise
                a list of length ``len(channels) - 1`` should be provided.
            block_factory: Factory to create the blocks in each stage.
            stage_factory: Optional stage factory to create the encoder
                stages. Defaults to ``SequentialStage``.
            downsampler_factory: Optional factory to create downsampling
                layers. If not provided, the block factory must provide
                downsampling functionality.
            base_scale: An floating point number representing the scale of the input to
                the encoder. Will be used to calculate the scales corresponding
                to each stage of the encoder.
            skip_connections: If 'True', the encoder will return the outputs
                from the end of each stage, which can forward to a decoder
                that expects skip connections.
        """
        super().__init__()

        n_stages = len(stage_depths)
        self.n_stages = n_stages

        if not len(channels) == self.n_stages:
            raise ValueError(
                "The list of given channel numbers must match the number " "of stages."
            )
        self.channels = channels

        if downsampling_factors is None:
            downsampling_factors = [2] * (n_stages - 1)
        if len(stage_depths) != len(downsampling_factors) + 1:
            raise ValueError(
                "The list of downsampling factors numbers must have one "
                "element less than the number of stages."
            )

        # No downsampling applied in first layer.
        downsampling_factors = [1] + downsampling_factors

        self.base_scale = base_scale
        self.has_skips = skip_connections
        self.scales = _calculate_output_scales(base_scale, downsampling_factors)

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
            if isinstance(block_factory, list):
                if len(block_factory) < n_stages:
                    raise RuntimeError(
                        "If a list of block factories is provided, its length must match "
                        "the number of stages in the encoder."
                    )
                stage_factories = [s_fac(b_fac) for s_fac, b_fac in zip(stage_factory, block_factory)]
            else:
                stage_factories = [stage_factory(block_factory) for _ in range(n_stages)]


        # Populate list of down samplers and stages.
        self.downsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()
        in_channels = channels[0] if in_channels is None else in_channels
        for scale, stage_depth, channels_out, f_dwn, s_fac in zip(
                self.scales, stage_depths, channels, downsampling_factors, stage_factories
        ):
            if downsampler_factory is None:
                self.downsamplers.append(nn.Identity())
                self.stages.append(
                    s_fac(
                        in_channels,
                        channels_out,
                        stage_depth,
                        downsample=f_dwn,
                        scale=scale,
                    )
                )
            else:
                down = f_dwn if isinstance(f_dwn, int) else max(f_dwn)
                if down > 1:
                    self.downsamplers.append(
                        downsampler_factory(in_channels, channels_out, f_dwn)
                    )
                else:
                    self.downsamplers.append(nn.Identity())

                self.stages.append(
                    s_fac(
                        channels_out,
                        channels_out,
                        stage_depth,
                        downsample=None,
                    )
                )
            in_channels = channels_out

    @property
    def skip_connections(self) -> Dict[int, int]:
        """
        Dictionary specifying the number of channels in the skip tensors
        produced by this encoder.
        """
        return {scl: chans for scl, chans in zip(self.scales, self.channels)}

    def forward_with_skips(
            self,
            x: torch.Tensor,
            **kwargs: Dict[Any, Any]
    ) -> Dict[Tuple[int], torch.Tensor]:
        """
        Legacy implementation of the forward_with_skips function of the
        SpatialEncoder.

        Args:
            x: A ``torch.Tensor`` to feed into the encoder.
            kwargs: Keywords to pass down to stages in encoder.

        Return:
            A dictionary mapping scale tuples to corresponding feature
            tensors.
        """
        y = x
        skips = {}
        for scl, down, stage in zip(self.scales, self.downsamplers, self.stages):
            y = stage(down(y), **kwargs)
            skips[scl] = y
        return skips

    def forward(
            self,
            x: torch.Tensor,
            **kwargs: Dict[Any, Any]
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Args:
            x: A ``torch.Tensor`` to feed into the encoder.

        Return:
            If the encoder has skip connections, the output is a dictionary mapping
            scales to the corresponding output tensors from each stage in the
            encoder. If the encoder does not have skip connections, the output is
            just the output tensor from the last stage of the encoder.
        """
        if self.has_skips:
            return self.forward_with_skips(x, **kwargs)

        y = x

        for down, stage in zip(self.downsamplers, self.stages):
            y = stage(down(y), **kwargs)

        return y


class MultiInputSharedEncoder(Encoder, ParamCount):
    """
    The MultiIinputSpatialEncoder is an Encoder  that supports multiple
    inputs with different scales. Each input is fed into the encoder at
    the corresponding scale.
    """

    def __init__(
        self,
        inputs: Dict[str, Tuple[int]],
        channels: List[int],
        stage_depths: List[int],
        downsampling_factors: List[int] = None,
        input_channels: Dict[str, int] = None,
        block_factory: Optional[Callable[[int, int], nn.Module]] = None,
        aggregator_factory: Optional[Callable[[int], nn.Module]] = None,
        stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
        downsampler_factory: Callable[[int, int, int], nn.Module] = None,
        base_scale: Union[int, Tuple[int]] = 1,
        skip_connections: bool = True,
    ):
        """
        Args:
            inputs: A dictionary mapping input names to the
                 corresponding scale of the encoder.
            channels: A list specifying the channels in each stage of the
                 encoder.
            stage_depths: The depth of each stage in the encoder.
            downsampling_factors: A list of integers, tuples or Scale objects
                defining the degree of downsampling applied between consecutive
                stages.
            input_channels: An optional dictionary mapping input names to
                the corresponding number of input channels.
            block_factory: Factory to create the blocks in each stage.
            aggregator_factory: Factory object to use to create the aggregation
                blocks.
            stage_factory: Optional stage factory to create the encoder
                stages. Defaults to ``SequentialStageFactory``.
            downsampler_factory: Optional factory to create downsampling
                layers. If not provided, the block factory must provide
                downsampling functionality.
            base_scale: The scale of the input with the highest resolution.
            skip_connections: If 'True', the encoder will return the outputs
                from the end of each stage, to forward to a decoder that
                expects skip connections.
        """
        # If input dict contain channels use that to determine
        # number of inputs to first stage.

        base_scale = Scale(base_scale)
        scale = base_scale

        scale_chans = {scale: channels[0]}
        for chans, f_d in zip(channels[1:], downsampling_factors):
            scale *= f_d
            if not scale in scale_chans:
                scale_chans[scale] = chans

        input_chans = {}
        for input_name, scl in inputs.items():
            scl = Scale(scl)
            if input_channels is None or input_name not in input_channels:
                input_chans.setdefault(scl, []).append(scale_chans[scl])
            else:
                input_chans.setdefault(scl, []).append(input_channels[input_name])


        in_channels = None
        if len(input_chans) > 0:
            min_scale = min(input_chans.keys())
            in_channels = input_chans[min_scale]
            if len(in_channels) == 1:
                in_channels = in_channels[0]
            else:
                in_channels = None

        super().__init__(
            channels,
            stage_depths,
            in_channels = in_channels,
            downsampling_factors=downsampling_factors,
            block_factory=block_factory,
            stage_factory=stage_factory,
            downsampler_factory=downsampler_factory,
            base_scale=base_scale,
            skip_connections=skip_connections,
        )

        if downsampler_factory is None:
            self.aggregate_after = True
        else:
            self.aggregate_after = False

        if aggregator_factory is None:
            aggregator_factory = DEFAULT_AGGREGATOR_FACTORY

        self.aggregators = nn.ModuleDict()

        # Parse inputs into stage_inputs, which maps stage indices to
        # input names, and create stems.
        self.stage_inputs = {}
        for input_name, scale in inputs.items():
            scale = Scale(scale)
            self.stage_inputs.setdefault(scale, []).append(input_name)

            if not scale in self.scales:
                raise ValueError(
                    f"Input '{input_name}' has scale {scale}, which doesn't "
                    f" match any stage of the encoder."
                )

        # Create aggregators for all inputs at each scale.
        for ind, (scale, names) in enumerate(self.stage_inputs.items()):
            if scale != self.base_scale:
                if scale in self.stage_inputs:
                    inpts = dict(zip(self.stage_inputs[scale], input_chans[scale]))
                    inpts["__enc__"] = scale_chans[scale]
                    self.aggregators[str(scale)] = aggregator_factory(
                        inpts, scale_chans[scale]
                    )
            # Multiple inputs at base scale.
            elif len(names) > 1:
                inpts = dict(zip(self.stage_inputs[scale], input_chans[scale]))
                self.aggregators[str(scale)] = aggregator_factory(
                    inpts, scale_chans[scale]
                )

    def forward_with_skips(self, x: torch.Tensor) -> Dict[set, torch.Tensor]:
        """
        Args:
            x: A ``torch.Tensor`` to feed into the encoder.

        Return:
            A list containing the outputs of each encoder stage with the
            last element in the list corresponding to the output of the
            last encoder stage.
        """
        skips = {}
        y = None

        # Need to keep track of repeating scales to avoid aggregating inputs multiple
        # times.
        prev_scale = -1
        first_stage = True

        for scale, down, stage in zip(self.scales, self.downsamplers, self.stages):
            # Apply downsampling
            if down is not None:
                y = down(y)

            # Collect inputs
            agg_inputs = []
            inputs = self.stage_inputs.get(scale, []) if scale != prev_scale else []

            # Aggregate and propagate through stage
            if y is not None:
                if self.aggregate_after:
                    y = stage(y)
                    first_stage = False

            if y is None and len(inputs) == 1:
                y = stage(x[inputs[0]])
                first_stage = False
            elif len(inputs) > 0:
                agg_inputs = {inpt: x[inpt] for inpt in inputs}
                if y is not None:
                    agg_inputs["__enc__"] = y

                scl = str(scale)
                if scl in self.aggregators:
                    agg = self.aggregators[scl]
                    y = agg(agg_inputs)
                else:
                    y = next(iter(agg_inputs.values()))

            if first_stage or not self.aggregate_after:
                first_stage = False
                y = stage(y)


            prev_scale = scale
            skips[scale] = y

        return skips

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: A dictionary mapping input names to corresponding input
                tensors.

        Return:
            If the encoder has skip connections, the output is a dictionary mapping
            scales to the corresponding output tensors from each stage in the
            encoder. If the encoder does not have skip connections, the output is
            just the output tensor from the last stage of the encoder.
        """
        if not isinstance(x, dict):
            raise ValueError(
                "A multi-input encoder expects a dict of tensors "
                " mapping input names to corresponding input "
                " tensors."
            )

        if self.has_skips:
            return self.forward_with_skips(x)

        y = None

        prev_scale = -1

        for scale, down, stage in zip(self.scales, self.downsamplers, self.stages):
            # Apply downsampling
            if down is not None:
                y = down(y)

            # Collect inputs
            agg_inputs = []
            inputs = self.stage_inputs.get(scale, []) if scale != prev_scale else []

            # Aggregate and propagate through stage
            if y is not None:
                if self.aggregate_after:
                    y = stage(y)

            if y is None and len(inputs) == 1:
                y = stage(x[inputs[0]])
            if len(inputs) > 0:
                agg_inputs = {inpt: x[inpt] for inpt in inputs}
                if y is not None:
                    agg_inputs["__enc__"] = y

                scl = str(scale)
                if scl in self.aggregators:
                    agg = self.aggregators[scl]
                    y = agg(agg_inputs)
                else:
                    y = next(iter(agg_inputs.values()))

            if not self.aggregate_after:
                y = stage(y)

            prev_scale = scale

        return y


class MultiInputParallelEncoder(nn.Module, ParamCount):
    """
    The MultiInputParallelEncoder is a multi-input encoder that handles
    multiple inputs through separate encoders. In contrast to the
    MultiInputSharedEncoder every input has its own encoder.
    """

    def __init__(
        self,
        inputs: Dict[str, int],
        channels: Union[int],
        stage_depths: List[int],
        downsampling_factors: List[int] = None,
        block_factory: Optional[Callable[[int, int], nn.Module]] = None,
        aggregator_factory: Optional[Callable[[int], nn.Module]] = None,
        stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
        downsampler_factory: Callable[[int, int, int], nn.Module] = None,
        base_scale: int = 1,
        skip_connections: bool = True,
    ):
        """
        Args:
            inputs: A dictionary mapping input names to either to the
                 corresponding scale of the encoder input.
            channels: A list specifying the channels in each stage of the
                 encoder.
            stage_depths: The depth of each stage in the encoder.
            block_factory: Factory to create the blocks in each stage.
            stage_factory: Optional stage factory to create the encoder
                stages. Defaults to ``SequentialStageFactory``.
            downsampler_factory: Optional factory to create downsampling
                layers. If not provided, the block factory must provide
                downsampling functionality.
            aggregator_factory: Factory to create block to merge inputs.
            base_scale: The scale of the input with the highest resolution.
            skip_connections: If 'True', the encoder will return the outputs
                from the end of each stage, which can forward to a decoder
                that expects skip connections.
        """
        super().__init__()

        if downsampler_factory is None:
            self.aggregate_after = True
        else:
            self.aggregate_after = False

        if aggregator_factory is None:
            aggregator_factory = DEFAULT_AGGREGATOR_FACTORY

        self.has_skips = skip_connections

        scale = base_scale
        scales = [scale]
        for f_d in downsampling_factors:
            scale *= f_d
            scales.append(scale)
        self.scales = scales

        self.encoders = nn.ModuleDict()
        for inpt, inpt_scale in inputs.items():
            try:
                ind = scales.index(inpt_scale)
                self.encoders[inpt] = Encoder(
                    channels[ind:],
                    stage_depths[ind:],
                    downsampling_factors=downsampling_factors[ind:],
                    block_factory=block_factory,
                    stage_factory=stage_factory,
                    downsampler_factory=downsampler_factory,
                    base_scale=inpt_scale,
                    skip_connections=skip_connections,
                )
            except ValueError:
                raise ValueError(
                    f"The scale of input '{inpt}' doese not match any of the "
                    f" scales of the stages of the encoder ({scales})."
                )

        self.agg_inputs = {}
        for scale in scales:
            agg_inputs = {}
            for inpt, inpt_scale in inputs.items():
                if scale >= inpt_scale:
                    agg_inputs[inpt] = channels[scales.index(scale)]
            self.agg_inputs[scale] = agg_inputs

        if skip_connections:
            self.aggregators = nn.ModuleDict()
            for scl, inpts in self.agg_inputs.items():
                out_channels = channels[scales.index(scl)]
                self.aggregators[str(scl)] = aggregator_factory(inpts, out_channels)
        else:
            inpts = self.agg_inputs[scales[-1]]
            self.aggregator = aggregator_factory(inpts, channels[-1])

    def forward_with_skips(self, x: Dict[str, torch.Tensor]) -> Dict[set, torch.Tensor]:
        """
        Args:
            x: A ``torch.Tensor`` to feed into the encoder.

        Return:
            A list containing the outputs of each encoder stage with the
            last element in the list corresponding to the output of the
            last encoder stage.
        """
        encs = {inpt: self.encoders[inpt](tensor) for inpt, tensor in x.items()}
        results = {}
        for scl in self.scales:
            agg = self.aggregators[str(scl)]
            inpts = {inpt: encs[inpt][scl] for inpt in self.agg_inputs[scl]}

            results[scl] = agg(inpts)
        return results

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: A dictionary mapping input names to corresponding input
                tensors.

        Return:
            If the encoder has skip connections, the output is a dictionary mapping
            scales to the corresponding output tensors from each stage in the
            encoder. If the encoder does not have skip connections, the output is
            just the output tensor from the last stage of the encoder.
        """
        if not isinstance(x, dict):
            raise ValueError(
                "A multi-input encoder expects a dict of tensors "
                " mapping input names to corresponding input "
                " tensors."
            )
        if self.has_skips:
            return self.forward_with_skips(x)

        encs = {inpt: self.encoders[inpt](tensor) for inpt, tensor in x.items()}
        return self.aggregator(encs)


class CascadingEncoder(nn.Module):
    def __init__(
        self,
        channels: Union[int, List[int]],
        stage_depths: List[int],
        downsampling_factors: List[int] = None,
        block_factory: Optional[Callable[[int, int], nn.Module]] = None,
        channel_scaling: int = 2,
        max_channels: int = None,
        downsampler_factory: Callable[[int, int], nn.Module] = None,
        upsampler_factory: Callable[[int, int], nn.Module] = None,
        stem_factory: Callable[[int], nn.Module] = None,
        base_scale: int = 1,
        **kwargs,
    ):
        """
        A cascading encoder processes stages in an overlapping fashion. Each
        stage begins processing as soon as the output from the first
        convolutional block of the previous stage is available. The cascading
        encoder also includes densely connects block across different stages.

        Args:
            channels: A list specifying the number of features (or channels)
                at the end of each stage of the encoder.
            stages: A list containing the stage specifications for each
                stage in the encoder.
            block_factory: Factory to create the blocks in each stage.
            channel_scaling: Scaling factor specifying the increase of the
                number of channels after every downsampling layer. Only used
                if channels is an integer.
            max_channels: Cutoff value to limit the number of channels. Only
                used if channels is an integer.
            downsampler_factory: The downsampler factory is currently ignored.
            downsampling_factors: The downsampling factors applied to the outputs
                of all but the last stage. For a constant downsampling factor
                between all layers this can be set to a single 'int'. Otherwise
                a list of length ``len(channels) - 1`` should be provided.
            stem_factory: A factory that takes a number of output channels and
                produces a stem module that is applied to the inputs prior
                to feeding them into the first stage of the encoder.
        """
        super().__init__()

        if block_factory is None:
            block_factory = DEFAULT_BLOCK_FACTORY

        self.channel_scaling = channel_scaling
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        n_stages = len(stage_depths)
        if isinstance(channels, int):
            channels = [channels * channel_scaling**i for i in range(n_stages)]
            if max_channels is not None:
                channels = [min(ch, max_channels) for ch in channels]
        self.channels = channels

        if not len(channels) == n_stages:
            raise ValueError(
                "The list of given channel numbers must match the number " "of stages."
            )

        if downsampling_factors is None:
            downsampling_factors = [2] * (n_stages - 1)
        if n_stages != len(downsampling_factors) + 1:
            raise ValueError(
                "The list of downsampling factors numbers must have one "
                "element less than the number of stages."
            )

        # No downsampling applied in first layer.
        downsampling_factors = [1] + downsampling_factors
        scale = 1
        self.scales = []
        for f_d in downsampling_factors:
            self.scales.append(scale)
            scale *= f_d

        self.scales = _calculate_output_scales(base_scale, downsampling_factors)

        in_channels = channels[0]
        modules = []
        module_map = {}

        if downsampler_factory is None:

            def downsampler_factory(in_channels, channels_out, f_down):
                return nn.AvgPool2d(kernel_size=f_down, stride=f_down)

        def upsampler_factory(f_up):
            return nn.Upsample(scale_factor=f_up)

        stage_ind = 0
        for n_blocks, channels_out, f_dwn in zip(
            stage_depths, channels, downsampling_factors
        ):
            # Downsampling layer is included in stage.
            down = f_dwn if isinstance(f_dwn, int) else max(f_dwn)
            if down > 1:
                self.downsamplers.append(
                    downsampler_factory(in_channels, channels_out, f_dwn)
                )
                self.upsamplers.append(upsampler_factory(f_dwn))
            else:
                self.downsamplers.append(None)
                self.upsamplers.append(None)

            for block_ind in range(n_blocks):
                chans_combined = in_channels
                if (block_ind > 1) and (stage_ind < n_stages - 1):
                    chans_combined += channels[stage_ind + 1]
                if (
                    (stage_ind > 0)
                    and (block_ind > 0)
                    and (stage_depths[stage_ind - 1] > block_ind)
                ):
                    chans_combined += channels[stage_ind - 1]

                mod = block_factory(
                    chans_combined,
                    channels_out,
                    downsample=None,
                )
                modules.append(mod)
                depth_map = module_map.setdefault(block_ind + stage_ind, [])
                depth_map.append((stage_ind, mod))
                in_channels = channels_out
            stage_ind += 1

        self.modules = nn.ModuleList(modules)
        self.module_map = module_map

        if stem_factory is not None:
            self.stem = stem_factory(channels[0])
        else:
            self.stem = None

        self.depth = max(list(self.module_map.keys())) + 1

    @property
    def skip_connections(self) -> Dict[int, int]:
        """
        Dictionary specifying the number of channels in the skip tensors
        produced by this encoder.
        """
        return {scl: chans for scl, chans in zip(self.scales, self.channels)}

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[int, torch.Tensor]:
        """
        Forward input through encoder.

        Args:
            x: The input tensor.

        Return:
            A dict mapping stage indices to corresponding tensors.
        """
        if self.stem is None:
            x_in = {0: x}
        else:
            x_in = {0: self.stem(x)}

        results = {}

        for d_i in range(self.depth):
            y = {}
            for stage_ind, mod in self.module_map[d_i]:
                inputs = []

                if stage_ind - 1 in x_in:
                    down = self.downsamplers[stage_ind]
                    inputs.append(down(x_in[stage_ind - 1]))

                if stage_ind in x_in:
                    inputs.append(x_in[stage_ind])

                if stage_ind + 1 in x_in:
                    up = self.upsamplers[stage_ind + 1]
                    inputs.append(up(x_in[stage_ind + 1]))

                y[stage_ind] = mod(torch.cat(inputs, 1))

            x_in = y
            results.update(y)

        return {self.scales[ind]: tensor for ind, tensor in results.items()}
