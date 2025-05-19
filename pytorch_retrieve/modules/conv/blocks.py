"""
pytorch_retrieve.modules.conv_blocks
====================================

This module provide convolution block factories for reproducing various
CNN architectures.
"""
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from pytorch_retrieve.modules.utils import ParamCount
from .padding import calculate_padding, Reflect, get_padding_factory
from ..normalization import get_normalization_factory
from ..activation import get_activation_factory
from .downsampling import BlurPool
from .projection import get_projection


class BasicConvBlock(nn.Module, ParamCount):
    """
    Implements a basic convolution block with an optional residual
    connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Optional[Union[Tuple[int], int]] = None,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        residual_connection: bool = False,
        downsample: Optional[int] = None,
        anti_aliasing: bool = True,
    ):
        super().__init__()

        self.residual_connection = residual_connection

        if isinstance(downsample, int):
            downsample = (downsample,) * 2

        stride = (1, 1)
        if downsample is not None and max(downsample) > 1:
            stride = downsample

        if normalization_factory is not None:
            bias = False
        else:
            bias = True

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2

        if padding is None:
            padding = calculate_padding(kernel_size)

        if isinstance(padding, int):
            padding = (padding,) * 2

        blocks = []
        if max(padding) > 0:
            blocks.append(padding_factory(padding))

        blocks.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                stride=stride if not anti_aliasing else 1,
            )
        )

        if normalization_factory is not None:
            blocks.append(
                normalization_factory(out_channels),
            )

        if activation_factory is not None:
            blocks.append(
                activation_factory(),
            )

        if max(stride) > 1 and anti_aliasing:
            pad = tuple([1 if strd > 1 else 0 for strd in stride])
            filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
            blocks += [
                padding_factory(pad),
                BlurPool(out_channels, stride, filter_size)
            ]

        self.body = nn.Sequential(*blocks)
        if self.residual_connection:
            if in_channels != out_channels or max(stride) > 1:
                self.projection = nn.Conv2d(
                    in_channels, out_channels, kernel_size=stride, stride=stride
                )
            else:
                self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate tensor through block.
        """
        y = self.body(x)
        if self.residual_connection:
            y = y + self.projection(x)
        return y


class BasicConv:
    """
    Factory for basic convolution blocks.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        residual_connection: bool = False,
        anti_aliasing: bool = False
    ):
        """
        Args:
            kernel_size: Kernel size of the convolution blocks
            padding: Padding applied to the input.
            activation_factory: Factory to use for the instantiation of the
                activation functions.
            normalization_factory: Factory to use for the instantition of the
                normalization layers in the convolution block.
            anti_aliasing: Whether to use anti-aliased downsampling.
        """
        self.kernel_size = kernel_size
        if padding is None:
            padding = self.kernel_size // 2
        self.padding = padding
        if isinstance(activation_factory, str):
            activation_factory = get_activation_factory(activation_factory)
        self.activation_factory = activation_factory
        if isinstance(normalization_factory, str):
            normalization_factory = get_normalization_factory(normalization_factory)
        self.normalization_factory = normalization_factory
        self.residual_connection = residual_connection
        if isinstance(padding_factory, str):
            padding_factory = get_padding_factory(padding_factory)
        self.padding_factory = padding_factory
        self.anti_aliasing = anti_aliasing


    def __call__(
        self, in_channels: int, out_channels: int, downsample: int = 1, **kwargs
    ):
        return BasicConvBlock(
            in_channels,
            out_channels,
            self.kernel_size,
            padding=self.padding,
            downsample=downsample,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            padding_factory=self.padding_factory,
            anti_aliasing=self.anti_aliasing
        )


class BasicConv3dBlock(nn.Module, ParamCount):
    """
    3D version of a basic convolution block consisting of convolution layer
    follow by normalization and activation layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Optional[Union[Tuple[int], int]] = None,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm3d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        residual_connection: bool = False,
        downsample: Optional[int] = None,
        anti_aliasing: bool = False
    ):
        super().__init__()

        self.residual_connection = residual_connection

        if isinstance(downsample, int):
            downsample = (downsample,) * 3

        stride = (1, 1, 1)
        if downsample is not None and max(downsample) > 1:
            stride = downsample

        bias = normalization_factory is None

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        if padding is None:
            padding = calculate_padding(kernel_size)

        if isinstance(padding, int):
            padding = (padding,) * 3
        if padding is None:
            padding = calculate_padding(kernel_size)



        blocks = []
        if max(padding) > 0:
            blocks.append(padding_factory(padding))

        blocks.append(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                stride=stride if not anti_aliasing else 1,
            )
        )

        if normalization_factory is not None:
            blocks.append(
                normalization_factory(out_channels),
            )
        if activation_factory is not None:
            blocks.append(
                activation_factory(),
            )
        if anti_aliasing:
            pad = tuple([1 if strd > 1 else 0 for strd in stride])
            filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
            blocks += [
                padding_factory(pad),
                BlurPool(out_channels, stride, filter_size)
            ]

        self.body = nn.Sequential(*blocks)
        if self.residual_connection:
            if in_channels != out_channels or max(stride) > 1:
                self.projection = nn.Conv2d(
                    in_channels, out_channels, kernel_size=stride, stride=stride
                )
            else:
                self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate tensor through block.
        """
        y = self.body(x)
        if self.residual_connection:
            y = y + self.projection(x)
        return y


class BasicConv3d:
    """
    Factory for basic 3D convolution blocks.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm3d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        residual_connection: bool = False,
        anti_aliasing: bool = False,
    ):
        """
        Args:
            kernel_size: Kernel size of the convolution blocks
            padding: Padding applied to the input.
            activation_factory: Factory to use for the instantiation of the
                activation functions.
            normalization_factory: Factory to use for the instantition of the
                normalization layers in the convolution block.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        padding = calculate_padding(kernel_size)
        self.kernel_size = kernel_size
        self.padding = padding
        if isinstance(activation_factory, str):
            activation_factory = get_activation_factory(activation_factory)
        self.activation_factory = activation_factory
        if isinstance(normalization_factory, str):
            normalization_factory = get_normalization_factory(normalization_factory)
        self.normalization_factory = normalization_factory
        self.residual_connection = residual_connection
        if isinstance(padding_factory, str):
            padding_factory = get_padding_factory(padding_factory)
        self.padding_factory = padding_factory
        self.anti_aliasing = anti_aliasing

    def __call__(
        self, in_channels: int, out_channels: int, downsample: int = 1, **kwargs
    ):
        return BasicConv3dBlock(
            in_channels,
            out_channels,
            self.kernel_size,
            padding=self.padding,
            downsample=downsample,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            padding_factory=self.padding_factory,
            anti_aliasing=self.anti_aliasing
        )


class ResNetBlock(nn.Module, ParamCount):
    """
    Implements a basic ResNet block with optional bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int] = (3, 3),
        padding: Optional[int] = None,
        dilation: int = 1,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        downsample: Optional[int] = None,
        bottleneck: int = 4,
        anti_aliasing: bool = False
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2

        if padding is None:
            padding = (
                (dilation * kernel_size[0]) // 2,
                (dilation * kernel_size[1]) // 2,
            )

        if isinstance(downsample, int):
            downsample = (downsample,) * 2

        bias = normalization_factory is not None

        stride = (1, 1)
        if downsample is not None and max(downsample) > 1:
            stride = downsample

        # Short cut
        if in_channels != out_channels or max(stride) > 1:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=stride, stride=stride
            )
        else:
            self.projection = nn.Identity()

        # Actual body
        blocks = []
        if bottleneck > 1:
            blocks += [
                nn.Conv2d(
                    in_channels, out_channels // bottleneck, kernel_size=1, bias=bias
                ),
                normalization_factory(out_channels // bottleneck),
                activation_factory(inplace=True),
            ]
            in_channels = out_channels // bottleneck

        blocks += [
            padding_factory(padding),
            nn.Conv2d(
                in_channels,
                out_channels // bottleneck,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=bias,
                stride=stride if not anti_aliasing else 1,
            ),
            normalization_factory(out_channels // bottleneck),
            activation_factory(inplace=True),
        ]

        if anti_aliasing:
            pad = tuple([1 if strd > 1 else 0 for strd in stride])
            filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
            blocks.append(
                padding_factory(pad),
                BlurPool(out_channels // bottleneck, stride, filter_size)
            )

        if bottleneck > 1:
            blocks += [
                nn.Conv2d(
                    out_channels // bottleneck, out_channels, kernel_size=1, bias=bias
                ),
                normalization_factory(out_channels),
                activation_factory(inplace=True),
            ]
        else:
            blocks += [
                padding_factory(padding),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                ),
                normalization_factory(out_channels),
                activation_factory(inplace=True),
            ]

        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate tensor through block.
        """
        y = self.body(x)
        y = y + self.projection(x)
        return y


class ResNet:
    """
    Factory for producing standard and bottleneck ResNet blocks.
    """

    def __init__(
        self,
        kernel_size: Optional[Union[Tuple[int, int], int]] = (3, 3),
        dilation: int = 1,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        bottleneck: int = 4,
        anti_aliasing: bool = False,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        self.dilation = dilation
        self.kernel_size = kernel_size
        if isinstance(activation_factory, str):
            activation_factory = get_activation_factory(activation_factory)
        self.activation_factory = activation_factory
        if isinstance(normalization_factory, str):
            normalization_factory = get_normalization_factory(normalization_factory)
        self.normalization_factory = normalization_factory
        if isinstance(padding_factory, str):
            padding_factory = get_padding_factory(padding_factory)
        self.padding_factory = padding_factory
        self.bottleneck = bottleneck
        self.anti_aliasing = anti_aliasing

    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: Optional[Union[Tuple[int, int], int]] = None,
        kernel_size: Optional[Union[Tuple[int, int], int]] = None,
        padding: Optional[int] = None,
        dilation: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Instantiate ResNet module.

        Args:
            in_channels: The number of incoming channels.
            out_channels: The number of outgoing channels.

        Return:
            The ResNet block.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if kernel_size is None:
            kernel_size = self.kernel_size

        if dilation is None:
            dilation = self.dilation

        if padding is None:
            padding = (
                (dilation * kernel_size[0]) // 2,
                (dilation * kernel_size[1]) // 2,
            )
        return ResNetBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            downsample=downsample,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            padding_factory=self.padding_factory,
            bottleneck=self.bottleneck,
        )


class ResNeXtBlock(nn.Module, ParamCount):
    """
    Implements a ResNeXt block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int] = (3, 3),
        downsample: Optional[int] = None,
        padding: Optional[int] = None,
        dilation: int = 1,
        cardinality: int = 32,
        bottleneck: int = 2,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        anti_aliasing: bool = False
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2

        if padding is None:
            padding = (
                (dilation * kernel_size[0]) // 2,
                (dilation * kernel_size[1]) // 2,
            )

        if isinstance(downsample, int):
            downsample = (downsample,) * 2

        bias = normalization_factory is not None

        stride = (1, 1)
        if downsample is not None and max(downsample) > 1:
            stride = downsample

        # Short cut
        if in_channels != out_channels or max(stride) > 1:
            self.projection = nn.Conv2d(
                in_channels, out_channels, kernel_size=stride, stride=stride
            )
        else:
            self.projection = nn.Identity()

        # Actual body
        blocks = []

        activation_kwargs = {}
        if activation_factory == nn.ReLU:
            activation_kwargs["inplace"] = True

        blocks += [
            nn.Conv2d(
                in_channels, out_channels // bottleneck, kernel_size=1, bias=bias
            ),
            normalization_factory(out_channels // bottleneck),
            activation_factory(**activation_kwargs),
            padding_factory(padding),
            nn.Conv2d(
                out_channels // bottleneck,
                out_channels // bottleneck,
                groups=cardinality,
                kernel_size=kernel_size,
                dilation=dilation,
                bias=bias,
                stride=stride if not anti_aliasing else 1,
            ),
            normalization_factory(out_channels // bottleneck),
            activation_factory(**activation_kwargs),
        ]

        if anti_aliasing:
            pad = tuple([1 if strd > 1 else 0 for strd in stride])
            filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
            blocks.append(
                padding_factory(pad),
                BlurPool(out_channels // bottleneck, stride, filter_size)
            )

        blocks += [
            nn.Conv2d(
                out_channels // bottleneck, out_channels, kernel_size=1, bias=bias
            ),
            normalization_factory(out_channels),
            activation_factory(**activation_kwargs),
        ]
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate tensor through block.
        """
        y = self.body(x)
        y = y + self.projection(x)
        return y


class ResNeXt:
    """
    Factory for producing standard ResNeXt blocks.
    """

    def __init__(
        self,
        kernel_size: Optional[Union[Tuple[int, int], int]] = (3, 3),
        dilation: int = 1,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        bottleneck: int = 2,
        cardinality: int = 32,
        anti_aliasing: bool = False
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        self.dilation = dilation
        self.kernel_size = kernel_size
        if isinstance(activation_factory, str):
            activation_factory = get_activation_factory(activation_factory)
        self.activation_factory = activation_factory
        if isinstance(normalization_factory, str):
            normalization_factory = get_normalization_factory(normalization_factory)
        self.normalization_factory = normalization_factory
        if isinstance(padding_factory, str):
            padding_factory = get_padding_factory(padding_factory)
        self.padding_factory = padding_factory
        self.bottleneck = bottleneck
        self.cardinality = cardinality
        self.anti_aliasing = anti_aliasing

    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: Optional[Union[Tuple[int, int], int]] = None,
        kernel_size: Optional[Union[Tuple[int, int], int]] = None,
        padding: Optional[int] = None,
        dilation: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Instantiate ResNet module.

        Args:
            in_channels: The number of incoming channels.
            out_channels: The number of outgoing channels.

        Return:
            The ResNet block.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if kernel_size is None:
            kernel_size = self.kernel_size

        if dilation is None:
            dilation = self.dilation

        if padding is None:
            padding = (
                (dilation * kernel_size[0]) // 2,
                (dilation * kernel_size[1]) // 2,
            )
        return ResNeXtBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            downsample=downsample,
            padding=padding,
            dilation=dilation,
            cardinality=self.cardinality,
            bottleneck=self.bottleneck,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            padding_factory=self.padding_factory,
            anti_aliasing=self.anti_aliasing
        )




class ResNeXt2Plus1Block(nn.Module, ParamCount):
    """
    ResNeXt version of the R(2 + 1)D network.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Tuple[int, int], int] = (3, 3, 3),
        downsample: Optional[int] = None,
        padding: Optional[int] = None,
        dilation: int = 1,
        cardinality: int = 32,
        bottleneck: int = 2,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        kernel_size= tuple(kernel_size)

        if isinstance(downsample, int):
            downsample = (downsample,) * 3

        bias = normalization_factory is not None

        stride = (1, 1, 1)
        if downsample is not None and max(downsample) > 1:
            downsample = tuple(downsample)
            stride = downsample

        # Short cut
        if in_channels != out_channels or max(stride) > 1:
            self.projection = nn.Conv3d(
                in_channels, out_channels, kernel_size=stride, stride=stride
            )
        else:
            self.projection = nn.Identity()

        spatial_kernel = (1,) + kernel_size[1:]
        spatial_padding = calculate_padding(spatial_kernel)
        spatial_stride = (1,) + stride[1:]
        temporal_kernel = kernel_size[:1] + (1, 1)
        temporal_padding = calculate_padding(temporal_kernel)
        temporal_stride = stride[:1] + (1, 1)

        # Actual body
        blocks = [
            nn.Conv3d(
                in_channels, out_channels // bottleneck, kernel_size=1, bias=bias
            ),
            normalization_factory(out_channels // bottleneck),
            activation_factory(inplace=True),
            padding_factory(spatial_padding),
            nn.Conv3d(
                out_channels // bottleneck,
                out_channels // bottleneck,
                groups=cardinality,
                kernel_size=spatial_kernel,
                dilation=dilation,
                bias=bias,
                stride=spatial_stride,
            ),
            activation_factory(inplace=True),
            normalization_factory(out_channels // bottleneck),
            padding_factory(temporal_padding),
            nn.Conv3d(
                out_channels // bottleneck,
                out_channels // bottleneck,
                groups=cardinality,
                kernel_size=temporal_kernel,
                dilation=dilation,
                bias=bias,
                stride=temporal_stride,
            ),
            normalization_factory(out_channels // bottleneck),
            activation_factory(inplace=True),
            nn.Conv3d(
                out_channels // bottleneck, out_channels, kernel_size=1, bias=bias
            ),
            normalization_factory(out_channels),
            activation_factory(inplace=True),
        ]
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate tensor through block.
        """
        y = self.body(x)
        y = y + self.projection(x)
        return y


class ResNeXt2Plus1:
    """
    Factory for producing standard ResNeXt versions of the 2+1 D ResNet.
    """
    def __init__(
        self,
        kernel_size: Optional[Union[Tuple[int, int], int]] = (3, 3, 3),
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm3d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        bottleneck: int = 2,
        cardinality: int = 32,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        self.kernel_size = kernel_size
        self.activation_factory = activation_factory
        if isinstance(normalization_factory, str):
            normalization_factory = get_normalization_factory(normalization_factory)
        self.normalization_factory = normalization_factory
        if isinstance(padding_factory, str):
            padding_factory = get_padding_factory(padding_factory)
        self.padding_factory = padding_factory
        self.bottleneck = bottleneck
        self.cardinality = cardinality

    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: Optional[Union[Tuple[int, int], int]] = None,
        kernel_size: Optional[Union[Tuple[int, int], int]] = None,
        padding: Optional[int] = None,
        dilation: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Instantiate ResNet module.

        Args:
            in_channels: The number of incoming channels.
            out_channels: The number of outgoing channels.

        Return:
            The ResNet block.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if kernel_size is None:
            kernel_size = self.kernel_size


        padding = calculate_padding(kernel_size)

        return ResNeXt2Plus1Block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            downsample=downsample,
            padding=padding,
            cardinality=self.cardinality,
            bottleneck=self.bottleneck,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            padding_factory=self.padding_factory
        )


class SqueezeExcite(nn.Module):
    """
    Squeeze and excite module.
    """
    def __init__(
            self,
            in_channels: int,
            ratio: float = 0.25,
            activation_factory: Callable[[], nn.Module] = nn.ReLU,
            gate_factory: Callable[[], nn.Module] = nn.Sigmoid
    ):
        """
        Args:
        in_channels: The number of incoming channels.
            ratio: Reduction ratio
            activat_factory: An activation factory to use to create the activation function used within
                the SE block.
            gate_factory: Function to use for the gatin.
        """
        super().__init__()
        hidden_channels = int(ratio * in_channels)
        self.conv_reduce = nn.Conv2d(in_channels, hidden_channels, 1, bias=True)
        self.act = activation_factory()
        self.conv_expand = nn.Conv2d(hidden_channels, in_channels, 1, bias=True)
        self.gate = gate_factory()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((-2, -1), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class PWSqueezeExcite(nn.Module):
    """
    Squeeze and excite module.
    """
    def __init__(
            self,
            in_channels: int,
            ratio: float = 0.25,
            activation_factory: Callable[[], nn.Module] = nn.ReLU,
            gate_factory: Callable[[], nn.Module] = nn.Sigmoid
    ):
        """
        Args:
        in_channels: The number of incoming channels.
            ratio: Reduction ratio
            activat_factory: An activation factory to use to create the activation function used within
                the SE block.
            gate_factory: Function to use for the gatin.
        """
        super().__init__()
        hidden_channels = int(ratio * in_channels)
        self.conv_reduce = nn.Conv2d(in_channels, hidden_channels, 1, bias=True)
        self.act = activation_factory()
        self.conv_expand = nn.Conv2d(hidden_channels, in_channels, 1, bias=True)
        self.gate = gate_factory()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = self.conv_reduce(x)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class InvertedBottleneckBlock(nn.Module, ParamCount):
    """
    Inverted-bottleneck block is used in MobileNet and Efficient net where it is referred
    to as MBConv
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion_factor: int = 4,
            kernel_size: int = 3,
            excitation_ratio: float = 0.0,
            activation_factory: Callable[[], nn.Module] = nn.ReLU,
            normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
            padding: Optional[Tuple[int]] = None,
            padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
            downsample: Optional[int] = None,
            anti_aliasing: bool = False,
            fused: bool = False,
            stochastic_depth: Optional[float] = None,
            align: bool = False
    ):
        super().__init__()
        self.act = activation_factory()
        act = activation_factory()

        hidden_channels = out_channels * expansion_factor
        self.stochastic_depth = stochastic_depth

        stride = (1, 1)
        if downsample is not None:
            if isinstance(downsample, int):
                downsample = (downsample,) * 2
            if max(downsample) > 1:
                stride = downsample

        self.projection = get_projection(
            in_channels,
            out_channels,
            stride=stride,
            anti_aliasing=anti_aliasing,
            padding_factory=padding_factory
        )

        if padding is None:
            padding = calculate_padding(kernel_size)

        blocks = []
        if not fused:
            blocks += [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
                normalization_factory(hidden_channels),
                act
            ]

            if max(stride) > 1 and anti_aliasing:
                pad = tuple([1 if strd > 1 else 0 for strd in stride])
                filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
                blocks += [
                    padding_factory(pad),
                    BlurPool(hidden_channels, (1, 1), filter_size)
                ]

            blocks += [
                padding_factory(padding),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size if (max(stride) < 2 or not align) else stride,
                    stride=stride,
                    groups=hidden_channels,
                ),
                normalization_factory(hidden_channels),
                act
            ]
        else:
            if max(stride) > 1 and anti_aliasing:
                pad = tuple([1 if strd > 1 else 0 for strd in stride])
                filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
                blocks += [
                    padding_factory(pad),
                    BlurPool(in_channels, (1, 1), filter_size)
                ]

            blocks += [
                padding_factory(padding),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                normalization_factory(hidden_channels),
                act
            ]

        if excitation_ratio > 0.0:
            blocks.append(
                PWSqueezeExcite(
                    hidden_channels,
                    excitation_ratio,
                    activation_factory
                )
            )
        blocks += [
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            normalization_factory(out_channels),
            act
        ]
        self.body = nn.Sequential(*blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate input through layer.
        """
        shortcut = self.projection(x)

        ## Apply stochastic depth.
        #if self.stochastic_depth is not None and self.training:
        #    p = torch.rand(1)
        #    if p <= self.stochastic_depth:
        #        return shortcut + self.body(x)
        #    return shortcut

        return shortcut + self.body(x)


class InvertedBottleneck:
    """
    Factory for producing inverted bottleneck blocks.
    """
    def __init__(
        self,
        kernel_size: Optional[Union[Tuple[int, int], int]] = (3, 3),
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        expansion_factor: int = 4,
        excitation_ratio: float = 0.0,
        anti_aliasing: bool = False,
        fused: bool = False,
        stochastic_depth: Optional[float] = None
    ):
        """
        Args:
            kernel_size: The size of the convolution kernel used in the
                grouped convolution.
            activation_factory: The factory to use to produce the activation
                function for the block.
            normalization_factory: The factory to use to produce the normalization
                layers in the block.
            padding_factory: The factory to use to produce the padding modules
                used in the block.
            expansion_factor: The factor by which to expand the channels in
                the inverted bottleneck.
            excitation_ratio: The excitation ratio to use in the squeeze and
                excite block. If <= 0.0, the squeeze-and-excite block is omitted.
            anti_aliasing: Whether to apply anit-aliasing filter prior to
                downsampling.
            fused: It 'True', the expansion and grouped convolution are fused
                into a single convolution, which is more performant for large
                image sizes.
            stochastic_depth: Survival probability of the block for stochastic depth. Disabled
                if 'None'.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        self.kernel_size = kernel_size
        if isinstance(activation_factory, str):
            activation_factory = get_activation_factory(activation_factory)
        self.activation_factory = activation_factory
        if isinstance(normalization_factory, str):
            normalization_factory = get_normalization_factory(normalization_factory)
        self.normalization_factory = normalization_factory
        if isinstance(padding_factory, str):
            padding_factory = get_padding_factory(padding_factory)
        self.padding_factory = padding_factory
        self.expansion_factor = expansion_factor
        self.excitation_ratio = excitation_ratio
        self.anti_aliasing = anti_aliasing
        self.fused = fused
        self.stochastic_depth = stochastic_depth

    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: Optional[Union[Tuple[int, int], int]] = None,
        kernel_size: Optional[Union[Tuple[int, int], int]] = None,
        padding: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Instantiate InvertedBottleneck module.

        Args:
            in_channels: The number of incoming channels.
            out_channels: The number of outgoing channels.
            downsample: An integer or tuple of integers specifying the
                downsampling to apply in the block.
            kernel_size: An optional kernel argument to overwrite the
                default kernel size of the factory.
            padding: An iteger specifying the padding to apply to each
                side of the spatial dimensions of the input tensor.

        Return:
            The inverted-bottleneck block.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if kernel_size is None:
            kernel_size = self.kernel_size

        if padding is None:
            padding = (
                (kernel_size[0]) // 2,
                (kernel_size[1]) // 2,
            )
        return InvertedBottleneckBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            downsample=downsample,
            padding=padding,
            expansion_factor=self.expansion_factor,
            excitation_ratio=self.excitation_ratio,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            padding_factory=self.padding_factory,
            anti_aliasing=self.anti_aliasing,
            fused=self.fused,
            stochastic_depth=self.stochastic_depth
        )


class PWSqueezeExcite3D(nn.Module):
    """
    3D pointwise squeeze and excite module.
    """
    def __init__(
            self,
            in_channels: int,
            ratio: float = 0.25,
            activation_factory: Callable[[], nn.Module] = nn.ReLU,
            gate_factory: Callable[[], nn.Module] = nn.Sigmoid
    ):
        """
        Args:
        in_channels: The number of incoming channels.
            ratio: Reduction ratio
            activat_factory: An activation factory to use to create the activation function used within
                the SE block.
            gate_factory: Function to use for the gatin.
        """
        super().__init__()
        hidden_channels = int(ratio * in_channels)
        self.conv_reduce = nn.Conv3d(in_channels, hidden_channels, 1, bias=True)
        self.act = activation_factory()
        self.conv_expand = nn.Conv3d(hidden_channels, in_channels, 1, bias=True)
        self.gate = gate_factory()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = self.conv_reduce(x)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class InvertedBottleneck2Plus1Block(nn.Module, ParamCount):
    """
    (2 + 1)-dimensional version of the standard inverted bottleneck block.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion_factor: int = 4,
            kernel_size: int = 3,
            excitation_ratio: float = 0.0,
            activation_factory: Callable[[], nn.Module] = nn.ReLU,
            normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm3d,
            padding: Optional[Tuple[int]] = None,
            padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
            downsample: Optional[int] = None,
            anti_aliasing: bool = False,
            fused: bool = False,
            stochastic_depth: Optional[float] = None
    ):
        """
        Args:
            in_channels: The number of incoming channels.
            out_channels: The number of outgoing channels.
            expansion_factor: The factor by which to expand the channels in
                the inverted bottleneck.
            kernel_size: The size of the convolution kernel used in the
                grouped convolution.
            activation_factory: The factory to use to produce the activation
                function for the block.
            normalization_factory: The factory to use to produce the normalization
                layers in the block.
            padding: The padding to apply before the spatial convolution.
            padding_factory: The factory to use to produce the padding modules
                used in the block.
            excitation_ratio: The excitation ratio to use in the squeeze and
                excite block. If <= 0.0, the squeeze-and-excite block is omitted.
            anti_aliasing: Whether to apply anit-aliasing filter prior to
                downsampling.
            fused: It 'True', the expansion and grouped convolution are fused
                into a single convolution, which is more performant for large
                image sizes.
            stochastic_depth: Survival rate for stochastic depth. Disabled if 'None'.
        """
        super().__init__()
        self.act = activation_factory()
        act = activation_factory()

        hidden_channels = out_channels * expansion_factor

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        else:
            kernel_size = tuple(kernel_size)

        stride = (1, 1, 1)
        if downsample is not None:
            if isinstance(downsample, int):
                downsample = (downsample,) * 3
            else:
                downsample = tuple(downsample)
            if max(downsample) > 1:
                stride = downsample

        self.stochastic_depth = stochastic_depth

        self.projection = get_projection(
            in_channels,
            out_channels,
            stride=stride,
            anti_aliasing=anti_aliasing,
            padding_factory=padding_factory
        )

        if padding is None:
            padding = calculate_padding(kernel_size)

        k_spatial = (1,) + kernel_size[1:]
        s_spatial = (1,) + stride[1:]
        p_spatial = (0,) + padding[1:]
        k_temporal = kernel_size[:1] + (1, 1)
        s_temporal = stride[:1] + (1, 1)
        p_temporal = padding[:1] + (0, 0)

        blocks = []
        if not fused:
            blocks += [
                nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
                normalization_factory(hidden_channels),
                act
            ]

            if max(stride) > 1 and anti_aliasing:
                pad = tuple([1 if strd > 1 else 0 for strd in stride])
                filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
                blocks += [
                    padding_factory(pad),
                    BlurPool(hidden_channels, (1, 1, 1), filter_size)
                ]

            blocks += [
                padding_factory(p_spatial),
                nn.Conv3d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=k_spatial,
                    stride=s_spatial,
                    groups=hidden_channels,
                ),
                normalization_factory(hidden_channels),
                act
            ]
            if max(k_temporal) > 1:
                blocks += [
                    padding_factory(p_temporal),
                    nn.Conv3d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=k_temporal,
                        stride=s_temporal,
                        groups=hidden_channels,
                    ),
                    normalization_factory(hidden_channels),
                    act
                ]
        else:
            if max(stride) > 1 and anti_aliasing:
                pad = tuple([1 if strd > 1 else 0 for strd in stride])
                filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
                blocks += [
                    padding_factory(pad),
                    BlurPool(in_channels, (1, 1, 1), filter_size)
                ]

            blocks += [
                padding_factory(p_spatial),
                nn.Conv3d(
                    in_channels,
                    hidden_channels,
                    kernel_size=k_spatial,
                    stride=s_spatial,
                ),
                normalization_factory(hidden_channels),
                act
            ]
            if max(k_temporal) > 1:
                blocks += [
                    padding_factory(p_temporal),
                    nn.Conv3d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=k_temporal,
                        stride=s_temporal,
                    ),
                    normalization_factory(hidden_channels),
                    act
                ]

        if excitation_ratio > 0.0:
            blocks.append(
                PWSqueezeExcite3D(
                    hidden_channels,
                    excitation_ratio,
                    activation_factory
                )
            )

        blocks += [
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
            normalization_factory(out_channels),
            act
        ]
        if stochastic_depth is not None:
            from torchvision.ops import StochasticDepth
            blocks.append(StochasticDepth(1.0 - stochastic_depth, "row"))
        self.body = nn.Sequential(*blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate input through layer.
        """
        shortcut = self.projection(x)
        return shortcut + self.body(x)


class InvertedBottleneck2Plus1:
    """
    Factory for producing inverted bottleneck blocks.
    """
    def __init__(
        self,
        kernel_size: Optional[Union[Tuple[int, int], int]] = (3, 3, 3),
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm3d,
        padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
        expansion_factor: int = 4,
        excitation_ratio: float = 0.0,
        anti_aliasing: bool = False,
        fused: bool = False,
        stochastic_depth: Optional[float] = None
    ):
        """
        Args:
            kernel_size: The size of the convolution kernel used in the
                grouped convolution.
            activation_factory: The factory to use to produce the activation
                function for the block.
            normalization_factory: The factory to use to produce the normalization
                layers in the block.
            padding_factory: The factory to use to produce the padding modules
                used in the block.
            expansion_factor: The factor by which to expand the channels in
                the inverted bottleneck.
            excitation_ratio: The excitation ratio to use in the squeeze and
                excite block. If <= 0.0, the squeeze-and-excite block is omitted.
            anti_aliasing: Whether to apply anit-aliasing filter prior to
                downsampling.
            fused: It 'True', the expansion and grouped convolution are fused
                into a single convolution, which is more performant for large
                image sizes.
            stochastic_depth: Survival rate for stochastic depth. 'None' to disable
                stochastic depth.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        self.kernel_size = kernel_size
        if isinstance(activation_factory, str):
            activation_factory = get_activation_factory(activation_factory)
        self.activation_factory = activation_factory
        if isinstance(normalization_factory, str):
            normalization_factory = get_normalization_factory(normalization_factory)
        self.normalization_factory = normalization_factory
        if isinstance(padding_factory, str):
            padding_factory = get_padding_factory(padding_factory)
        self.padding_factory = padding_factory
        self.expansion_factor = expansion_factor
        self.excitation_ratio = excitation_ratio
        self.anti_aliasing = anti_aliasing
        self.fused = fused
        self.stochastic_depth = stochastic_depth

    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: Optional[Union[Tuple[int, int], int]] = None,
        kernel_size: Optional[Union[Tuple[int, int], int]] = None,
        padding: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Instantiate InvertedBottleneck module.

        Args:
            in_channels: The number of incoming channels.
            out_channels: The number of outgoing channels.
            downsample: An integer or tuple of integers specifying the
                downsampling to apply in the block.
            kernel_size: An optional kernel argument to overwrite the
                default kernel size of the factory.
            padding: An iteger specifying the padding to apply to each
                side of the spatial dimensions of the input tensor.

        Return:
            The inverted-bottleneck block.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if kernel_size is None:
            kernel_size = self.kernel_size

        if padding is None:
            padding = (
                (kernel_size[0]) // 2,
                (kernel_size[1]) // 2,
                (kernel_size[2]) // 2,
            )
        return InvertedBottleneck2Plus1Block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            downsample=downsample,
            padding=padding,
            expansion_factor=self.expansion_factor,
            excitation_ratio=self.excitation_ratio,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            padding_factory=self.padding_factory,
            anti_aliasing=self.anti_aliasing,
            fused=self.fused,
            stochastic_depth=self.stochastic_depth
        )


ALL = set([
    cls for cls in globals().values()
    if isinstance(cls, type) and issubclass(cls, nn.Module)
])



class SatformerBlock(nn.Module, ParamCount):
    """
    Convolutional block that operates on sequences of images and applies self
    attention between the image sequences.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion_factor: int = 4,
            kernel_size: int = 3,
            activation_factory: Callable[[], nn.Module] = nn.ReLU,
            normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
            padding: Optional[Tuple[int]] = None,
            attention: bool = True,
            padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
            n_heads: int = 2,
            dropout: Optional[float] = None,
            excitation_ratio: float = 0.0,
            downsample: Optional[int] = None,
            anti_aliasing: bool = False,
            fused: bool = False,
            stochastic_depth: Optional[float] = None
    ):
        """
        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels to put out. This number of output channels is used to calculate
                 the number of channels in the inverted bottle neck.
            expansion_factor: Increase in number of channels in the inverted bottleneck.
            kernel_size: The kernel size in the grouped convolution.
            activation_factory: Factory for the activation functions used in the block.
            normalization_factory: Factory for the normalization layers used in the block.
            padding: Padding applied to the spatial dimensions of the input.
            attention: Whether or not to include an attention layer in the block.
            padding_factory: Factory for the padding layers applied in the block.
            n_heads: The number fo heads in the attention module.
            dropout: The dropout to apply in the attention layer.
            excitation_ratio: The excitation ratio to apply in the squeeze and excite block.
            anti_aliasing: Wheter to apply anti-aliasing.
            fused: Whether or not to fuse the first two blocks.
            stochastic_depth: If given, survival probability for stochastic depth.
        """
        super().__init__()
        self.act = activation_factory()

        hidden_channels = out_channels * expansion_factor

        stride = (1, 1)
        if downsample is not None:
            if isinstance(downsample, int):
                downsample = (downsample,) * 2
            if max(downsample) > 1:
                stride = downsample

        if padding is None:
            padding = calculate_padding(kernel_size)

        if max(stride) > 1 or in_channels != out_channels:
            blocks = []
            if max(stride) > 1:
                blocks.append(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                )
            blocks += [
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                normalization_factory(out_channels)
            ]
            self.downsampling_block = nn.Sequential(*blocks)
            in_channels = out_channels
        else:
            self.downsampling_block = None

        blocks = []
        if not fused:
            blocks += [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            ]

            blocks += [
                padding_factory(padding),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    groups=hidden_channels,
                ),
                normalization_factory(hidden_channels),
                self.act
            ]
        else:
            blocks += [
                padding_factory(padding),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                ),
                normalization_factory(hidden_channels),
                self.act
            ]

        if excitation_ratio > 0.0:
            blocks.append(
                PWSqueezeExcite(
                    hidden_channels,
                    excitation_ratio,
                    activation_factory
                )
            )

        blocks += [
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            normalization_factory(out_channels),
            self.act
        ]

        if stochastic_depth is not None:
            from torchvision.ops import StochasticDepth
            blocks.append(StochasticDepth(1.0 - stochastic_depth, "row"))

        self.conv_body = nn.Sequential(*blocks)

        if attention:
            self.att_norm = nn.LayerNorm(out_channels)
            self.attention = nn.MultiheadAttention(
                embed_dim=out_channels,
                num_heads=n_heads,
                batch_first=True
            )
            if stochastic_depth is not None:
                from torchvision.ops import StochasticDepth
                self.stochastic_depth = StochasticDepth(1.0 - stochastic_depth, "row")
            else:
                self.stochastic_depth = nn.Identity()
        else:
            self.attention = None
            self.stochast_depth = None

    def forward(
            self,
            x: torch.Tensor,
            x_in: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            source_seq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Propagate input through layer.
        """
        if x.dim() < 5:
            x = x[:, :, None]
            collapse_seq = True
        else:
            collapse_seq = False


        n_batch, _, n_seq, *_ = x.shape

        # Shape: [n_batch * n_seq, n_chans, n_y, n_x]
        x_conv = x.transpose(1, 2).flatten(0, 1)

        if self.downsampling_block is not None:
            x_conv = self.downsampling_block(x_conv)
        n_y, n_x = x_conv.shape[-2:]

        x = x_conv + self.conv_body(x_conv)
        n_embed = x.shape[1]

        # Restore sequence dimensions
        # Shape: [n_batch, n_seq, n_embed, n_y, n_x]
        x = x.unflatten(0, (n_batch, n_seq))


        if self.attention is not None:
            if mask is not None:
                mask = mask.repeat_interleave(n_x * n_y, 0)

            # Shape: [n_batch, n_y, n_x, n_seq, n_embed] -> [n_batch * n_y * n_x, n_seq, n_embed]
            x_att = self.att_norm(torch.permute(x, (0, 3, 4, 1, 2)).reshape(-1, n_seq, n_embed))

            if x_in is None:
                x_in = x_att
            else:
                n_seq_in = x_in.shape[2]
                x_in = torch.permute(x_in, (0, 3, 4, 2, 1)).reshape(-1, n_seq_in, n_embed)

            x_att, _ = self.attention(x_att, x_in, x_in, key_padding_mask=mask, attn_mask=attn_mask)
            # Shape: [n_batch, n_y, n_x, n_seq, n_embed]
            x_att = x_att.reshape((n_batch, n_y, n_x, n_seq, n_embed))
            # Shape: [n_batch, n_seq, n_embed, n_y, n_x]
            x_att = x_att.permute(0, 3, 4, 1, 2)
            x = x + self.stochastic_depth(x_att)

        x = x.transpose(1, 2)
        if collapse_seq:
            x = x[:, :, 0]

        return x


class Satformer():
    """
    Factory for Satformer blocks.
    """
    def __init__(
            self,
            kernel_size: Optional[Union[Tuple[int, int], int]] = (3, 3),
            activation_factory: Callable[[], nn.Module] = nn.ReLU,
            normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
            padding_factory: Callable[[Union[Tuple[int], int]], nn.Module] = Reflect,
            expansion_factor: int = 4,
            excitation_ratio: float = 0.0,
            anti_aliasing: bool = False,
            fused: bool = False,
            attention: bool = True,
            n_heads: bool = 4,
            dropout: float = 0.0,
            stochastic_depth: Optional[float] = None
    ):
        """
        Args:
            kernel_size: The size of the convolution kernel used in the
                grouped convolution.
            activation_factory: The factory to use to produce the activation
                function for the block.
            normalization_factory: The factory to use to produce the normalization
                layers in the block.
            padding_factory: The factory to use to produce the padding modules
                used in the block.
            expansion_factor: The factor by which to expand the channels in
                the inverted bottleneck.
            excitation_ratio: The excitation ratio to use in the squeeze and
                excite block. If <= 0.0, the squeeze-and-excite block is omitted.
            anti_aliasing: Whether to apply anit-aliasing filter prior to
                downsampling.
            fused: It 'True', the expansion and grouped convolution are fused
                into a single convolution, which is more performant for large
                image sizes.
            attention: Whether to include cross-channel attention in this layer.
            n_heads: The number of attention heads.
            dropout: Whether to apply dropout in the attention layer.
            stochastic_depth: If given, survival probability for stochastic depth.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        self.kernel_size = kernel_size
        if isinstance(activation_factory, str):
            activation_factory = get_activation_factory(activation_factory)
        self.activation_factory = activation_factory
        if isinstance(normalization_factory, str):
            normalization_factory = get_normalization_factory(normalization_factory)
        self.normalization_factory = normalization_factory
        if isinstance(padding_factory, str):
            padding_factory = get_padding_factory(padding_factory)
        self.padding_factory = padding_factory
        self.expansion_factor = expansion_factor
        self.excitation_ratio = excitation_ratio
        self.anti_aliasing = anti_aliasing
        self.fused = fused
        self.attention = attention
        self.n_heads = n_heads
        self.dropout = dropout
        self.stochastic_depth = stochastic_depth

    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: Optional[Union[Tuple[int, int], int]] = None,
        kernel_size: Optional[Union[Tuple[int, int], int]] = None,
        padding: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Instantiate InvertedBottleneck module.

        Args:
            in_channels: The number of incoming channels.
            out_channels: The number of outgoing channels.
            downsample: An integer or tuple of integers specifying the
                downsampling to apply in the block.
            kernel_size: An optional kernel argument to overwrite the
                default kernel size of the factory.
            padding: An iteger specifying the padding to apply to each
                side of the spatial dimensions of the input tensor.

        Return:
            The inverted-bottleneck block.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if kernel_size is None:
            kernel_size = self.kernel_size

        if padding is None:
            padding = (
                (kernel_size[0]) // 2,
                (kernel_size[1]) // 2,
            )
        return SatformerBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            downsample=downsample,
            padding=padding,
            expansion_factor=self.expansion_factor,
            excitation_ratio=self.excitation_ratio,
            activation_factory=self.activation_factory,
            normalization_factory=self.normalization_factory,
            padding_factory=self.padding_factory,
            fused=self.fused,
            attention=self.attention,
            n_heads=self.n_heads,
            dropout=self.dropout,
            stochastic_depth=self.stochastic_depth
        )
