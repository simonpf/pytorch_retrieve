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

        blocks += [
            nn.Conv2d(
                in_channels, out_channels // bottleneck, kernel_size=1, bias=bias
            ),
            normalization_factory(out_channels // bottleneck),
            activation_factory(inplace=True),
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
            activation_factory(inplace=True),
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

ALL = set([
    cls for cls in globals().values()
    if isinstance(cls, type) and issubclass(cls, nn.Module)
])
