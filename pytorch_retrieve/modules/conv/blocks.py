"""
pytorch_retrieve.modules.conv_blocks
====================================

This module provide convolution block factories for reproducing various
CNN architectures.
"""
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from pytorch_retrieve.modules.utils import ParamCount
from .padding import calculate_padding


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
        padding: int,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm2d,
        residual_connection: bool = False,
        downsample: Optional[int] = None,
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

        blocks = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                stride=stride,
                padding=padding,
            )
        ]
        if normalization_factory is not None:
            blocks.append(
                normalization_factory(out_channels),
            )
        if activation_factory is not None:
            blocks.append(
                activation_factory(),
            )

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
        residual_connection: bool = False,
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
        self.kernel_size = kernel_size
        if padding is None:
            padding = self.kernel_size // 2
        self.padding = padding
        self.activation_factory = activation_factory
        self.normalization_factory = normalization_factory
        self.residual_connection = residual_connection

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
        padding: int,
        activation_factory: Callable[[], nn.Module] = nn.ReLU,
        normalization_factory: Callable[[int], nn.Module] = nn.BatchNorm3d,
        residual_connection: bool = False,
        downsample: Optional[int] = None,
    ):
        super().__init__()

        self.residual_connection = residual_connection

        if isinstance(downsample, int):
            downsample = (downsample,) * 3

        stride = (1, 1, 1)
        if downsample is not None and max(downsample) > 1:
            stride = downsample

        bias = normalization_factory is None

        blocks = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                stride=stride,
                padding=padding,
            )
        ]
        if normalization_factory is not None:
            blocks.append(
                normalization_factory(out_channels),
            )
        if activation_factory is not None:
            blocks.append(
                activation_factory(),
            )

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
        residual_connection: bool = False,
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
        self.activation_factory = activation_factory
        self.normalization_factory = normalization_factory
        self.residual_connection = residual_connection

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
        downsample: Optional[int] = None,
        bottleneck: int = 4,
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
            nn.Conv2d(
                in_channels,
                out_channels // bottleneck,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=bias,
                stride=stride,
            ),
            normalization_factory(out_channels // bottleneck),
            activation_factory(inplace=True),
        ]

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
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
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
        bottleneck: int = 4,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.activation_factory = activation_factory
        self.normalization_factory = normalization_factory
        self.bottleneck = bottleneck

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
            nn.Conv2d(
                out_channels // bottleneck,
                out_channels // bottleneck,
                groups=cardinality,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=bias,
                stride=stride,
            ),
            normalization_factory(out_channels // bottleneck),
            activation_factory(inplace=True),
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
        bottleneck: int = 2,
        cardinality: int = 32,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.activation_factory = activation_factory
        self.normalization_factory = normalization_factory
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
            nn.Conv3d(
                out_channels // bottleneck,
                out_channels // bottleneck,
                groups=cardinality,
                kernel_size=spatial_kernel,
                dilation=dilation,
                padding=spatial_padding,
                bias=bias,
                stride=spatial_stride,
            ),
            activation_factory(inplace=True),
            normalization_factory(out_channels // bottleneck),
            nn.Conv3d(
                out_channels // bottleneck,
                out_channels // bottleneck,
                groups=cardinality,
                kernel_size=temporal_kernel,
                dilation=dilation,
                padding=temporal_padding,
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
        bottleneck: int = 2,
        cardinality: int = 32,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        self.kernel_size = kernel_size
        self.activation_factory = activation_factory
        self.normalization_factory = normalization_factory
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
        )
