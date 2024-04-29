"""
Tests for the pytorch_retrieve.modules.conv.encoders module.
"""
import torch

from pytorch_retrieve.modules.conv.utils import Scale
from pytorch_retrieve.modules.conv import blocks
from pytorch_retrieve.modules.conv.encoders import (
    Encoder,
    MultiInputSharedEncoder,
    MultiInputParallelEncoder,
    CascadingEncoder,
)


def test_encoder():
    """
    Create an encoder and ensure that
      - Forwarding a tensor through the encoder works and the output tensor
        has the expected size.
    """

    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = Encoder(
        channels=channels,
        stage_depths=stage_depths,
        downsampling_factors=[2, 2, 2],
        skip_connections=False
    )
    x = torch.rand(1, 8, 64, 64)
    y = encoder(x)
    assert y.shape == (1, 64, 8, 8)

    encoder = Encoder(
        channels=channels,
        stage_depths=stage_depths,
        downsampling_factors=[2, 2, 2],
        skip_connections=True
    )
    x = torch.rand(1, 8, 64, 64)
    y = encoder(x)
    assert isinstance(y, dict)
    assert y[Scale(8)].shape == (1, 64, 8, 8)


def test_multi_input_shared_encoder():
    """
    Create a multi-input encoder and ensure that
      - Forwarding a tensors through the encoder works and the output tensor
        has the expected size.
    """
    inputs = {"x_1": 1, "x_2": 2}
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = MultiInputSharedEncoder(
        inputs, channels, stage_depths, [2, 2, 2], skip_connections=False
    )
    inpt = {
        "x_1": torch.rand(1, 8, 64, 64),
        "x_2": torch.rand(1, 16, 32, 32),
    }
    y = encoder(inpt)
    assert y.shape == (1, 64, 8, 8)

    encoder = MultiInputSharedEncoder(
        inputs, channels, stage_depths, [2, 2, 2], skip_connections=True
    )
    y = encoder(inpt)
    assert isinstance(y, dict)
    assert y[Scale(8)].shape == (1, 64, 8, 8)


def test_multi_input_parallel_encoder():
    """
    Create a multi-input parallel encoder and ensure that
      - Forwarding a tensors through the encoder works and the output tensor
        has the expected size.
    """
    inputs = {"x_1": 1, "x_2": 2}
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = MultiInputParallelEncoder(
        inputs, channels, stage_depths, [2, 2, 2], skip_connections=False
    )

    inpt = {
        "x_1": torch.rand(1, 8, 64, 64),
        "x_2": torch.rand(1, 16, 32, 32),
    }
    y = encoder(inpt)
    assert y.shape == (1, 64, 8, 8)


def test_cascading_encoder():
    """
    Create a multi-input parallel encoder and ensure that
      - Forwarding a tensors through the encoder works and the output tensor
        has the expected size.
    """
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = CascadingEncoder(channels, stage_depths, downsampling_factors=[2, 2, 2])

    x = torch.rand(1, 8, 64, 64)
    y = encoder(x)

    assert y[Scale(8)].shape == (1, 64, 8, 8)


def test_encoder_multiple_block_factories():
    """
    Create an encoder with different block factories in different stages.
    """
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    block_factories = [blocks.BasicConv(), blocks.BasicConv(), blocks.ResNet(), blocks.ResNet()]

    encoder = Encoder(
        channels=channels,
        stage_depths=stage_depths,
        downsampling_factors=[2, 2, 2],
        block_factory=block_factories,
        skip_connections=False
    )

    assert isinstance(encoder.stages[0][0], blocks.BasicConvBlock)
    assert isinstance(encoder.stages[2][0], blocks.ResNetBlock)

    x = torch.rand(1, 8, 64, 64)
    y = encoder(x)
    assert y.shape == (1, 64, 8, 8)

    encoder = Encoder(
        channels=channels,
        stage_depths=stage_depths,
        downsampling_factors=[2, 2, 2],
        skip_connections=True
    )
    x = torch.rand(1, 8, 64, 64)
    y = encoder(x)
    assert isinstance(y, dict)
    assert y[Scale(8)].shape == (1, 64, 8, 8)


def test_heterogeneous_downsampling():
    """
    Create a multi-input parallel encoder and ensure that
      - Forwarding a tensors through the encoder works and the output tensor
        has the expected size.
    """
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = Encoder(
        channels,
        stage_depths,
        downsampling_factors=[(2, 1), (1, 2), (2, 2)]
    )

    x = torch.rand(1, 8, 64, 64)
    y = encoder(x)

    assert y[Scale(4)].shape == (1, 64, 16, 16)


def test_multiple_inputs_first_stage():
    """
    Test an encoder with multiple inputs at base scale and ensure that
    first stage isn't skipped.
    """
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = MultiInputSharedEncoder(
        inputs={
            "x_1": Scale((1, 1, 1)),
            "x_2": Scale((1, 1, 1)),
        },
        channels=channels,
        stage_depths=stage_depths,
        downsampling_factors=[(2, 1), (1, 2), (2, 2)],
        input_channels = {
            "x_1": 7,
            "x_2": 9,
        }
    )

    x = {
        "x_1": torch.rand(1, 7, 64, 64),
        "x_2": torch.rand(1, 9, 64, 64),
    }
    y = encoder(x)

    y[Scale(4)].mean().backward()

    for param in encoder.parameters():
        assert param.grad is not None
