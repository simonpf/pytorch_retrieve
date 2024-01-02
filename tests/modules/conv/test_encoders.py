"""
Tests for the pytorch_retrieve.modules.conv.encoders module.
"""
import torch

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

    encoder = Encoder(channels, stage_depths, [2, 2, 2], skip_connections=False)
    x = torch.rand(1, 8, 64, 64)
    y = encoder(x)
    assert y.shape == (1, 64, 8, 8)

    encoder = Encoder(channels, stage_depths, [2, 2, 2], skip_connections=True)
    x = torch.rand(1, 8, 64, 64)
    y = encoder(x)
    assert isinstance(y, dict)
    assert y[8].shape == (1, 64, 8, 8)


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
    assert y[8].shape == (1, 64, 8, 8)


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

    assert y[8].shape == (1, 64, 8, 8)
