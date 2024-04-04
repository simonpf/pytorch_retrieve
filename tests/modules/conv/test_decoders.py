"""
Tests for the pytorch_retrieve.modules.conv.decoders module.
"""
import torch


from pytorch_retrieve.modules.conv import blocks
from pytorch_retrieve.modules.conv.encoders import Encoder
from pytorch_retrieve.modules.conv.decoders import Decoder, MultiScalePropagator
from pytorch_retrieve.modules.conv.utils import Scale


def test_decoder():
    """
    Create an encoder and corresponding decoder and ensure that
      - Forwarding inputs through both encoder and decoder works.
      - Forwarding works with skip connections
      - Return of multi-scale outputs works
    """
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = Encoder(
        channels=channels,
        stage_depths=stage_depths,
        downsampling_factors=[2, 2, 2]
    )

    # Without skip connections
    decoder = Decoder(channels[::-1], stage_depths[::-1][1:], [2, 2, 2])
    x = torch.rand(1, 8, 64, 64)
    y = decoder(encoder(x))
    assert y.shape == (1, 8, 64, 64)
    x = torch.rand(1, 8, 64, 64)
    y = decoder(encoder(x)[Scale((1, 8, 8))])
    assert y.shape == (1, 8, 64, 64)

    # With skip connections
    decoder = Decoder(
        channels[::-1],
        stage_depths[::-1][1:],
        [2, 2, 2],
        skip_connections=encoder.skip_connections
    )
    x = torch.rand(1, 8, 64, 64)
    y = decoder(encoder(x))
    assert y.shape == (1, 8, 64, 64)

    # Multi-scale output
    decoder = Decoder(channels[::-1], stage_depths[::-1][1:], [2, 2, 2])
    y = decoder.forward_multi_scale_output(encoder(x))
    assert isinstance(y, dict)
    assert len(y) == len(decoder.stages)
    assert decoder.base_scale not in y
    assert y[encoder.base_scale].shape == (1, 8, 64, 64)

    y = decoder.forward_multi_scale_output(encoder(x)[Scale((1, 8, 8))])
    assert isinstance(y, dict)
    assert len(y) == len(decoder.stages)
    assert decoder.base_scale not in y
    assert y[encoder.base_scale].shape == (1, 8, 64, 64)

    outputs = decoder.multi_scale_outputs
    assert list(outputs.keys()) == [
        Scale((1, 4, 4)),
        Scale((1, 2, 2)),
        Scale((1, 1, 1)),
    ]
    assert list(outputs.values()) == [32, 16, 8]


def test_decoder_multiple_block_factories():
    """
    Create an encoder and corresponding decoder and ensure that
      - Forwarding inputs through both encoder and decoder works.
    """
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    block_factories = [blocks.BasicConv(), blocks.BasicConv(), blocks.ResNet(), blocks.ResNet()]

    encoder = Encoder(
        channels=channels,
        stage_depths=stage_depths,
        downsampling_factors=[2, 2, 2],
        block_factory=block_factories
    )

    assert isinstance(encoder.stages[0][0], blocks.BasicConvBlock)
    assert isinstance(encoder.stages[2][0], blocks.ResNetBlock)
    decoder = Decoder(channels[::-1], stage_depths[::-1][1:], [2, 2, 2], block_factory=block_factories[1:][::-1])

    assert isinstance(decoder.stages[0][0], blocks.ResNetBlock)
    assert isinstance(decoder.stages[2][0], blocks.BasicConvBlock)

    x = torch.rand(1, 8, 64, 64)
    y = decoder(encoder(x))
    assert y.shape == (1, 8, 64, 64)


def test_multi_scale_propagator():
    """
    Tests predictions with a MultiScalePropagator.
    """
    inputs = {
        Scale((4, 8, 8)): 64,
        Scale((2, 4, 4)): 32,
        Scale((1, 2, 2)): 16,
    }
    channels = [48, 24, 12]
    stage_depths = [4, 3, 2]
    block_factory = blocks.BasicConv()

    propagator = MultiScalePropagator(
        inputs,
        stage_depths,
        block_factory
    )

    x = {
        Scale((4, 8, 8)): torch.rand(2, 64, 2, 16, 16),
        Scale((2, 4, 4)): torch.rand(2, 32, 4, 32, 32),
        Scale((1, 2, 2)): torch.rand(2, 16, 8, 64, 64),
    }
    y = propagator(x, 8)
    assert len(y) == 8
    assert y[0].shape == (2, 16, 64, 64)

    propagator = MultiScalePropagator(
        inputs,
        stage_depths,
        block_factory,
        residual=False
    )
    y = propagator(x, 8)
    assert len(y) == 8
    assert y[0].shape == (2, 16, 64, 64)

    y = propagator(x, 3)
    assert len(y) == 3
    assert y[0].shape == (2, 16, 64, 64)
