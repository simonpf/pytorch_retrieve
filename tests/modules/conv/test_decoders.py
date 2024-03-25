"""
Tests for the pytorch_retrieve.modules.conv.decoders module.
"""
import torch


from pytorch_retrieve.modules.conv import blocks
from pytorch_retrieve.modules.conv.encoders import Encoder
from pytorch_retrieve.modules.conv.decoders import Decoder


def test_decoder():
    """
    Create an encoder and corresponding decoder and ensure that
      - Forwarding inputs through both encoder and decoder works.
    """
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = Encoder(
        channels=channels,
        stage_depths=stage_depths,
        downsampling_factors=[2, 2, 2]
    )
    decoder = Decoder(channels[::-1], stage_depths[::-1][1:], [2, 2, 2])

    x = torch.rand(1, 8, 64, 64)
    y = decoder(encoder(x))
    assert y.shape == (1, 8, 64, 64)


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
