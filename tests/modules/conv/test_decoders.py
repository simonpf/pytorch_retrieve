"""
Tests for the pytorch_retrieve.modules.conv.decoders module.
"""
import torch

from pytorch_retrieve.modules.conv.encoders import Encoder
from pytorch_retrieve.modules.conv.decoders import Decoder


def test_encoder():
    """
    Create an encoder and corresponding decoder and ensure that
      - Forwarding inputs through both encoder and decoder works.
    """
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = Encoder(channels, stage_depths, [2, 2, 2])
    decoder = Decoder(channels[::-1], stage_depths[::-1][1:], [2, 2, 2])

    x = torch.rand(1, 8, 64, 64)
    y = decoder(encoder(x))
    assert y.shape == (1, 8, 64, 64)
