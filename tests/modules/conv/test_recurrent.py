import torch

from pytorch_retrieve.modules.conv.utils import Scale
from pytorch_retrieve.modules.conv.encoders import Encoder
from pytorch_retrieve.modules.conv.decoders import Decoder
from pytorch_retrieve.modules.conv.blocks import BasicConv
from pytorch_retrieve.modules.conv.recurrent import Assimilator, GRU

def test_assimilation_block():
    """
    Ensure that assimilation block factory produces modules that produce
    the expected output lists.
    """
    block_factory = BasicConv()
    rec_factory = Assimilator(block_factory)

    rec_block = rec_factory(8, 16, downsample=2)

    x = [torch.rand(2, 8, 32, 32) for _ in range(8)]
    y = rec_block(x)

    assert len(y) == 8
    assert y[0].shape == (2, 16, 16, 16)

def test_assimilation_encoder_decoder():
    """
    Create an encoder and ensure that
      - Forwarding a tensor through the encoder works and the output tensor
        has the expected size.
    """
    block_factory = BasicConv()
    rec_factory = Assimilator(block_factory)
    stage_depths = [4, 4, 4, 4]
    channels = [8, 16, 32, 64]

    encoder = Encoder(
        channels=channels,
        stage_depths=stage_depths,
        downsampling_factors=[2, 2, 2],
        skip_connections=True,
        block_factory=rec_factory
    )
    x = [torch.rand(1, 8, 64, 64) for _ in range(4)]
    y = encoder(x)

    assert len(y) == 4
    assert y[Scale(8)][0].shape == (1, 64, 8, 8)

    decoder = Decoder(
        channels[::-1],
        stage_depths[::-1][1:],
        [2, 2, 2],
        block_factory=rec_factory,
        skip_connections=encoder.skip_connections
    )

    y = decoder(y)
    assert len(y) == 4
    assert y[0].shape == (1, 8, 64, 64)

def test_lstm_block():
    """
    Ensure that assimilation block factory produces modules that produce
    the expected output lists.
    """
    block_factory = BasicConv()
    rec_factory = GRU(block_factory)

    rec_block = rec_factory(8, 16, downsample=2)

    x = [torch.rand(2, 8, 32, 32) for _ in range(8)]
    y = rec_block(x)

    assert len(y) == 8
    assert y[0].shape == (2, 16, 16, 16)

def test_gru_block():
    """
    Ensure that assimilation block factory produces modules that produce
    the expected output lists.
    """
    block_factory = BasicConv()
    rec_factory = GRU(block_factory)

    rec_block = rec_factory(8, 16, downsample=2)

    x = [torch.rand(2, 8, 32, 32) for _ in range(8)]
    y = rec_block(x)

    assert len(y) == 8
    assert y[0].shape == (2, 16, 16, 16)
