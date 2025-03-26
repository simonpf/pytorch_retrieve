"""
Tests for the pytorch_retrieve.modules.encodings module.
"""

import torch

from pytorch_retrieve.modules.encodings import FourierEncoding


def test_fourier_encoding():
    """
    Test fourier encoding for tabular and spatial inputs.
    """
    enc = FourierEncoding(in_channels=1, out_channels=16, dim=-1)

    x = torch.linspace(-1, 1, 11)[..., None]
    x_enc = enc(x)

    assert torch.isclose(x_enc[0, ::4], torch.tensor(-1.0)).all()
    assert torch.isclose(x_enc[0, 2::4], torch.tensor(1.0)).all()
    assert torch.isclose(x_enc[0, 1::2], torch.tensor(0.0), atol=1e-3).all()
    assert torch.isclose(x_enc[-1, ::4], torch.tensor(-1.0)).all()
    assert torch.isclose(x_enc[-1, 2::4], torch.tensor(1.0)).all()
    assert torch.isclose(x_enc[-1, 1::2], torch.tensor(0.0), atol=1e-3).all()

    enc = FourierEncoding(in_channels=1, out_channels=16, dim=-3)
    x = torch.linspace(-1, 1, 11)[None, None, None].repeat_interleave(11, 2).repeat_interleave(4, 0)
    x_enc = enc(x)

    assert torch.isclose(x_enc[:, ::4, 0, 0], torch.tensor(-1.0)).all()
    assert torch.isclose(x_enc[:, 2::4, 0, 0], torch.tensor(1.0)).all()
    assert torch.isclose(x_enc[:, 1::2, 0, 0], torch.tensor(0.0), atol=1e-3).all()
    assert torch.isclose(x_enc[:, ::4, -1, -1], torch.tensor(-1.0)).all()
    assert torch.isclose(x_enc[:, 2::4, -1, -1], torch.tensor(1.0)).all()
    assert torch.isclose(x_enc[:, 1::2, -1, -1], torch.tensor(0.0), atol=1e-3).all()
