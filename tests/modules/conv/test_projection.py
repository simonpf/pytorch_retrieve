from pytorch_retrieve.modules.conv.projection import get_projection


import torch


def test_projection():

    # Trivial case. No downsampling
    proj = get_projection(16, 16, (1, 1))
    x = torch.rand(1, 16, 32, 32)
    y = proj(x)
    assert y.shape == (1, 16, 32, 32)

    # Adapting channels
    proj = get_projection(16, 32, (1, 1))
    y = proj(x)
    assert y.shape == (1, 32, 32, 32)

    # Adapting channels and downscaling
    proj = get_projection(16, 32, (2, 2))
    y = proj(x)
    assert y.shape == (1, 32, 16, 16)

    # Adapting channels and downscaling with anti-aliasing
    proj = get_projection(16, 32, (2, 2), anti_aliasing=True)
    y = proj(x)
    assert y.shape == (1, 32, 16, 16)
