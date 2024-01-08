"""
Tests for pytorch_retrieve.modules.conv.gru module.
"""
import torch

from pytorch_retrieve.modules.conv.gru import GRUCell, GRUNetwork


def test_gru_cell():
    """
    Instantiate GRU cell and ensure that:
       - Calculation of hidden state works.
       - Propagation of a sequence of inputs works.
    """
    gru_cell = GRUCell(16, 32, 3)

    x = [torch.rand((1, 16, 32, 32)) for _ in range(3)]
    y = gru_cell(x)
    assert len(y) == 3
    assert y[0].shape == (1, 32, 32, 32)

    x = torch.rand(3, 1, 16, 32, 32)
    y = gru_cell(x, hidden=y[0])
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 3
    assert y[0].shape == (1, 32, 32, 32)


def test_gru_network():
    """
    Instantiate GRU network and ensure that:
       - Calculation of hidden state works.
       - Propagation of a sequence of inputs works.
    """
    gru_cell = GRUNetwork(16, 32, 3, 3)

    x = [torch.rand((1, 16, 32, 32)) for _ in range(3)]
    y = gru_cell(x)
    assert len(y) == 3
    assert y[0].shape == (1, 32, 32, 32)

    x = torch.rand(3, 1, 16, 32, 32)
    y = gru_cell(x, hidden=y[0])
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 3
    assert y[0].shape == (1, 32, 32, 32)
