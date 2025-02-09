"""
Tests for the pytorch_retrieve.tiling module.
=============================================
"""
import numpy as np
import torch

from pytorch_retrieve.tiling import Tiler


def test_tiler():
    """
    Ensure that tiling and reassembling a tensor conserves its content.
    """
    for _ in range(100):
        height = np.random.randint(64, 1024)
        width = np.random.randint(64, 1024)
        x = np.arange(width).astype(np.float32)
        y = np.arange(height).astype(np.float32)
        xy = np.stack(np.meshgrid(x, y))

        tiler = Tiler(xy, tile_size=128, overlap=32)
        tiles = []
        for row_ind in range(tiler.M):
            row = []
            for col_ind in range(tiler.N):
                tile = tiler.get_tile(row_ind, col_ind)
                assert tile.shape[1:] == (128, 128)
                row.append(tile)
            tiles.append(row)

        xy_tiled = tiler.assemble(tiles)
        assert np.all(np.isclose(xy, xy_tiled))


def test_tiler_torch():
    """
    Test tiling of torch tensors.
    """
    for _ in range(100):
        height = np.random.randint(64, 1024)
        width = np.random.randint(64, 1024)
        x = np.arange(width).astype(np.float32)
        y = np.arange(height).astype(np.float32)
        xy = torch.tensor(np.stack(np.meshgrid(x, y)))

        tiler = Tiler(xy, tile_size=128, overlap=32)
        tiles = []
        for row_ind in range(tiler.M):
            row = []
            for col_ind in range(tiler.N):
                tile = tiler.get_tile(row_ind, col_ind)
                assert tile.shape[1:] == (128, 128)
                row.append(tile)
            tiles.append(row)

        xy_tiled = tiler.assemble(tiles)
        assert torch.isclose(xy, xy_tiled).all()


def test_tiler_multi_scale():
    """
    Ensure that tiling and reassembling a tensor conserves its content even for multi-scale tensors.
    """
    for _ in range(100):
        height = 8 * np.random.randint(8, 128)
        width = 8 * np.random.randint(8, 128)
        x = np.arange(width).astype(np.float32)
        y = np.arange(height).astype(np.float32)
        xy = np.stack(np.meshgrid(x, y))

        inpt = {
            "x_1": xy,
            "x_4": xy[..., ::4, ::4]
        }

        tiler = Tiler(inpt, tile_size=128, overlap=32)
        tiles = []
        for row_ind in range(tiler.M):
            row = []
            for col_ind in range(tiler.N):
                tile = tiler.get_tile(row_ind, col_ind)
                assert tile["x_1"].shape[1:] == (128, 128)
                assert tile["x_4"].shape[1:] == (32, 32)
                row.append(tile)
            tiles.append(row)

        inpt_tiled = tiler.assemble(tiles)
        assert np.all(np.isclose(inpt_tiled["x_1"], inpt["x_1"]))
        assert np.all(np.isclose(inpt_tiled["x_4"], inpt["x_4"]))


def test_tiler_multi_scale_torch():
    """
    Test multi-scale tiling with torch.Tensors.
    """
    for _ in range(100):
        height = 8 * np.random.randint(8, 128)
        width = 8 * np.random.randint(8, 128)
        x = np.arange(width).astype(np.float32)
        y = np.arange(height).astype(np.float32)
        xy = torch.Tensor(np.stack(np.meshgrid(x, y)))

        inpt = {
            "x_1": xy,
            "x_4": xy[..., ::4, ::4]
        }

        tiler = Tiler(inpt, tile_size=128, overlap=32)
        tiles = []
        for row_ind in range(tiler.M):
            row = []
            for col_ind in range(tiler.N):
                tile = tiler.get_tile(row_ind, col_ind)
                assert tile["x_1"].shape[1:] == (128, 128)
                assert tile["x_4"].shape[1:] == (32, 32)
                row.append(tile)
            tiles.append(row)

        inpt_tiled = tiler.assemble(tiles)
        assert torch.isclose(inpt_tiled["x_1"], inpt["x_1"]).all()
        assert torch.isclose(inpt_tiled["x_4"], inpt["x_4"]).all()


def test_tiler_wrap_columns():
    """
    Test tiling with column-wrap.
    """
    for _ in range(100):
        height = np.random.randint(64, 1024)
        width = np.random.randint(64, 1024)
        x = np.arange(width).astype(np.float32)
        y = np.arange(height).astype(np.float32)
        xy = np.stack(np.meshgrid(x, y))

        tiler = Tiler(xy, tile_size=128, overlap=32, wrap_columns=True)
        tiles = []
        for row_ind in range(tiler.M):
            row = []
            for col_ind in range(tiler.N):
                tile = tiler.get_tile(row_ind, col_ind)
                assert tile.shape[1:] == (128, 128)
                row.append(tile)
            tiles.append(row)

        xy_tiled = tiler.assemble(tiles)
        assert np.all(np.isclose(xy, xy_tiled))


def test_tiler_wrap_columns_torch():
    """
    Test tiling with column-wrap and torch.Tensors.
    """
    for _ in range(100):
        height = np.random.randint(64, 1024)
        width = np.random.randint(64, 1024)
        x = np.arange(width).astype(np.float32)
        y = np.arange(height).astype(np.float32)
        xy = torch.Tensor(np.stack(np.meshgrid(x, y)))

        tiler = Tiler(xy, tile_size=128, overlap=32, wrap_columns=True)
        tiles = []
        for row_ind in range(tiler.M):
            row = []
            for col_ind in range(tiler.N):
                tile = tiler.get_tile(row_ind, col_ind)
                assert tile.shape[1:] == (128, 128)
                row.append(tile)
            tiles.append(row)

        xy_tiled = tiler.assemble(tiles)
        assert torch.isclose(xy, xy_tiled).all()
