"""
Tests for the pytorch_retrieve.modules.conv.utils module.
"""
from pytorch_retrieve.modules.conv.utils import Scale


def test_scale_arithmetic():
    """
    Test arithmetic operations on scales.
    """
    scale = Scale(1)
    scale *= 2
    scale_4 = scale * (1, 1, 2)
    assert scale_4.scale == (1, 2, 4)

    scale_2 = scale_4 // 2
    scale_2 //= (1, 1, 2)

    assert scale_2 == Scale((1, 1, 1))


def test_min_max():
    """
    Test min and max operators on lists of scales.
    """
    scale_1 = Scale(1)
    scale_2 = Scale((1, 2, 2))
    scale_3 = Scale((1, 4, 2))
    scales = [scale_1, scale_2, scale_3]

    assert min(scales) == Scale(1)
    assert max(scales) == Scale((1, 4, 2))


def test_string_repr():
    """
    Ensure that string representations are consistent.
    """
    scale_1 = Scale(1)
    scale_2 = Scale((1, 1))
    scale_3 = Scale((1, 1, 1))

    assert str(scale_1) == str(scale_2)
    assert str(scale_2) == str(scale_3)


def test_hash():
    """
    Ensure that hash values of scales are consistent.
    """
    scale_1 = Scale(1)
    scale_2 = Scale((1, 1))
    scale_3 = Scale((1, 1, 1))

    dct = {}
    dct[scale_1] = "a"
    dct[scale_2] = "a"
    dct[scale_3] = "a"
    assert len(dct) == 1
