"""
Tests for the pytorch_retrieve.tensors.masked_tensor module.
"""
import numpy as np
import pytest
import torch
from torch import nn

from pytorch_retrieve.tensors.masked_tensor import MaskedTensor
from pytorch_retrieve.modules.transformations import SquareRoot


def test_cat():
    """
    Test concatenation of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)

    masked_tensor_2 = torch.cat([masked_tensor, masked_tensor], 1)
    assert masked_tensor_2.shape[1] == 20

    masked_tensor_2 = torch.cat([masked_tensor, tensor], 1)
    assert masked_tensor_2.shape[1] == 20

    masked_tensor_2 = torch.cat([tensor, masked_tensor, tensor], 1)
    assert masked_tensor_2.shape[1] == 30

    torch.cat([tensor, masked_tensor], 1, out=masked_tensor_2)
    assert masked_tensor_2.shape[1] == 20


def test_stack():
    """
    Test stacking of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)

    masked_tensor_2 = torch.stack([masked_tensor, masked_tensor], 1)
    assert masked_tensor_2.shape[1] == 2

    masked_tensor_2 = torch.stack([masked_tensor, tensor], 1)
    assert masked_tensor_2.shape[1] == 2

    masked_tensor_2 = torch.stack([tensor, masked_tensor, tensor], 1)
    assert masked_tensor_2.shape[1] == 3

    torch.stack([tensor, masked_tensor], 1, out=masked_tensor_2)
    assert masked_tensor_2.shape[1] == 2


def test_add():
    """
    Test addition of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask_1 = torch.rand(10, 10, 10) - 0.5 > 0
    mask_2 = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = MaskedTensor(tensor, mask=mask_2)

    masked_tensor_2 = masked_tensor_1 + masked_tensor_2
    assert (masked_tensor_2.mask == (torch.logical_or(mask_1, mask_2))).all()
    assert torch.isclose(masked_tensor_2, 2.0 * tensor).all()

    tensor = torch.rand(10, 10, 10)
    mask_1 = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)

    masked_tensor_3 = tensor + masked_tensor_1
    assert isinstance(masked_tensor_3, MaskedTensor)

    masked_tensor_4 = (torch.tensor(1.0) - masked_tensor_1) / torch.tensor(0.5)
    assert isinstance(masked_tensor_4, MaskedTensor)


def test_sub():
    """
    Test difference of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask_1 = torch.rand(10, 10, 10) - 0.5 > 0
    mask_2 = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = MaskedTensor(tensor, mask=mask_2)

    masked_tensor_2 = torch.sub(masked_tensor_1, masked_tensor_2)
    assert (masked_tensor_2.mask == (torch.logical_or(mask_1, mask_2))).all()
    assert torch.isclose(masked_tensor_2, torch.zeros_like(masked_tensor_2)).all()

    masked_tensor_2 = masked_tensor_1 - masked_tensor_2.base
    assert (masked_tensor_2.mask == masked_tensor_1.mask).all()


def test_mul():
    """
    Test multiplication of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask_1 = torch.rand(10, 10, 10) - 0.5 > 0
    mask_2 = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = MaskedTensor(tensor, mask=mask_2)

    masked_tensor_2 = masked_tensor_1 * masked_tensor_2
    assert (masked_tensor_2.mask == (torch.logical_or(mask_1, mask_2))).all()
    assert masked_tensor_2.mask.shape == masked_tensor_1.shape
    assert torch.isclose(masked_tensor_2, tensor**2).all()

    masked_tensor_2 = tensor * masked_tensor_2
    assert masked_tensor_2.mask.shape == masked_tensor_1.shape


def test_pow():
    """
    Test exponentiation of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask_1 = torch.rand(10, 10, 10) - 0.5 > 0
    mask_2 = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)

    masked_tensor_2 = torch.pow(masked_tensor_1, 2)
    assert torch.isclose(masked_tensor_2, tensor**2).all()

    # masked_tensor_2 = torch._C._TensorBase.pow(masked_tensor_2, 2)
    masked_tensor_2 = masked_tensor_2**2
    assert torch.isclose(masked_tensor_2, tensor**4).all()


def test_permute():
    """
    Test permutation of masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = torch.permute(masked_tensor_1, (2, 1, 0))
    assert masked_tensor_2.shape == (3, 2, 1)
    assert masked_tensor_2.mask.shape == (3, 2, 1)


def test_reshape():
    """
    Test reshaping of masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = torch.reshape(masked_tensor_1, (3, 2, 1))
    assert masked_tensor_2.shape == (3, 2, 1)
    assert masked_tensor_2.mask.shape == (3, 2, 1)


def test_view():
    """
    Test view applied to masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = masked_tensor_1.view((3, 2, 1))
    assert masked_tensor_2.shape == (3, 2, 1)
    assert masked_tensor_2.mask.shape == (3, 2, 1)


def test_squeeze():
    """
    Test squeezing of masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = masked_tensor_1.squeeze()
    assert masked_tensor_2.shape == (2, 3)

    masked_tensor_2 = torch.squeeze(masked_tensor_1)
    assert masked_tensor_2.shape == (2, 3)


def test_unsqueeze():
    """
    Test unsqueezing of masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = masked_tensor_1.unsqueeze(0)
    assert masked_tensor_2.shape == (1, 1, 2, 3)

    masked_tensor_2 = torch.unsqueeze(masked_tensor_1, 0)
    assert masked_tensor_2.shape == (1, 1, 2, 3)


def test_sum():
    """
    Test summing of tensor elements.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)

    masked_sum = masked_tensor_1.sum()
    sum_ref = tensor[~mask_1].sum()
    assert torch.isclose(sum_ref, masked_sum)

    masked_sum = torch.sum(masked_tensor_1)
    sum_ref = torch.sum(tensor[~mask_1])
    assert torch.isclose(sum_ref, masked_sum)


def test_mean():
    """
    Test calculating the mean of a masked tensor.
    """
    tensor = torch.rand(16, 16, 3)
    mask_1 = torch.rand(16, 16, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)

    masked_mean = masked_tensor_1.mean()
    mean_ref = tensor[~mask_1].mean()
    assert torch.isclose(mean_ref, masked_mean)


def test_abs():
    """
    Test calculating the abs of a masked tensor.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)

    masked_tensor_2 = torch.abs(masked_tensor_1)
    assert type(masked_tensor_2) == MaskedTensor


def test_gt():
    """
    Test calculating the mean of a masked tensor.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)

    masked_tensor_1 > 1


def test_maximum():
    """
    Test calculating maximum of two masked tensors.
    """
    tensor_1 = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    tensor_1[mask_1] = np.nan
    masked_tensor_1 = MaskedTensor(tensor_1, mask=mask_1)

    tensor_2 = torch.rand(1, 2, 3)
    mask_2 = torch.rand(1, 2, 3) - 0.5 > 0
    tensor_2[mask_2] = np.nan
    masked_tensor_2 = MaskedTensor(tensor_2, mask=mask_2)

    masked_tensor_3 = torch.maximum(masked_tensor_1, masked_tensor_2)

    mask_3 = masked_tensor_3.mask
    assert (mask_3 == (mask_1 & mask_2)).all()


def test_max():
    """
    Test calculating the maximum of a single tensor.
    """
    tensor_1 = torch.rand(100)
    mask_1 = torch.rand(100) - 0.5 > 0
    tensor_1[mask_1] = torch.nan
    masked_tensor_1 = MaskedTensor(tensor_1, mask=mask_1)

    masked_max = masked_tensor_1.max()
    assert torch.isfinite(masked_max)

    masked_max = masked_tensor_1.max(dim=0)
    ref_max = tensor_1[~mask_1].max(dim=0)

    assert masked_max[0] == ref_max[0]


def test_min():
    """
    Test calculating the minimum of a single tensors.
    """
    tensor_1 = torch.rand(100)
    mask_1 = torch.rand(100) - 0.5 > 0
    tensor_1[mask_1] = torch.nan
    masked_tensor_1 = MaskedTensor(tensor_1, mask=mask_1)

    masked_min = masked_tensor_1.min()
    assert torch.isfinite(masked_min)

    masked_min = masked_tensor_1.min(dim=0)
    ref_min = tensor_1[~mask_1].min(dim=0)

    assert masked_min[0] == ref_min[0]


def test_minimum():
    """
    Test calculating minimum of two masked tensors.
    """
    tensor_1 = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    tensor_1[mask_1] = np.nan
    masked_tensor_1 = MaskedTensor(tensor_1, mask=mask_1)

    tensor_2 = torch.rand(1, 2, 3)
    mask_2 = torch.rand(1, 2, 3) - 0.5 > 0
    tensor_2[mask_2] = np.nan
    masked_tensor_2 = MaskedTensor(tensor_2, mask=mask_2)

    masked_tensor_3 = torch.minimum(masked_tensor_1, masked_tensor_2)

    mask_3 = masked_tensor_3.mask
    assert (mask_3 == (mask_1 & mask_2)).all()


def test_select():
    """
    Test selecting sub-tensors.
    """
    tensor_1 = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    tensor_1[mask_1] = np.nan
    masked_tensor_1 = MaskedTensor(tensor_1, mask=mask_1)

    masked_tensor_2 = torch.select(masked_tensor_1, 1, 0)

    assert (masked_tensor_1[:, 0, :].eq(masked_tensor_2)).all()


def test_any_all():
    """
    Test any and all operations.
    """
    tensor_1 = torch.rand(4, 4, 4)
    tensor_1 = tensor_1 > 0.5
    masked_tensor_1 = MaskedTensor(tensor_1, mask=tensor_1)

    assert (masked_tensor_1 <= 0.5).all()
    assert not (masked_tensor_1 > 0.5).any()

    assert torch.all(masked_tensor_1 <= 0.5)
    assert not torch.any(masked_tensor_1 > 0.5)


def test_transpose():
    """
    Test transpose of tensor.
    """
    tensor_1 = torch.rand(1, 2, 3)
    tensor_1 = tensor_1 > 0.5
    masked_tensor_1 = MaskedTensor(tensor_1, mask=tensor_1)

    masked_tensor_2 = torch.transpose(masked_tensor_1, 0, 1)

    assert masked_tensor_2.shape == (2, 1, 3)
    assert masked_tensor_2.mask.shape == (2, 1, 3)


def test_tensor_ops():
    """
    Test basic operations with tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)
    assert hasattr(masked_tensor, "mask")

    masked_tensor_2 = MaskedTensor(masked_tensor)
    assert hasattr(masked_tensor_2, "mask")
    assert (masked_tensor.mask == masked_tensor_2.mask).all()

    tensor_2 = torch.rand(10, 10, 10)
    mask_2 = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor_3 = MaskedTensor(tensor_2, mask=mask_2)

    masked_tensor_4 = masked_tensor_2 + masked_tensor_3
    mask = masked_tensor_4.mask
    assert torch.isfinite(masked_tensor_4[~mask]).all()
    assert (masked_tensor_4[~mask] == (tensor + tensor_2)[~mask]).all()

    masked_tensor_5 = masked_tensor_2 * masked_tensor_3
    mask = masked_tensor_5.mask
    assert (masked_tensor_5[~mask] == (tensor * tensor_2)[~mask]).all()

    masked_tensor_5 = masked_tensor[:5, :5]
    assert masked_tensor_5.mask.shape == (5, 5, 10)

    masked_tensor_6 = torch.permute(masked_tensor_5, (2, 0, 1))
    assert masked_tensor_6.shape == (10, 5, 5)
    assert masked_tensor_6.mask.shape == (10, 5, 5)

    masked_tensor_7 = masked_tensor_5.reshape((50, 5))
    assert masked_tensor_7.shape == (50, 5)
    assert masked_tensor_7.mask.shape == (50, 5)

    masked_tensor_7.add_(1.0)

    masked_tensor_8 = torch.stack([masked_tensor_7, masked_tensor_7])
    assert masked_tensor_8.shape == (2, 50, 5)
    assert masked_tensor_8.mask.shape == (2, 50, 5)


def test_type_as():
    """
    Test type conversion of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)
    other = torch.rand(10, 10, 10).to(dtype=torch.int64)
    masked_tensor = masked_tensor.type_as(other)
    assert masked_tensor.dtype == torch.int64


def test_setitem():
    """
    Test type conversion of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)
    masked_tensor[0] = 1.0
    assert torch.all(torch.isclose(masked_tensor[0], torch.tensor(1.0)))


def test_relu():
    """
    Test ReLU function.
    """
    tensor = torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    tensor[mask] = torch.nan
    masked_tensor = nn.functional.relu(MaskedTensor(tensor, mask=mask))

    assert not (masked_tensor.base >= 0.0).all()
    assert (masked_tensor >= 0.0).all()


def test_binary_cross_entropy():
    """
    Ensure that masked values are ignored in calculation of masked binary
    cross entropy.
    """
    tensor = 100 * torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    tensor[mask] = -10.0
    masked_tensor = MaskedTensor(tensor, mask=mask)

    target = torch.ones_like(tensor)

    loss = nn.functional.binary_cross_entropy_with_logits(
        tensor[~mask], target[~mask]
    )
    loss_masked = nn.functional.binary_cross_entropy_with_logits(
        masked_tensor, target
    )

    assert torch.isclose(loss, loss_masked)


def test_cross_entropy():
    """
    Ensure that masked values are ignored in calculation of masked
    cross entropy.
    """
    tensor = 100 * torch.rand(100, 3, 100)
    mask = torch.rand(100, 3, 100) - 0.5 > 0
    tensor[mask] = -10.0
    masked_tensor = MaskedTensor(tensor, mask=mask)

    target = 2 * torch.ones_like(tensor[:, 0], dtype=torch.int64)

    target_mask = mask.any(1)

    loss = nn.functional.cross_entropy(
        tensor.transpose(1, -1)[~target_mask],
        target[~target_mask]
    )
    loss_masked = nn.functional.cross_entropy(
        masked_tensor, target
    )

    assert torch.isclose(loss, loss_masked)


def test_clone_tensor():
    """
    Ensure that masks are cloned when masked tensors are cloned.
    """
    tensor = 100 * torch.rand(100, 3, 100)
    mask = torch.rand(100, 3, 100) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)
    assert not masked_tensor.mask.all()

    masked_tensor2 = masked_tensor.clone()
    masked_tensor2.mask[:] = True
    assert not masked_tensor.mask.all()


def test_to():
    """
    Test .to method.
    """
    tensor = 100 * torch.rand(100, 3, 100)
    mask = torch.rand(100, 3, 100) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)

    assert not masked_tensor.dtype == torch.bfloat16

    masked_tensor = masked_tensor.to(device="cpu", dtype=torch.bfloat16)
    assert masked_tensor.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs GPU.")
def test_cpu():
    """
    Test .cpu method.
    """
    tensor = 100 * torch.rand(100, 3, 100)
    mask = torch.rand(100, 3, 100) - 0.5 > 0
    tensor = MaskedTensor(tensor, mask=mask)

    tensor = tensor.to("cuda:0")
    assert tensor.device == torch.device('cuda:0')
    assert tensor.mask.device == torch.device('cuda:0')

    tensor = tensor.cpu()
    assert tensor.device == torch.device('cpu')
    assert tensor.mask.device == torch.device('cpu')


def test_transformation():
    """
    Ensur that transformations are passed on when new tensors are created.
    """
    mask = torch.rand(4, 32, 64) - 0.5 > 0
    tnsr = MaskedTensor(torch.rand(4, 32, 64), mask=mask, transformation=SquareRoot)
    assert tnsr.__transformation__ is not None

    tnsr_new = tnsr.reshape(4, 32, 8, 8)
    assert tnsr_new.__transformation__ is not None

    tnsr_new = tnsr_new + tnsr_new
    assert tnsr_new.__transformation__ is not None
