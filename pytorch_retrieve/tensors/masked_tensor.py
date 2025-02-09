"""
pytorch_retrieve.tensor.masked_tensor
=====================================

Provides a masked tensor class that allows masking of invalid elements.
"""
from collections.abc import Sequence, Mapping
import functools

import numpy as np
import torch


HANDLED_FUNCTIONS = {}


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class MaskedTensor(torch.Tensor):
    """
    Extends the torch.Tensor class by adding a mask that identifies
    invalid elements. The masked tensor also provides functionality
    to compress the tensor along the batch axis to speed up
    calculations.
    """

    def __new__(cls, *args, transformation=None, **kwargs):
        mask = kwargs.pop("mask", None)
        tensor = super().__new__(cls, *args, **kwargs)

        if transformation is not None:
            tensor.__transformation__ = transformation
        if not hasattr(tensor, "__transformation__") and hasattr(args[0], "__transformation__"):
            new_tensor.__transformation__ = tensor.__transformation__

        # Keep reference to original tensor.
        if isinstance(args[0], MaskedTensor):
            tensor.base = args[0].base
        else:
            tensor.base = args[0]

        # Infer mask if not given.
        if mask is None:
            if isinstance(args[0], MaskedTensor):
                mask = args[0].mask
        if mask is None:
            tensor = args[0]
            mask = torch.zeros(tensor.shape, dtype=bool, device=tensor.device)
        tensor.mask = mask.detach().to(device=args[0].device)

        if isinstance(mask, MaskedTensor):
            mask = torch.tensor(mask)

        return tensor

    def strip(self):
        return self.base

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """ """
        if kwargs is None:
            kwargs = {}

        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

        if len(args) == 1 and len(kwargs) == 0:
            result = func(args[0].base)
            if isinstance(result, torch.Tensor):
                return MaskedTensor(
                    result,
                    mask=args[0].mask,
                    transformation=getattr(args[0], "__transforation__", None)
                )

        if len(args) > 0 and isinstance(args[0], MaskedTensor):
            masked_args = [arg for arg in args[1:] if isinstance(arg, MaskedTensor)]
            masked_args += [
                arg for arg in kwargs.values() if isinstance(arg, MaskedTensor)
            ]
            if len(masked_args) == 0:
                return func(args[0].base, *args[1:], **kwargs)

        return NotImplemented

    def compress(self):
        n_tot = self.shape[0]
        all_missing = self.mask.view((n_tot, -1)).all(dim=-1)
        valid = torch.nonzero(~all_missing)[:, 0]
        return MaskedTensor(self[valid], transformation=get_transformation(args))

    def __getitem__(self, *args, **kwargs):
        """
        Slices tensor and corresponding mask.
        """
        return MaskedTensor(
            self.strip().__getitem__(*args, **kwargs),
            mask=self.mask.__getitem__(*args, **kwargs),
            transformation=getattr(self, "__transformation__", None)
        )

    def __pow__(self, exp):
        return pow(self, exp)


def get_base(tensor: torch.Tensor) -> torch.Tensor:
    """
    Get raw data tensor from a tensor.

    This function extract the underlying data tensor 'base' from a
    MaskedTensor or simply returns the given tensor if it is not
    a MaskedTensor.

    Args:
         tensor: The tensor from which to extract the underlying data tensor.

    Return:
         A torch.Tensor object containing the raw data in 'tensor'.
    """
    if isinstance(tensor, MaskedTensor):
        return tensor.base
    return tensor


def get_mask(tensor):
    """
    Get mask from a tensor argument.

    Generic function to retrieve a mask identifying invalid elements in
    a standard tensor or a masked tensor.

    Args:
        tensor: If this is a MaskedTensor, its mask is returned. If it is
            a standard torch.Tensor a mask identifying NAN elements is
            returned.
    """
    if isinstance(tensor, MaskedTensor):
        return tensor.mask
    elif isinstance(tensor, torch.Tensor):
        return torch.zeros(tensor.shape, device=tensor.device, dtype=bool)
    torch.zeros(1, dtype=bool)


def get_transformation(tensor):
    """
    Get transformation from tensor arguments.

    Args:
        tensor: A tensor, a list of tensors, a dict of tensors or an arbitary Python object.

    Return:
        The transformation object from the first  object that has a __transformation__ attribute.
    """
    if isinstance(tensor, Sequence):
        for elem in tensor:
            result = get_transformation(elem)
            if result is not None:
                return result
    elif isinstance(tensor, Mapping):
        for elem in tensor.values():
            result = get_transformation(elem)
            if result is not None:
                return result
    else:
        if hasattr(tensor, "__transformation__"):
            return tensor.__transformation__
    return None


@implements(torch.cat)
def cat(tensors, dim=0, out=None):
    """
    Concatenate tensors and their masks.
    """
    if out is None:
        res = torch.cat([get_base(t) for t in tensors], dim=dim)
        mask = torch.cat([get_mask(t) for t in tensors], dim=dim)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(tensors))
    else:
        res = torch.cat([get_base(t) for t in tensors], dim=dim, out=get_base(out))
        mask = torch.cat([get_mask(t) for t in tensors], dim=dim, out=out.mask)
        return res


@implements(torch.stack)
def stack(tensors, dim=0, out=None):
    """
    Stack tensors and their masks.
    """
    if out is None:
        res = torch.stack([get_base(t) for t in tensors], dim=dim)
        mask = torch.stack([get_mask(t) for t in tensors], dim=dim)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(tensors))
    else:
        res = torch.stack([get_base(t) for t in tensors], dim=dim, out=get_base(out))
        mask = torch.stack([get_mask(t) for t in tensors], dim=dim)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(tensors))


###############################################################################
# Addition
###############################################################################


@implements(torch.add)
def add(inpt, other, alpha=1, out=None):
    """
    Concatenate tensors and their masks.
    """
    if out is None:
        res = torch.add(
            get_base(inpt),
            get_base(other),
            alpha=alpha,
        )
        mask = torch.logical_or(
            get_mask(inpt),
            get_mask(other),
        )
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.add(get_base(inpt), get_base(other), alpha=alpha, out=get_base(out))
        mask = torch.logical_or(get_mask(inpt), get_mask(other), out=out.mask)
        return res


@implements(torch.Tensor.add)
def tadd(inpt, other, alpha=1, out=None):
    """
    Concatenate tensors and their masks.
    """
    return add(inpt, other, alpha=alpha, out=out)


@implements(torch.Tensor.add_)
def iadd(inpt, other, alpha=1):
    """
    Concatenate tensors and their masks.
    """
    get_base(inpt).add_(get_base(other), alpha=alpha)
    if isinstance(inpt, MaskedTensor):
        inpt.mask = combine_masks(inpt, other)
    return inpt


@implements(torch.sub)
def sub(inpt, other, alpha=1, out=None):
    """
    Concatenate tensors and their masks.
    """
    if out is None:
        res = torch.sub(
            get_base(inpt),
            get_base(other),
            alpha=alpha,
        )
        mask = combine_masks(inpt, other, torch.logical_or, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.sub(get_base(inpt), get_base(other), alpha=alpha, out=get_base(out))
        mask = torch.logical_or(
            get_mask(inpt), get_mask(other), shape=res.shape, out=out.mask
        )
        return res


@implements(torch.Tensor.sub)
def tsub(inpt, other, alpha=1, out=None):
    """
    Concatenate tensors and their masks.
    """
    return sub(inpt, other, alpha=alpha, out=out)


@implements(torch.Tensor.sub_)
def isub(inpt, other, alpha=1):
    """
    Concatenate tensors and their masks.
    """
    get_base(inpt).sub_(other, alpha=alpha)
    if isinstance(inpt, MaskedTensor):
        inpt.mask = torch.logical_or(get_mask(inpt), get_mask(other))
    return inpt


###############################################################################
# Multiplication
###############################################################################


def combine_masks(tensor_1, tensor_2, op=torch.logical_or, out=None, shape=None):
    """
    Combine masks of arguments, one of which must be a masked tensor.

    Args:
        tensor_1: The first tensor argument.
        tensor_2: The second tensor argument.
        op: The operation to use to combine the masks.
        out: Optional tensor to write the output to.
        shape: Tuple specifying the expected shape of the result.

    Return:
        A mask tensor obtained by combining the masks from 'tensor_1' and
        'tensor_2'.
    """
    if isinstance(tensor_1, MaskedTensor):
        if isinstance(tensor_2, MaskedTensor):
            if out is not None:
                return op(tensor_1.mask, tensor_2.mask)
            else:
                return op(tensor_1.mask, tensor_2.mask, out=out)
        if shape is not None and tensor_1.mask != shape:
            return torch.broadcast_to(tensor_1.mask, shape)
        return tensor_1.mask
    if shape is not None and tensor_2.mask != shape:
        return torch.broadcast_to(tensor_2.mask, shape)
    return tensor_2.mask


@implements(torch.mul)
def mul(inpt, other, out=None):
    """
    Multiplicate tensors and combine their masks using 'and'.
    """
    if out is None:
        res = torch.mul(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.mul(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return res


@implements(torch.Tensor.mul)
def tmul(inpt, other, out=None):
    """
    Multiplication class method.
    """
    return mul(inpt, other, out=out)


@implements(torch.Tensor.mul_)
def imul(inpt, other):
    """
    Inplace multiplication.
    """
    get_base(inpt).mul_(other)
    if isinstance(inpt, MaskedTensor):
        inpt.mask = torch.logical_or(get_mask(inpt), get_mask(other))


@implements(torch.div)
def div(inpt, other, out=None):
    """
    Divide tensors and combine their masks using 'and'.
    """
    if out is None:
        res = torch.div(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.div(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return res


@implements(torch.Tensor.div)
def tdiv(inpt, other, out=None):
    """
    Division class method.
    """
    return div(inpt, other, out=out)


@implements(torch.Tensor.div_)
def idiv(inpt, other):
    """
    Inplace division.
    """
    get_base(inpt).div_(other)
    if isinstance(inpt, MaskedTensor):
        inpt.mask = torch.logical_or(get_mask(inpt), get_mask(other))

@implements(torch.permute)
def permute(inpt, dims):
    """
    Permutation of masked tensors.
    """
    base = torch.permute(inpt.base, dims)
    mask = torch.permute(inpt.mask, dims)
    return MaskedTensor(base, mask=mask, transformation=get_transformation(inpt))


@implements(torch.reshape)
def reshape(inpt, *args):
    """
    Reshaping of masked tensors.
    """
    base = torch.reshape(inpt.base, *args)
    mask = torch.reshape(inpt.mask, *args)
    return MaskedTensor(base, mask=mask, transformation=get_transformation(inpt))


@implements(torch.Tensor.reshape)
def reshape(inpt, *args):
    """
    Reshaping of masked tensors.
    """
    base = inpt.base.reshape(*args)
    mask = inpt.mask.reshape(*args)
    return MaskedTensor(base, mask=mask, transformation=get_transformation(inpt))


@implements(torch.squeeze)
def squeeze(inpt, dim=None):
    """
    Squeezing of masked tensors.
    """
    if dim is not None:
        base = torch.squeeze(inpt.base, dim=dim)
        mask = torch.squeeze(inpt.mask, dim=dim)
    else:
        base = torch.squeeze(inpt.base)
        mask = torch.squeeze(inpt.mask)
    return MaskedTensor(base, mask=mask, transformation=get_transformation(inpt))


@implements(torch.Tensor.squeeze)
def tsqueeze(inpt, dim=None):
    """
    Squeezing of masked tensors.
    """
    if dim is not None:
        base = inpt.base.squeeze(dim=dim)
        mask = inpt.mask.squeeze(dim=dim)
    else:
        base = inpt.base.squeeze()
        mask = inpt.mask.squeeze()
    return MaskedTensor(base, mask=mask, transformation=get_transformation(inpt))


@implements(torch.unsqueeze)
def unsqueeze(inpt, dim=None):
    """
    Unsqueezing of masked tensors.
    """
    base = torch.unsqueeze(inpt.base, dim=dim)
    mask = torch.unsqueeze(inpt.mask, dim=dim)
    return MaskedTensor(base, mask=mask, transformation=get_transformation(inpt))


@implements(torch.Tensor.unsqueeze)
def unsqueeze(inpt, dim=None):
    """
    Unsqueezing of masked tensors.
    """
    base = torch.unsqueeze(inpt.base, dim=dim)
    mask = torch.unsqueeze(inpt.mask, dim=dim)
    return MaskedTensor(base, mask=mask, transformation=get_transformation(inpt))


@implements(torch.sum)
def sum(inpt, dim=None):
    """
    Test summing of tensors.
    """
    return torch.where(inpt.mask, 0.0, inpt.base).sum(dim=dim)


@implements(torch.Tensor.sum)
def tsum(inpt, dim=None):
    """
    Test summing of tensors.
    """
    return torch.where(inpt.mask, 0.0, inpt.base).sum(dim=dim)


@implements(torch.mean)
def mean(inpt, dim=None):
    """
    Test mean of masked tensors.
    """
    inpt_sum = inpt.sum()
    n_elem = (~inpt.mask).sum()
    return inpt_sum / n_elem


@implements(torch.Tensor.mean)
def tmean(inpt, dim=None):
    """
    Test mean of masked tensors.
    """
    inpt_sum = inpt.sum()
    n_elem = (~inpt.mask).sum()
    return inpt_sum / n_elem


@implements(torch.sum)
def tsum(inpt, dim=None):
    """
    Test summing of tensors.
    """
    return torch.where(inpt.mask, 0.0, inpt.base).sum(dim=dim)


@implements(torch.Tensor.view)
def view(inpt, new_shape):
    """
    Reshaping of masked tensors.
    """
    base = inpt.base.view(new_shape)
    mask = inpt.mask.view(new_shape)
    return MaskedTensor(base, mask=mask, transformation=get_transformation(inpt))


@implements(torch.isclose)
def isclose(inpt, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    """
    Implementation of is close for masked tensors.
    """
    return torch.isclose(
        get_base(inpt), get_base(other), rtol=rtol, atol=atol, equal_nan=equal_nan
    )


@implements(torch.Tensor.__repr__)
def repr(inpt, **kwargs):
    """
    Implementation of __repr__ operator.
    """
    return inpt.base.__repr__(**kwargs)


@implements(torch.Tensor.eq)
@implements(torch.eq)
def eq(inpt, other, **kwargs):
    """
    Implementation of element-wise comparison.
    """
    if isinstance(inpt, MaskedTensor):
        return MaskedTensor(
            get_base(inpt).__eq__(get_base(other), **kwargs),
            mask=torch.logical_or(get_mask(inpt), get_mask(other)),
            transformation=get_transformation(inpt)
        )
    return get_base(inpt).__eq__(get_base(other), **kwargs)


@implements(torch.Tensor.to)
def to(inpt, *args, **kwargs):
    """
    Implementation of .to method.
    """
    other = inpt.base.to(*args, **kwargs)
    kwargs.pop("dtype", None)
    if len(args) > 0 and isinstance(args[0], torch.dtype):
        args = list(args)
        args[0] = bool

    mask = inpt.mask.to(*args, **kwargs)
    return MaskedTensor(other, mask=mask, transformation=get_transformation(inpt))


@implements(torch.Tensor.requires_grad.__set__)
def set_requires_grad(inpt, grad):
    inpt.base.requires_grad = grad


@implements(torch.where)
def where(cond, inpt, other, out=None):
    mask_inpt = get_mask(inpt)
    mask_other = get_mask(other)
    cond = get_base(cond)
    inpt = get_base(inpt)
    other = get_base(other)

    if out is None:
        base = torch.where(cond, inpt, other)
        mask = torch.where(cond, mask_inpt, mask_other)
        return MaskedTensor(base, mask=mask, transformation=get_transformation(inpt))

    base = torch.where(cond, inpt, other, out=out)
    mask = torch.where(cond, mask_inpt, mask_other)
    out.mask = mask
    return out


@implements(torch.ge)
def ge(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    if out is None:
        res = torch.ge(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, torch.logical_or, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.ge(get_base(inpt), get_base(other), out=get_base(out))
        mask = torch.logical_or(get_mask(inpt), get_mask(other), out=out.mask)
        return res


@implements(torch.Tensor.ge)
def tge(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    if out is None:
        res = torch.ge(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, torch.logical_or, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.ge(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(
            inpt, other, torch.logical_or, out=out.mask, shape=res.shape
        )
        return res


@implements(torch.gt)
def gt(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    if out is None:
        res = torch.gt(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, torch.logical_or, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.gt(get_base(inpt), get_base(other), out=get_base(out))
        mask = torch.logical_or(get_mask(inpt), get_mask(other), out=out.mask)
        return res


@implements(torch.Tensor.gt)
def tgt(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    if out is None:
        res = torch.gt(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, torch.logical_or, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.gt(get_base(inpt), get_base(other), out=get_base(out))
        mask = torch.logical_or(get_mask(inpt), get_mask(other), out=out.mask)
        return res


@implements(torch.le)
def le(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    if out is None:
        res = torch.le(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.le(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))


@implements(torch.Tensor.le)
def tle(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    if out is None:
        res = torch.le(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.le(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))


@implements(torch.lt)
def lt(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    if out is None:
        res = torch.lt(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.lt(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return res


@implements(torch.Tensor.lt)
def tlt(inpt, other, out=None):
    """
    Element-wise comparison of tensor
    """
    if out is None:
        res = torch.lt(
            get_base(inpt),
            get_base(other),
        )
        mask = combine_masks(inpt, other, shape=res.shape)
        return MaskedTensor(res, mask=mask, transformation=get_transformation(inpt))
    else:
        res = torch.lt(get_base(inpt), get_base(other), out=get_base(out))
        mask = combine_masks(inpt, other, out=out.mask, shape=res.shape)
        return res


@implements(torch.Tensor.type_as)
def type_as(inpt, other):
    """
    Implements function of the same name.
    """
    return inpt.to(dtype=other.dtype)


@implements(torch.Tensor.__setitem__)
def setitem(inpt, *args, **kwargs):
    """
    Setting of tensor content.
    """
    get_base(inpt).__setitem__(*[get_base(arg) for arg in args], **kwargs)


@implements(torch.nn.functional.relu)
def relu(inpt, **kwargs):
    """
    Application of ReLU activation function.
    """
    return MaskedTensor(
        torch.nn.functional.relu(inpt.base, **kwargs),
        mask=inpt.mask,
        transformation=get_transformation(inpt)
    )


@implements(torch.pow)
def pow(inpt, exp, *args, out=None):
    """
    Pow function.
    """
    return MaskedTensor(
        torch.pow(inpt.base, exp, *args, out=out),
        mask=inpt.mask,
        transformation=getattr(inpt, "__transformation__", None)
    )

@implements(torch.Tensor.pow)
def tpow(inpt, exp, *args, out=None):
    """
    Pow function.
    """
    return pow(inpt, exp, *args, out=out)


@implements(torch._C._TensorBase.pow)
def tpow(inpt, exp, *args, out=None):
    """
    Pow function.
    """
    return pow(inpt, exp, *args, out=out)


@implements(torch.Tensor.pow_)
def ipow(inpt, exp, *args):
    """
    Pow function.
    """
    return pow(inpt, exp, *args, out=inpt.base)


@implements(torch.maximum)
def maximum(inpt, other, *args):
    """
    Element-wise maximum.
    """
    min_val = torch.finfo(inpt.dtype).min

    if isinstance(inpt, MaskedTensor):
        inpt = torch.where(inpt.mask, min_val, inpt.base)

    if isinstance(other, MaskedTensor):
        other = torch.where(other.mask, min_val, other.base)

    res = torch.maximum(inpt, other)
    mask = res == min_val
    return MaskedTensor(res, mask=mask, transformation=getattr(inpt, "__transformation__", None))


@implements(torch.Tensor.maximum)
def tmaximum(inpt, exp, *args):
    """
    Member function version of maximum.
    """
    return maximum(inpt, exp, *args)


@implements(torch.max)
def max(inpt, dim, keepdim=False, *args, out=None):
    """
    Max function.
    """
    min_val = torch.finfo(inpt.dtype).min
    inpt = torch.where(inpt.mask, min_val, inpt.base)
    if dim is None:
        return inpt.max()
    res, inds = torch.max(inpt, dim, keepdim=keepdim, *args, out=out)
    if not isinstance(res, torch.Tensor):
        if res == min_val:
            return torch.return_types.max((torch.nan, inds))
        return torch.return_types.max((res, inds))
    return torch.return_types.max((MaskedTensor(res, mask=res == min_val), inds))


@implements(torch.Tensor.max)
def tmax(inpt, dim=None, keepdim=False, *args, out=None):
    """
    Member-function version of max function.
    """
    return torch.max(inpt, dim, keepdim=False, *args, out=None)


@implements(torch.minimum)
def minimum(inpt, other, *args):
    """
    Element-wise minimum.
    """
    max_val = torch.finfo(inpt.dtype).max

    if isinstance(inpt, MaskedTensor):
        inpt = torch.where(inpt.mask, max_val, inpt.base)

    if isinstance(other, MaskedTensor):
        other = torch.where(other.mask, max_val, other.base)

    res = torch.minimum(inpt, other)
    mask = res == max_val
    return MaskedTensor(res, mask=mask)


@implements(torch.Tensor.minimum)
def tminimum(inpt, exp, *args):
    """
    Member function version of minimum.
    """
    return minimum(inpt, exp, *args)


@implements(torch.min)
def min(inpt, dim, keepdim=False, *args, out=None):
    """
    Min function.
    """
    max_val = torch.finfo(inpt.dtype).max
    inpt = torch.where(inpt.mask, max_val, inpt.base)
    if dim is None:
        return inpt.min()
    res, inds = torch.min(inpt, dim, keepdim=keepdim, *args, out=out)
    if not isinstance(res, torch.Tensor):
        if res == max_val:
            return torch.return_types.min((torch.nan, inds))
        return torch.return_types.min((res, inds))
    return torch.return_types.min((MaskedTensor(res, mask=res == max_val), inds))


@implements(torch.Tensor.min)
def tmin(inpt, dim=None, keepdim=False, *args, out=None):
    """
    Member function version of min function.
    """
    return torch.min(inpt, dim, keepdim=False, *args, out=None)


@implements(torch.select)
def select(inpt, dim, index):
    """
    Select function.
    """
    return MaskedTensor(
        torch.select(inpt.base, dim, index),
        mask=torch.select(inpt.mask, dim, index),
        transformation=getattr(inpt, "__transformation__", None)
    )


@implements(torch.all)
def all(inpt, dim=None, keepdim=False, *args, out=None):
    """
    All function.
    """
    if dim is None:
        return inpt.base[~inpt.mask].all()
    res = (inpt.base | inpt.mask).all(dim=dim, keepdim=keepdim, *args, out=out)
    mask = inpt.mask.all(dim=dim, keepdim=keepdim, *args, out=out)
    return MaskedTensor(res, mask=mask, transformation=getattr(inpt, "__transformation__", None))

@implements(torch.Tensor.all)
def all(inpt, dim=None, keepdim=False, *args):
    """
    Member-function version of all function.
    """
    return torch.all(inpt, dim=dim, keepdim=keepdim, *args)


@implements(torch.any)
def any(inpt, dim=None, keepdim=False, *args, out=None):
    """
    All function.
    """
    if dim is None:
        return inpt.base[~inpt.mask].any()
    res = (inpt.base & ~inpt.mask).any(dim=dim, keepdim=keepdim, *args, out=out)
    mask = inpt.mask.any(dim=dim, keepdim=keepdim, *args, out=out)
    return MaskedTensor(res, mask=mask, transformation=getattr(inpt, "__transformation__", None))


@implements(torch.Tensor.any)
def tany(inpt, dim=None, keepdim=False, *args):
    """
    Member-function version of all function.
    """
    return torch.any(inpt, dim=dim, keepdim=keepdim, *args)


@implements(torch.transpose)
def transpose(inpt, dim1, dim2):
    """
    Transpose function.
    """
    return MaskedTensor(
        torch.transpose(inpt.base, dim1, dim2),
        mask=torch.transpose(inpt.mask, dim1, dim2),
        transformation=getattr(inpt, "__transformation__", None)
    )


@implements(torch.Tensor.transpose)
def ttranspose(inpt, dim1, dim2):
    """
    Member function version of transpose function.
    """
    return transpose(inpt, dim1, dim2)


@implements(torch.nn.functional.binary_cross_entropy_with_logits)
def binary_cross_entropy(pred, target, *args, **kwargs):
    """
    Masked binary cross entropy.
    """
    mask = ~combine_masks(pred, target)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        get_base(pred)[mask],
        get_base(target)[mask],
        *args,
        **kwargs
    )


@implements(torch.nn.functional.cross_entropy)
def cross_entropy(pred, target, *args, **kwargs):
    """
    Masked binary cross entropy.
    """
    if isinstance(pred, MaskedTensor):
        mask = get_mask(pred).any(1)
    else:
        mask = None
    if isinstance(target, MaskedTensor):
        if mask is None:
            mask = get_mask(target)
        else:
            mask = mask + get_mask(target)


    pred = pred.transpose(-1, 1)

    valid = ~mask
    return torch.nn.functional.cross_entropy(
        get_base(pred)[valid],
        get_base(target)[valid].to(dtype=torch.int64),
        *args,
        **kwargs
    )


@implements(torch.bucketize)
def bucketize(inpt, boundary, *args, **kwargs):
    """
    Masked binary cross entropy.
    """
    return MaskedTensor(
        torch.bucketize(inpt.base, boundary, *args, **kwargs),
        mask=inpt.mask,
        transformation=getattr(inpt, "__transformation__", None)
    )


@implements(torch.Tensor.clone)
def clone(inpt):
    """
    Clone masked tensor.
    """
    return MaskedTensor(
        inpt.base.clone(),
        mask=inpt.mask.clone(),
        transformation=getattr(inpt, "__transformation__", None)
    )
