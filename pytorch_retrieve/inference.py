"""
pytorch_retrieve.inference
==========================



"""
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Tuple
from queue import Queue

from pytorch_retrieve.tensors import (
    ProbabilityTensor,
    MeanTensor,
    QuantileTensor,
)


import torch
from torch import nn


def batch_size_rec(inputs) -> int:
    """
    Recursively infer the batch size of a collections of tensors.

    Args:
        input: A tensor, list of tensors, dict of tensors or arbitrary nesting
             of those to process.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.shape[0]
    elif isinstance(inputs, (list, tuple)):
        return batch_size_rec(inputs[0])
    elif isinstance(inputs, dict):
        return batch_size_rec(next(iter(inputs.values())))
    raise RuntimeError(
        f"Encountered unsupported type {type(inputs)} in inputs."
    )


def cat_n_rec(inputs: List[Any], batch_size: int) -> Tuple[Any]:
    """
    Recursively infer the batch size of a collections of tensors.

    Args:
        input: A tensor, list of tensors, dict of tensors or arbitrary nesting
             of those to process.
    """
    if not isinstance(inputs, (list, tuple)):
        raise RuntimeError(
            "Input 'inputs' to 'cat_n_rec' must be a list."
        )

    if isinstance(inputs[0], torch.Tensor):
        flat = torch.cat(inputs, 0)
        head, tail = flat[:batch_size], flat[batch_size:]
        return (head, tail)

    if isinstance(inputs[0], (list, Tuple)):
        heads = []
        tails = []
        for head, tail in [cat_n_rec(zpd, batch_size) for zpd in zip(*inputs)]:
            heads.append(head)
            tails.append(tail)
        return heads, tails

    if isinstance(inputs[0], dict):
        heads = {}
        tails = {}
        for key in inputs[0]:
            head, tail = cat_n_rec([inpt[key] for inpt in inputs], batch_size)
            heads[key] = head
            tails[key] = tail
        return heads, tails

    raise RuntimeError(
        f"Encountered unsupported type {type(inputs[0])} in input to 'cat_n_rec'."
    )


def to_rec(tensor, device=None, dtype=None) -> Any:
    """
    Recursive appliation of 'tensor.to(...)' to a collectiosn of tensors.

    Args:
        device: The device to which to move the tensor.
        dtype: The dtype to which to convert the tensor.

    Return:
        The same collection of tensor but moved to the given device and
        transformed to the given dtype.
    """
    if isinstance(tensor, (torch.Tensor, nn.Module)):
        return tensor.to(device=device, dtype=dtype)
    if isinstance(tensor, list):
        return [to_rec(tnsr, device=device, dtype=dtype) for tnsr in tensor]
    if isinstance(tensor, tuple):
        return tuple([to_rec(tnsr, device=device, dtype=dtype) for tnsr in tensor])
    if isinstance(tensor, dict):
        return {
            key: to_rec(tnsr, device=device, dtype=dtype)
            for key, tnsr in tensor.items()
        }

    raise RuntimeError(
        f"Encountered unsupported type {type(inputs)} in input to 'to_rec'."
    )


@dataclass
class InferenceConfig:
    """
    Defines which output quantities to compute for a given output.
    """
    tile_size: Optional[int] = None
    spatial_overlap: Optional[int] = None
    temporal_overlap: Optional[int] = None



def process(
        model: nn.Module,
        inputs,
        inference_config: Dict[str, InferenceConfig],
):
    """
    Process batch of inputs.
    """
    results = {}

    with torch.no_grad():
        preds = model(inputs)
        if isinstance(preds, torch.Tensor):
            preds = {"retrieved": preds}
        for key, tensor in preds.items():
            if isinstance(tensor, (QuantileTensor, MeanTensor, ProbabilityTensor)):
                results[key] = tensor.expected_value().cpu()
            else:
                results[key] = tensor.cpu()

    return results

    

class BatchProcessor:
    """
    The batch processor implements performs buffered processing of retrieval inputs by combining
    inputs into batches of a given size.
    """
    def __init__(
            self,
            model: nn.Module,
            batch_size: int,
            inference_config: Dict[str, InferenceConfig] = None,
            device: str = "cuda",
            dtype: str = "float32"
    ):
        super().__init__()
        self.model = torch.load(model)
        self.batch_size = batch_size
        self.input_queue = Queue(maxsize=batch_size)
        self.output_queue = Queue()

        if inference_config is None:
            inference_config = {}
        self.inference_config = inference_config
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)

        self.input_buffer_size = 0
        self.input_buffer = None
        self.output_buffer_size = 0
        self.output_buffer = None
        self.output_splits = []

    def process(self, inpt: Any) -> List[Any]:
        """
        Process inputs in batches.

        This method should be called with a sequence of loaded inputs. It returns a list containing
        the outputs that could be processes using full batches. If the number of provided input
        samples is lower than that the an empty list is returned as the processor waits to fill up
        the input buffer to reach the desired batch size. Due to the buffering, the process function
        may return more than one output with the first returned element corresponding to the inputs from
        previous 'process' calls.

        Args:
            inpt: A collection of tensors containing the input tensors to process. If 'None' the
                currently buffered inputs will be processed and the corresponding outputs
                returned.

        Return:
             A containing all the results that could be processed in full batches using
             the given batch size. This list may be empty.
        """

        if inpt is None: if self.input_buffer is None or len(self.input_buffer) == 0: return []

            inpt, _ = cat_n_rec([self.input_buffer], self.input_buffer_size)
            res = process(
                self.model,
                inpt,
                self.inference_config
            )
            self.output_buffer_size += self.input_buffer_size
            if self.output_buffer is None:
                self.output_buffer = res
            else:
                self.output_buffer, _ = cat_n_rec([self.output_buffer, res], self.output_buffer_size)

            results = []
            while len(self.output_splits) > 0  and self.output_buffer_size >= self.output_splits[0]:
                curr_split, *self.output_splits = self.output_splits
                output, self.output_buffer = cat_n_rec([self.output_buffer], curr_split)
                self.output_buffer_size -= curr_split
                results.append(output)

            assert len(self.output_splits) == 0
            assert self.output_buffer_size == 0
            return results


        isize = batch_size_rec(inpt)
        self.output_splits.append(isize)

        if self.input_buffer is None:
            self.input_buffer = inpt
        else:
            self.input_buffer, _ = cat_n_rec([self.input_buffer, inpt], self.input_buffer_size + isize)
        self.input_buffer_size += isize

        results = []

        while self.input_buffer_size > self.batch_size:

            batch, self.input_buffer = cat_n_rec([self.input_buffer], self.batch_size)
            self.input_buffer_size -= self.batch_size

            res = process(self.model, batch, self.inference_config)

            self.output_buffer_size += self.batch_size
            if self.output_buffer is None:
                self.output_buffer = res
            else:
                self.output_buffer, _ = cat_n_rec(
                    [self.output_buffer, res], self.output_buffer_size
                )

        while len(self.output_splits) > 0 and self.output_buffer_size >= self.output_splits[0]:
            curr_split, *self.output_splits = self.output_splits
            output, self.output_buffer = cat_n_rec([self.output_buffer], curr_split)
            self.output_buffer_size -= curr_split
            results.append(output)


        return results


def run_inference(
        model: nn.Module,
        input_path: Path,
        input_loader: Any,
        input_loader_args: Dict[str, Any],
        inference_config: InferenceConfig
) -> None:
    """
    Run inference.
    """

    input_loader = input_loader(input_path, **input_loader_args)
    for input_data, *args in input_loader:
