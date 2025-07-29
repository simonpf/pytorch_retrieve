"""
pytorch_retrieve.inference
==========================

This module implements generic inference functionality for pytorch_retrieve retrievals.
"""

from contextlib import contextmanager
import logging
import importlib
from multiprocessing.queues import Empty
import torch.multiprocessing as mp
from pathlib import Path
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from queue import Queue

import click
import numpy as np
import torch
import toml
from torch import nn
from rich.progress import Progress
import xarray as xr

from pytorch_retrieve.tensors import (
    ProbabilityTensor,
    MeanTensor,
    QuantileTensor,
)
import pytorch_retrieve
from pytorch_retrieve.tiling import Tiler
from pytorch_retrieve.architectures import RetrievalModel, load_model
from pytorch_retrieve.config import get_config_attr, OutputConfig, InferenceConfig
from pytorch_retrieve.retrieval_output import RetrievalOutput


LOGGER = logging.getLogger(__name__)


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
    raise RuntimeError(f"Encountered unsupported type {type(inputs)} in inputs.")


def cat_n_rec(inputs: List[Any], batch_size: int) -> Tuple[Any]:
    """
    Recursively infer the batch size of a collections of tensors.

    Args:
        input: A tensor, list of tensors, dict of tensors or arbitrary nesting
             of those to process.
    """
    if not isinstance(inputs, (list, tuple)):
        raise RuntimeError("Input 'inputs' to 'cat_n_rec' must be a list.")

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
        f"Encountered unsupported type {type(tensor)} in input to 'to_rec'."
    )


def get_dimensions(
    result_name: str,
    output_cfg: Optional[Dict[str, OutputConfig]],
    inference_cfg: Optional["InferenceConfig"],
) -> Tuple[str]:
    """
    Try and infer dimensions from inference and output config.

    Args:
        result_name: The name of the inference results.
        output_cfg: A dictionary containing the output configuration of
            the model.
        inference_cfg: The inference configuration.

    Return:
        A tuple containing the dimensions name if 'result_name' is found in
        either the retrieval results or the output config. Otherwise, 'None'.
    """
    retrieval_output_cfg = None
    if inference_cfg is not None:
        for name, outputs in inference_cfg.retrieval_output.items():
            for output_name, out_cfg in outputs.items():
                if output_name == result_name:
                    retrieval_output_cfg = out_cfg

    if retrieval_output_cfg is not None:
        return tuple(retrieval_output_cfg.output.dimensions)

    if result_name in output_cfg:
        return tuple(output_cfg[result_name].dimensions)

    return None


def process(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    inference_config: InferenceConfig,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """
    Process batch of inputs.
    """
    model = to_rec(model, dtype=dtype, device=device)
    inputs = to_rec(inputs, dtype=dtype, device=device)

    results = {}
    with torch.no_grad():
        preds = model(inputs)
        if isinstance(preds, dict):
            for key, tensor in preds.items():
                if inference_config is None:
                    retrieval_output = None
                else:
                    retrieval_output = inference_config.retrieval_output.get(key, None)
                if retrieval_output is None:
                    results[key] = to_rec(tensor, device="cpu")
                else:
                    for output_name, output in retrieval_output.items():
                        results[output_name] = to_rec(
                            output.output.compute(tensor), device="cpu"
                        )
        else:
            if inference_config is None:
                results["retrieved"] = preds
            elif len(inference_config.retrieval_output) > 1:
                raise ValueError(
                    "The model output is a single tensor but retrieval outputs "
                    "for more than one output are provided."
                )
            elif len(inference_config.retrieval_output) == 1:
                retrieval_output = next(
                    iter(inference_config.retrieval_output.values())
                )
                for output_name, output in retrieval_output.items():
                    results[output_name] = to_rec(
                        output.output.compute(preds), device="cpu"
                    )
            else:
                results["retrieved"] = preds.cpu()
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
        inference_config: InferenceConfig = None,
        device: str = "cuda",
        dtype: str = "float32",
    ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.input_queue = Queue(maxsize=batch_size)
        self.output_queue = Queue()
        self.inference_config = inference_config
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype

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

        if inpt is None:
            if self.input_buffer is None or len(self.input_buffer) == 0:
                return []

            inpt, _ = cat_n_rec([self.input_buffer], self.input_buffer_size)
            res = process(
                self.model,
                inpt,
                self.inference_config,
                device=self.device,
                dtype=self.dtype,
            )
            self.output_buffer_size += self.input_buffer_size
            if self.output_buffer is None:
                self.output_buffer = res
            else:
                self.output_buffer, _ = cat_n_rec(
                    [self.output_buffer, res], self.output_buffer_size
                )

            results = []
            while (
                len(self.output_splits) > 0
                and self.output_buffer_size >= self.output_splits[0]
            ):
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
            self.input_buffer, _ = cat_n_rec(
                [self.input_buffer, inpt], self.input_buffer_size + isize
            )
        self.input_buffer_size += isize

        results = []

        while self.input_buffer_size > self.batch_size:
            batch, self.input_buffer = cat_n_rec([self.input_buffer], self.batch_size)
            self.input_buffer_size -= self.batch_size

            res = process(
                self.model,
                batch,
                self.inference_config,
                device=self.device,
                dtype=self.dtype,
            )

            self.output_buffer_size += self.batch_size
            if self.output_buffer is None:
                self.output_buffer = res
            else:
                self.output_buffer, _ = cat_n_rec(
                    [self.output_buffer, res], self.output_buffer_size
                )

        while (
            len(self.output_splits) > 0
            and self.output_buffer_size >= self.output_splits[0]
        ):
            curr_split, *self.output_splits = self.output_splits
            output, self.output_buffer = cat_n_rec([self.output_buffer], curr_split)
            self.output_buffer_size -= curr_split
            results.append(output)

        return results


def load_input_parallel(
    input_loader: Any, worker_offset: int, n_workers: int, input_queue: Queue
) -> None:
    """
    Load input samples in parallel using multiple workers.

    This requires the input_loader to have a fixed length and support item access using
    the [ind] operator.

    Args:
        input_loader: The input loader object implementing the data loading.
        worker_offset: The offset applied to each worker in the queue.
        n_workers: The number of total workers.
        input_queue: The queue in which the loaded input samples are placed.
    """
    torch.set_num_threads(1)
    assert hasattr(input_loader, "__getitem__") and hasattr(input_loader, "__len__")
    for ind in range(worker_offset, len(input_loader), n_workers):
        try:
            input_queue.put(input_loader[ind])
        except Exception as exc:
            trb = traceback.format_exc()
            exc._traceback = trb
            input_queue.put(exc)
    input_queue.join()



def load_input_sequential(input_loader: Any, input_queue: Queue) -> None:
    """
    Load input samples sequentially using a single thread.

    This is more flexible in that it does not require the input loader to support indexing.

    Args:
        input_loader: The input loader object implementing the data loading.
        input_queue: The queue in which the loaded input samples are placed.
    """
    torch.set_num_threads(1)

    input_iterator = iter(input_loader)
    while True:
        try:
            inpt = next(input_iterator)
            input_queue.put(inpt)
        except StopIteration:
            break
        except Exception as exc:
            trb = traceback.format_exc()
            exc._traceback = trb
            input_queue.put(exc)
    input_queue.join()


def finalize_results_tiled(
    input_loader: Any,
    output_config: OutputConfig,
    inference_config: InferenceConfig,
    output_path: Optional[Path],
    output_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Target function for offloading finalization of retrieval results to dedicated thread/process.

    This function waits for results to be put on the output queue and finalizes them by either
    calling the 'finalized_results' function provided by the input loader or by converting the
    results to an xarray.Dataset. If 'output_path' is not None, results will be written to that
    path. Otherwise the results will be returned as a list of xarray.Datasets.

    Args:
        input_loader: The input loader object.
        output_config: The output configuration of the model.
        inference_config: The inference configuration of the model.
        output_path: An optional output path to which the retrieval results will be written.
        result_queue: A queue object on which the retrieval results will be put.
    """
    LOGGER = logging.getLogger(__name__)
    cntr = 0

    while True:

        cntr += 1

        output = output_queue.get()

        if output is None:
            result_queue.put(None)
            break
        results_assembled, args = output

        if hasattr(input_loader, "finalize_results"):
            try:
                results = input_loader.finalize_results(results_assembled, *args, output_path=output_path)
            except Exception as exc:
                LOGGER.exception("An error occurred when finalizing the results.")
                trb = traceback.format_exc()
                exc._traceback = trb

                result_queue.put(exc)
                continue
        else:
            results = {}
            for key, tensor in results_assembled.items():
                dims = get_dimensions(key, output_config, inference_config)
                if dims is None:
                    dims = tuple([f"{key}_dim_{ind}" for ind in range(tensor.ndim - 2)])

                # Discard dummy dimensions if necessary.
                if isinstance(tensor, list):
                    tensor = [
                        t_i.squeeze() if len(dims) < t_i.ndim else t_i for t_i in tensor
                    ]
                    results[key] = (
                        tuple(dims) + (f"{key}_step", "x", "y"),
                        np.stack(tensor),
                    )
                elif len(dims) < tensor.ndim:
                    tensor = torch.squeeze(tensor)
                    results[key] = (tuple(dims) + ("x", "y"), tensor)

            results = xr.Dataset(results)

        if isinstance(results, (str, Path)):
            result_queue.put(results)
        else:
            if output_path is not None:
                filename = f"results_{cntr}.nc"
                if isinstance(results, tuple):
                    results, filename = results
                if output_path.is_dir():
                    results.to_netcdf(output_path / filename)
                    result_queue.put(output_path / filename)
                else:
                    results.to_netcdf(output_path)
                    result_queue.put(output_path)
            else:
                result_queue.put(results)


def finalize_results_no_tiling(
    input_loader: Any,
    output_config: OutputConfig,
    inference_config: InferenceConfig,
    output_path: Optional[Path],
    output_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Target function for offloading finalization of retrieval results to dedicated thread/process.

    This function waits for results to be put on the output queue and finalizes them by either
    calling the 'finalized_results' function provided by the input loader or by converting the
    results to an xarray.Dataset. If 'output_path' is not None, results will be written to that
    path. Otherwise the results will be returned as a list of xarray.Datasets.

    Args:
        input_loader: The input loader object.
        output_config: The output configuration of the model.
        inference_config: The inference configuration of the model.
        output_path: An optional output path to which the retrieval results will be written.
        output_queue: The queue collecting the raw retrieval output.
        result_queue: A queue object on which the retrieval results will be put.
    """
    cntr = 0
    while True:

        cntr += 1

        output = output_queue.get()
        if output is None:
            result_queue.put(None)
            break
        results, args = output

        if hasattr(input_loader, "finalize_results"):
            try:
                results = input_loader.finalize_results(results, *args, output_path=output_path)
            except Exception as exc:
                trb = traceback.format_exc()
                exc._traceback = trb
                result_queue.put(exc)
                continue
        else:
            results = {}
            for key, tensor in results.items():
                dims = get_dimensions(key, output_config, inference_config)
                if dims is None:
                    dims = tuple([f"{key}_dim_{ind}" for ind in range(tensor.ndim - 2)])
                if len(dims) < tensor.ndim:
                    tensor = torch.squeeze(tensor)

                results[key] = (("samples",) + tuple(dims), tensor)
            results = xr.Dataset(results)

        if isinstance(results, (str, Path)):
            result_queue.put(results)
        else:
            if output_path is not None:
                filename = f"results_{cntr}.nc"
                if isinstance(results, tuple):
                    results, filename = results
                if output_path.is_dir():
                    results.to_netcdf(output_path / filename)
                    result_queue.put(output_path / filename)
                else:
                    results.to_netcdf(output_path)
                    result_queue.put(output_path)
            else:
                result_queue.put(results)


class SimpleInput:
    """
    Input loader for the case that input data for a single sample is provided directly in the form
    of a list, dict, or tensor.
    """
    def __init__(self, inpt: Any):
        """
        Args:
            inpt:
        """
        self.inpt = inpt

    def __len__(self) -> int:
        if isinstance(self.inpt, list):
            return len(self.inpt)
        else:
            return 1

    def __iter__(self) -> Any:
        if isinstance(self.inpt, list):
            for inpt in self.inpt:
                yield (inpt,)
        else:
            yield (self.inpt,)

    def __getitem__(self, ind: int):
        if isinstance(self.inpt, list):
            return (self.inpt[ind],)
        return (self.inpt,)


class InferenceRunner:
    """
    The inference runner coordinates the running of the inference.
    """

    def __init__(
        self,
        model: nn.Module,
        input_loader: Any,
        inference_config: InferenceConfig,
        n_input_loaders: int = 1,
    ):
        self.model = model
        self.input_loader = input_loader
        self.inference_config = inference_config
        self.n_input_loaders = n_input_loaders
        self.input_queue = mp.JoinableQueue(n_input_loaders)
        self.output_queue = mp.Queue(4)
        self.result_queue = mp.Queue()

    @contextmanager
    def start_workers(self, n_input_loaders: int, output_path: Optional[Path]) -> None:
        """
        Start input loading processes or thread.

        If n_input_loaders is '1', this function will start a single thread to load input data
        from the input loader. If n_input_loaders is larger than one it will use multiple processes
        to load the input samples in parallel.

        Args:
            n_input_loader: The number of loader processes to use for data loading.
        """
        self.input_workers = []
        if n_input_loaders > 1:
            if not hasattr(self.input_loader, "__len__") or not hasattr(
                self.input_loader, "__getitem__"
            ):
                raise ValueError(
                    "In order to use multiple input loaders, the input loader must support "
                    "'len' and item access '[]' operations."
                )
            for ind in range(n_input_loaders):
                process = mp.Process(
                    target=load_input_parallel,
                    args=(self.input_loader, ind, n_input_loaders, self.input_queue),
                )
                self.input_workers.append(process)
        else:
            process = mp.Process(
                target=load_input_sequential, args=(self.input_loader, self.input_queue)
            )
            self.input_workers.append(process)
        for worker in self.input_workers:
            worker.start()

        if isinstance(self.model, RetrievalModel):
            output_config = self.model.output_config
        else:
            output_config = None

        if self.inference_config.tile_size is not None:
            self.output_worker = mp.Process(
                target=finalize_results_tiled,
                args=(
                    self.input_loader,
                    output_config,
                    self.inference_config,
                    output_path,
                    self.output_queue,
                    self.result_queue,
                ),
            )
        else:
            self.output_worker = mp.Process(
                target=finalize_results_no_tiling,
                args=(
                    self.input_loader,
                    output_config,
                    self.inference_config,
                    output_path,
                    self.output_queue,
                    self.result_queue,
                ),
            )
        self.output_worker.start()

        try:
            yield None
        finally:
            for input_worker in self.input_workers:
                if isinstance(input_worker, mp.Process):
                    input_worker.terminate()
                input_worker.join()
            if isinstance(self.output_worker, mp.Process):
                self.output_worker.terminate()
            self.output_worker.join()
            for queue in [self.input_queue, self.output_queue, self.result_queue]:
                queue.close()
                queue.join_thread()

    def run(
        self,
        output_path,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> List[Union[Path, xr.Dataset]]:
        """
        Run inference.

        This iterates over all inputs provided by the input loader, processes them using the
        given model, and finalizes the results.

        Args:
            output_path: An optional output path to which to write the retrieval results.
                If 'None', retrieval results will be returned as xr.Datasets.
            device: The device to run the ML inference on.
            dtype: The dtype to use for the processing.
            n_input_loaders: The number of concurrent processes to use to load the input.

        Return:
            If 'output_path' is not None, a list of the produced output files is returned.
            If 'output_path' is None, a list containing the retrieval results as xarray.Datasets
            is returned.
        """

        outputs = []
        arg_stack = []
        cntr = 1

        model = self.model.eval()
        if isinstance(model, RetrievalModel):
            output_cfg = model.output_config
        else:
            output_cfg = None

        exclude_from_tiling = self.inference_config.exclude_from_tiling
        if exclude_from_tiling is None:
            exclude_from_tiling = []

        with self.start_workers(self.n_input_loaders, output_path):
            with Progress() as progress:
                task = progress.add_task(
                    "Processing input:", total=len(self.input_loader)
                )

                while any([worker.is_alive() for worker in self.input_workers]):

                    try:
                        input_data = self.input_queue.get(timeout=1)
                    except Empty:
                        continue
                    except mp.TimeoutError:
                        if any([worker.is_alive() for worker in self.input_workers]):
                            break
                        else:
                            continue

                    if input_data is None:
                        self.input_queue.task_finished()
                        break

                    if isinstance(input_data, Exception):
                        self.input_queue.task_done()
                        exc = input_data
                        LOGGER.error(
                            "A '%s' was encountered when loading input data: \nMessage: %s \nTraceback: %s",
                            exc.__class__.__name__,
                            exc,
                            exc._traceback,
                        )
                        continue
                    args = []
                    if isinstance(input_data, tuple):
                        input_data, *args = input_data

                    tile_size = self.inference_config.tile_size
                    overlap = self.inference_config.spatial_overlap
                    batch_size = self.inference_config.batch_size

                    processor = BatchProcessor(
                        model,
                        batch_size=batch_size,
                        inference_config=self.inference_config,
                        device=device,
                        dtype=dtype,
                    )

                    if tile_size is not None:
                        if overlap is None:
                            overlap = (tile_size[0] // 8, tile_size[1] // 8)

                        if exclude_from_tiling is not None:
                            not_tiled = {
                                name: input_data.pop(name)
                                for name in exclude_from_tiling
                            }
                        else:
                            not_tiled = {}
                        tiler = Tiler(input_data, tile_size=tile_size, overlap=overlap)

                        results_tiled = np.array([[None] * tiler.N] * tiler.M)
                        tile_stack = []

                        for row_ind in range(tiler.M):
                            for col_ind in range(tiler.N):
                                tiled_input = tiler.get_tile(row_ind, col_ind)
                                if len(not_tiled) > 0:
                                    tiled_input.update(not_tiled)

                                results_s = processor.process(tiled_input)
                                tile_stack.append((row_ind, col_ind))
                                for output in results_s:
                                    tile_inds, *tile_stack = tile_stack
                                    results_tiled.__setitem__(tile_inds, output)

                        results_s = processor.process(None)
                        for output in results_s:
                            tile_inds, *tile_stack = tile_stack
                            results_tiled.__setitem__(tile_inds, output)

                        assert len(tile_stack) == 0

                        results_ass = tiler.assemble(results_tiled)
                        self.output_queue.put((results_ass, args))
                    else:
                        arg_stack.append(args)
                        results_stack = processor.process(input_data)
                        results_stack += processor.process(None)
                        for results in results_stack:
                            args, *arg_stack = arg_stack
                            self.output_queue.put((results, args))

                    cntr += 1
                    progress.update(task, advance=1.0)
                    self.input_queue.task_done()

            self.output_queue.put(None)

            outputs = []
            while self.output_worker.is_alive() or self.result_queue.qsize() > 0:
                output = self.result_queue.get()
                if output is None:
                    break
                if isinstance(output, Exception):
                    exc = output
                    LOGGER.error(
                        "A '%s' was encountered when loading input data: \nMessage: %s \nTraceback: %s",
                        exc.__class__.__name__,
                        exc,
                        exc._traceback,
                    )
                outputs.append(output)

            return outputs


class SequentialInferenceRunner:
    """
    Inference runner that doesn't use any parallel processing.
    """
    def __init__(
        self,
        model: nn.Module,
        input_loader: Any,
        inference_config: InferenceConfig,
    ):
        """
        Args:
            model: The retrieval model to perform inference with.
            input_loader: An input-loader object to load the retrieval input data.
            inference_config: The inference configuration.
        """
        self.model = model
        self.input_loader = input_loader
        self.inference_config = inference_config
        self.tile_callback = None

    def process_simple(
            self,
            counter: int,
            input_data: Dict[str, torch.Tensor],
            input_args: List[Any],
            processor: BatchProcessor,
            input_loader: "InputLoader",
            output_config: OutputConfig,
            output_path: Optional[Path]
    ):
        """
        Processes inputs without tiling.

        Uses the batch_processor to batch inputs and process them. The results are finalized
        using either the input_loader's 'finalize_results' method or converted to an xarray.Dataset.
        The results are stored in NetCDF4 to output_path if given otherwise the xarray.Dataset
        is returned directly.

        Args:
            counter: A counter tracking which input sample is being processed.
            input_data: The input data for the current sample.
            input_args: The auxiliary arguments returned from the data loader.
            processor: The batch processor to use for processing.
            input_loader: The input laoder.
            output_config: The output config of the retrieval model. Used to infer dimensions of
                the output results.
            output_path: Optional path to which to write the outputs.

        Return:
            An xarray.Dataset containing the results or a path pointing to the file to which
            the retrieval results were written.
        """
        results_stack = processor.process(input_data)
        results_stack += processor.process(None)
        for results in results_stack:
            if hasattr(input_loader, "finalize_results"):
                try:
                    results = input_loader.finalize_results(results, *input_args, output_path=output_path)
                except Exception as exc:
                    LOGGER.exception(
                        "Encoutered an error when finalizing retrieval results."
                    )
            else:
                results = {}
                for key, tensor in results.items():
                    dims = get_dimensions(key, output_config, self.inference_config)
                    if dims is None:
                        dims = tuple([f"{key}_dim_{ind}" for ind in range(tensor.ndim - 2)])
                    if len(dims) < tensor.ndim:
                        tensor = torch.squeeze(tensor)

                    results[key] = (("samples",) + tuple(dims), tensor)
                results = xr.Dataset(results)

            if isinstance(results, (str, Path)):
                return results

            filename = f"results_{counter}.nc"
            if isinstance(results, tuple):
                results, filename = results
            if output_path is not None:
                if output_path.is_dir():
                    results.to_netcdf(output_path / filename)
                    return output_path / filename
                else:
                    results.to_netcdf(output_path)
                    return output_path / filename

            return results

    def process_tiled(
            self,
            counter: int,
            input_data: Dict[str, torch.Tensor],
            input_args: List[Any],
            processor: BatchProcessor,
            input_loader: "InputLoader",
            output_config: OutputConfig,
            output_path: Path,
            tile_size: Tuple[int, int],
            overlap: Optional[int] = None,
            exclude_from_tiling: Optional[List[str]] = None,
            progress_bar: Optional[Progress] = None
    ):
        """
        Processes inputs with tiling.

        Tiles the input data, processes each tile an re-assembles the outputs. The assembled results
        are finalized using either the input_loader's 'finalize_results' method or converted to an xarray.Dataset.
        The results are stored in NetCDF4 to output_path if given otherwise the xarray.Dataset
        is returned directly.

        Args:
            counter: A counter tracking which input sample is being processed.
            input_data: The input data for the current sample.
            input_args: The auxiliary arguments returned from the data loader.
            processor: The batch processor to use for processing.
            input_loader: The input laoder.
            output_config: The output config of the retrieval model. Used to infer dimensions of
                the output results.
            output_path: Optional path to which to write the outputs.
            tile_size: A tuple defining the tile size to use for the processing.
            overlap: The size of the overlap between neighboring tiles.
            exclude_from_tiling: Names of the variables to exlude from tiling.
            progress_bar: An optional rich.Progress object to use to track the progress of the tile
                processing.

        Return:
            An xarray.Dataset containing the results or a path pointing to the file to which
            the retrieval results were written.
        """
        if overlap is None:
            overlap = (tile_size[0] // 8, tile_size[1] // 8)

        if exclude_from_tiling is not None:
            not_tiled = {
                name: input_data.pop(name)
                for name in exclude_from_tiling
            }
        else:
            not_tiled = {}

        tiler = Tiler(input_data, tile_size=tile_size, overlap=overlap)
        if self.tile_callback is not None:
            tiler.tile_callback = self.tile_callback


        results_tiled = np.array([[None] * tiler.N] * tiler.M)
        tile_stack = []

        total = tiler.M * tiler.N

        if progress_bar is not None:
            task = progress_bar.add_task(f"[bold hot_pink3] >>> Processing tile: 1/{total}", total=total)
        else:
            task = None

        tile_ctr = 0
        for row_ind in range(tiler.M):
            for col_ind in range(tiler.N):
                tiled_input = tiler.get_tile(row_ind, col_ind)
                if len(not_tiled) > 0:
                    tiled_input.update(not_tiled)

                results_s = processor.process(tiled_input)
                tile_stack.append((row_ind, col_ind))
                for output in results_s:
                    tile_ctr += 1
                    tile_inds, *tile_stack = tile_stack
                    results_tiled.__setitem__(tile_inds, output)
                    if progress_bar is not None:
                        progress_bar.update(
                            task,
                            description=f"[bold hot_pink3]Processing tile: {min(tile_ctr + 1, total)}/{total}",
                            advance=1.0
                        )



        results_s = processor.process(None)
        for output in results_s:
            tile_ctr += 1
            tile_inds, *tile_stack = tile_stack
            results_tiled.__setitem__(tile_inds, output)
            if progress_bar is not None:
                progress_bar.update(
                    task,
                    description=f"[bold hot_pink3]Processing tile: {min(tile_ctr + 1, total)}/{total}",
                    advance=1.0
                )

        if progress_bar is not None:
            progress_bar.remove_task(task)

        assert len(tile_stack) == 0

        results_ass = tiler.assemble(results_tiled)

        if hasattr(input_loader, "finalize_results"):
            try:
                results = input_loader.finalize_results(results_ass, *input_args, output_path=output_path)
            except Exception as exc:
                LOGGER.exception("An error occurred when finalizing the results.")
                return None
        else:
            results = {}
            for key, tensor in results_ass.items():
                dims = get_dimensions(key, output_config, self.inference_config)
                if dims is None:
                    dims = tuple([f"{key}_dim_{ind}" for ind in range(tensor.ndim - 2)])

                # Discard dummy dimensions if necessary.
                if isinstance(tensor, list):
                    tensor = [
                        t_i.squeeze() if len(dims) < t_i.ndim else t_i for t_i in tensor
                    ]
                    results[key] = (
                        tuple(dims) + (f"{key}_step", "x", "y"),
                        np.stack(tensor),
                    )
                elif len(dims) < tensor.ndim:
                    tensor = torch.squeeze(tensor)
                    results[key] = (tuple(dims) + ("x", "y"), tensor)
            results = xr.Dataset(results)

        filename = f"results_{counter}.nc"
        if isinstance(results, (str, Path)):
            return results
        if isinstance(results, tuple):
            results, filename = results
        if output_path is not None:
            if output_path.is_dir():
                results.to_netcdf(output_path / filename)
                return output_path / filename
            else:
                results.to_netcdf(output_path)
                return output_path

        return results


    def run(
        self,
        output_path,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> List[Union[Path, xr.Dataset]]:
        """
        Run inference.

        This iterates over all inputs provided by the input loader, processes them using the
        given model, and finalizes the results.

        Args:
            output_path: An optional output path to which to write the retrieval results.
                If 'None', retrieval results will be returned as xr.Datasets.
            device: The device to run the ML inference on.
            dtype: The dtype to use for the processing.
            n_input_loaders: The number of concurrent processes to use to load the input.

        Return:
            If 'output_path' is not None, a list of the produced output files is returned.
            If 'output_path' is None, a list containing the retrieval results as xarray.Datasets
            is returned.
        """
        if output_path is not None:
            output_path = Path(output_path)

        outputs = []
        arg_stack = []
        cntr = 1

        model = self.model.eval()
        if isinstance(model, RetrievalModel):
            output_cfg = model.output_config
        else:
            output_cfg = None

        exclude_from_tiling = self.inference_config.exclude_from_tiling
        if exclude_from_tiling is None:
            exclude_from_tiling = []

        input_iterator = iter(self.input_loader)
        cntr = 0

        try:
            total = len(self.input_loader)
        except AttributeError:
            total = "?"

        with Progress() as progress:
            task = progress.add_task(
                f"[bold dark_orange]Processing retrieval input: 1/{total}", total=len(self.input_loader)
            )

            while True:
                try:
                    input_data = next(input_iterator)
                except StopIteration:
                    break
                except Exception as exc:
                    LOGGER.exception(
                        "Encountered an error when loading input data."
                    )
                    cntr += 1
                    continue

                args = []
                if isinstance(input_data, (list, tuple)):
                    input_data, *args = input_data
                tile_size = self.inference_config.tile_size
                overlap = self.inference_config.spatial_overlap
                batch_size = self.inference_config.batch_size

                processor = BatchProcessor(
                    model,
                    batch_size=batch_size,
                    inference_config=self.inference_config,
                    device=device,
                    dtype=dtype,
                )

                if tile_size is not None:
                    outputs.append(
                        self.process_tiled(
                            cntr,
                            input_data,
                            args,
                            processor,
                            self.input_loader,
                            output_cfg,
                            output_path,
                            tile_size=tile_size,
                            overlap=overlap,
                            exclude_from_tiling=exclude_from_tiling,
                            progress_bar=progress
                        )
                    )
                else:
                    outputs.append(
                        self.process_simple(
                            cntr,
                            input_data,
                            args,
                            processor,
                            self.input_loader,
                            output_cfg,
                            output_path
                        )
                    )

                cntr += 1
                progress.update(
                    task,
                    description=f"[bold dark_orange] Processing retrieval input: {min(cntr + 1, total)}/{total}",
                    advance=1.0
                )

        return outputs


def run_inference(
    model: nn.Module,
    input_loader: Any,
    inference_config: InferenceConfig,
    output_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    exclude_from_tiling: Optional[List[str]] = None,
    tile_callback = None
) -> Union[List[Path], List[xr.Dataset]]:
    """
    Run inference using the given model on a sequence of inputs provided by an
    input loader.

    Args:
        model: The retrieval model to use for inference.
        input_loader: A loader object providing access to the input data.
        inference_config: An InferenceConfig object defining the details of the inference to
            perform.
        output_path: An optional output path to which to write the results.
        device: The device on which to perform the inference.
        dtype: The floating point type to use for the inference.
        exclude_from_tiling: List of input tensor names to exclude from tiling.

    Return:
        If an output path is provided, a list of the output files that were written is returned.
        If no output path is provided, the retrieval results are returned as a list of xarray.Datasets.
    """
    if isinstance(input_loader, (torch.Tensor, list, dict)):
        input_loader = SimpleInput(input_loader)
    runner = SequentialInferenceRunner(
        model, input_loader, inference_config
    )
    runner.tile_callback = tile_callback
    return runner.run(output_path=output_path, device=device, dtype=dtype)


@click.argument(
    "model",
    type=str,
)
@click.argument("input_path", type=str, nargs=-1)
@click.option(
    "--input_loader",
    type=str,
    help="Name of the input loader class to use to load the input data.",
)
@click.option(
    "--input_loader_args",
    type=str,
    help=(
        "A string specifying a Python dictionary that will be passed as additional "
        "arguments to the __init__ call of the input loader."
    ),
)
@click.option(
    "--output_path",
    type=str,
    default=None,
    help=("An optional destination to which to write the inference results."),
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help=("The device on which to perform inference."),
)
@click.option(
    "--dtype",
    type=str,
    default="float32",
    help=("The floating point type to use for inference."),
)
@click.option(
    "--inference_config",
    type=str,
    default=None,
    help=("Path of an inference config file to load."),
)
def cli(
    model: str,
    input_path: Path,
    input_loader: Optional[str] = None,
    input_loader_args: Dict[str, Any] = {},
    output_path: Optional[Path] = None,
    device: str = "cpu",
    dtype: str = "float32",
    inference_config: Optional[str] = None,
) -> None:
    """
    Run inference using MODEL on input files in INPUT_PATH.

    This function will load retrieval input data from INPUT_PATH using the provided
    INPUT_LOADER class. Results will be written to the current working directory
    or the location pointed to by --output_path if provfided.
    """

    try:
        model = load_model(model).eval()
    except Exception:
        LOGGER.exception(
            "Encountered the following error when trying to load the model from "
            " file '%s'.",
            model,
        )
        return 1

    if inference_config is not None:
        inference_config = Path(inference_config)
        if not inference_config.exists():
            raise ValueError(
                "If given, 'inference_config' must point to an existing file."
            )
        inference_config = InferenceConfig.parse(
            model.output_config, toml.loads(open(inference_config).read())
        )
    else:
        inference_config = model.inference_config
        if inference_config is None:
            inference_config = InferenceConfig()

    if input_loader is None:
        input_loader = inference_config.input_loader

    if input_loader is None:
        LOGGER.error(
            "The inference config (provided implicitly by the retrieval model or "
            " explicitly using the '--inference_config' option ) doesn't contain an "
            "'input loader'. In that case, the '--input_loader' (and, if required, "
            " the '--input_loader_args') options must be provided."
        )
        return 1

    input_loader_parts = input_loader.split(".")
    input_loader_module = ".".join(input_loader_parts[:-1])
    try:
        module = importlib.import_module(input_loader_module)
    except ImportError:
        LOGGER.error(
            "Could not import the module '%s' containing the data loader. Please make sure "
            "that the corresponding package and all of its dependencies are installed.",
            input_loader_module,
        )
        return 1

    if input_loader_args is not None:
        if isinstance(input_loader_args, str):
            try:
                input_loader_args = eval(input_loader_args)

            except Exception:
                LOGGER.error(
                    "Encountered an error when trying to parse the 'input_loader_args' dict."
                )
                return 1
            if inference_config.input_loader_args is not None:
                input_loader_args = inference_config.input_loader_args.update(
                    input_loader_args
                )
    else:
        input_loader_args = inference_config.input_loader_args
        if input_loader_args is None:
            input_loader_args = {}

    if len(input_path) == 1:
        input_path = input_path[0]

    try:
        input_loader = getattr(module, input_loader_parts[-1])(
            input_path, **input_loader_args
        )
    except Exception:
        LOGGER.exception(
            "Encountered the following error when trying to instantiate the input loader."
        )
        return 1

    if output_path is None:
        output_path = Path(".")
    else:
        output_path = Path(output_path)

    device = torch.device(device)
    dtype = getattr(torch, dtype)

    runner = InferenceRunner(
        model,
        input_loader,
        inference_config=inference_config,
    )
    runner.run(output_path=output_path, device=device, dtype=dtype)
