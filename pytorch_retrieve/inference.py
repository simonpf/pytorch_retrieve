"""
pytorch_retrieve.inference
==========================

This module implements generic inference functionality for pytorch_retrieve retrievals.
"""
from dataclasses import dataclass, asdict
import logging
import importlib
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union
from queue import Queue

import click
import numpy as np
import torch
from torch import nn
from rich.progress import Progress
import xarray as xr

from pytorch_retrieve.tensors import (
    ProbabilityTensor,
    MeanTensor,
    QuantileTensor,
)
import pytorch_retrieve
from pytorch_retrieve.architectures import RetrievalModel
from pytorch_retrieve.tiling import Tiler
from pytorch_retrieve.architectures import load_model
from pytorch_retrieve.config import get_config_attr, OutputConfig
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
            for output_name, output_cfg in outputs.items():
                if output_name == result_name:
                    retrieval_output_cfg = output_cfg
    if retrieval_output_cfg is not None:
        return tuple(retrieval_output_cfg.output.dimensions)

    if result_name in output_cfg:
        return tuple(output_cfg[results_name].dimensions)

    return None


@dataclass
class RetrievalOutputConfig:
    """
    Describes inference output to be calculated from the model predictions.
    """
    output_config: OutputConfig
    retrieval_output: str
    parameters: Optional[Dict[str, Any]]

    def __init__(
            self,
            output_config: OutputConfig,
            retrieval_output: str,
            parameters: Optional[Dict[str, Any]] = None,
    ):
        self.output_config = output_config
        self.retrieval_output = retrieval_output
        self.parameters = parameters

        try:
            output_class = getattr(pytorch_retrieve.retrieval_output, retrieval_output)
        except AttributeError:
            raise RuntimeError(
                f"Could not find a retrieval output class matching the name '{retrieval_output}'. Please "
                "refer to the 'pytorch_retrieve.retrieval_output' module for available output classes."
            )

        try:
            if parameters is None:
                self.output = output_class(output_config)
            else:
                self.output = output_class(output_config, **parameters)
        except ValueError:
            raise RuntimeError(
                f"Coud not instantiate retrieval output class '{retrieval_output}' with given parameters "
                f"{parameters}. Plase refer to the 'pytorch_retrieve.retrieval_output' module for available "
                " output classes."
            )


    @staticmethod
    def parse(
            name: str,
            output_cfg: OutputConfig,
            config_dict: Union[str, Dict[str, Any]]
    ) -> "RetrievalOuputConfig":
        """
        Parse retrieval output config.

        Args:
            name: The name of the retrieval output.
            output_cfg: The OutputConfig object describing the model output from which the retrieval
                output is computed.
            config_dict: The dictionary from which to parse the RetrievalOutputConfig.
        """
        if isinstance(config_dict, str):
            return RetrievalOutputConfig(output_cfg, config_dict, None)

        retrieval_output = config_dict.pop("retrieval_output", None)
        if retrieval_output is None:
            raise RuntimeError(
                f"Retrieval output entry for output '{name}' lacks '{retrieval_output}' attribute."
            )
        return RetrievalOutputConfig(
            output_config=output_cfg,
            retrieval_output=retrieval_output,
            parameters=config_dict,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Return dictionary representation of retrieval output.
        """
        dct = {"retrieval_output": self.retrieval_output}
        if self.parameters is not None:
            dct.update(self.parameters)
        return dct


@dataclass
class InferenceConfig:
    """
    Defines which output quantities to compute for a given output.
    """
    batch_size: int = 8
    tile_size: Optional[Tuple[int, int]] = None
    spatial_overlap: Optional[Tuple[int, int]] = None
    temporal_overlap: Optional[int] = None
    input_loader: Optional[str] = None
    input_loader_args: Optional[Dict[str, Any]] = None
    retrieval_output: Optional[Dict[str, Dict[str, RetrievalOutputConfig]]] = None

    @staticmethod
    def parse(
        output_cfg: Dict[str, OutputConfig],
        config_dict: Dict[str, Any],
    ) -> "InferenceConfig":
        """
        Parse inference config.

        Args:
            config_dict: A dictionary describing the inference settings.
            output_cfg: A dictionary mapping model output names to corresponding OutputConfig
                objects.

        Return:
            An InferenceConfig object holding the inference configuration.
        """
        batch_size = get_config_attr(
            "batch_size",
            int,
            config_dict,
            "inference config",
            default=8
        )

        tile_size = get_config_attr(
            "tile_size",
            None,
            config_dict,
            "inference config",
            default=None
        )
        if tile_size is not None:
            if isinstance(tile_size, int):
                tile_size = (tile_size, tile_size)
            else:
                tile_size = tuple(tile_size)

        spatial_overlap = get_config_attr(
            "spatial_overlap",
            None,
            config_dict,
            "inference config",
            default=None
        )

        temporal_overlap = get_config_attr(
            "temporal_overlap",
            None,
            config_dict,
            "inference config",
            default=None
        )

        input_loader = get_config_attr(
            "input_loader",
            None,
            config_dict,
            "inference config",
            default=None
        )
        input_loader_args = get_config_attr(
            "input_loader_args",
            None,
            config_dict,
            "inference config",
            default=None
        )

        retrieval_output_dct = config_dict.get("retrieval_output", {})
        retrieval_output = {}
        for model_output, outputs in retrieval_output_dct.items():

            if model_output not in output_cfg:
                raise ValueError(
                    f"Found output name '{model_output}' in inference config but model has no "
                    f"corresponding output."
                )

            outputs = {
                output_name: RetrievalOutputConfig.parse(
                    f"{model_output}.{output_name}",
                    output_cfg[model_output],
                    cfg_dict,
                ) for
                output_name, cfg_dict in outputs.items()
            }
            retrieval_output[model_output] = outputs

        return InferenceConfig(
            batch_size=batch_size,
            tile_size=tile_size,
            spatial_overlap=spatial_overlap,
            temporal_overlap=temporal_overlap,
            input_loader=input_loader,
            input_loader_args=input_loader_args,
            retrieval_output=retrieval_output
        )


    def to_dict(self) -> Dict[str, Any]:
        """
        Convert InferenceConfig to dict for serialization.
        """
        dct = asdict(self)
        retrieval_output = {}
        for name, target_outputs in self.retrieval_output.items():
            retrieval_output[name] = {
                name: output.to_dict() for name, output in target_outputs.items()
            }
        dct["retrieval_output"] = retrieval_output
        return dct



def process(
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        inference_config: InferenceConfig,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32
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
                    results[key] = tensor
                else:
                    for output_name, output in retrieval_output.items():
                        results[output_name] = output.output.compute(tensor)
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
                    results[output_name] = output.output.compute(preds)
            else:
                results["retrieved"] = preds
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
            dtype: str = "float32"
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

            res = process(self.model, batch, self.inference_config, device=self.device, dtype=self.dtype)

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
        input_loader: Any,
        inference_config: InferenceConfig,
        output_path: Optional[Path] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32
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

    Return:
        If an output path is provided, a list of the output files that were written is returned.
        If no output path is provided, the retrieval results are returned as a list of xarray.Datasets.
    """
    outputs = []
    arg_stack = []
    cntr = 1

    model = model.eval()
    if isinstance(model, RetrievalModel):
        output_cfg = model.output_config
    else:
        output_cfg = None


    with Progress() as progress:

        task = progress.add_task("Processing input:", total=len(input_loader))

        input_iterator = iter(input_loader)

        while True:

            try:
                input_data = next(input_iterator)
                args = []
                if isinstance(input_data, tuple):
                    input_data, *args = input_data
            except StopIteration:
                break
            except Exception:
                LOGGER.exception(
                    "Encountered and error when iterating over input samples."
                )
                continue


            tile_size = inference_config.tile_size
            overlap = inference_config.spatial_overlap
            batch_size = inference_config.batch_size

            processor = BatchProcessor(
                model,
                batch_size=batch_size,
                inference_config=inference_config,
                device=device,
                dtype=dtype
            )

            if tile_size is not None:
                if overlap is None:
                    overlap = tile_size // 8

                tiler = Tiler(input_data, tile_size=tile_size, overlap=overlap)

                results_tiled = np.array([[None] * tiler.N] * tiler.M)
                tile_stack = []

                for row_ind in range(tiler.M):
                    for col_ind in range(tiler.N):
                        results_s = processor.process(tiler.get_tile(row_ind, col_ind))
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

                if hasattr(input_loader, "finalize_results"):
                    results = input_loader.finalize_results(results_ass, *args)
                else:
                    results = {}
                    for key, tensor in results_ass.items():
                        dims = get_dimensions(key, output_cfg, inference_config)
                        if dims is None:
                            dims = tuple(
                                [f"{key}_dim_{ind}" for ind in range(tensor.ndim - 2)]
                            )
                        if len(dims) < tensor.ndim:
                            tensor = torch.squeeze(tensor)

                        results[key] = (tuple(dims) + ("x", "y"), tensor)
                    results = xr.Dataset(results)

                filename = f"results_{cntr}.nc"

                if results is not None:
                    if isinstance(results, tuple):
                        results, filename = results
                    if output_path is not None:
                        results.to_netcdf(output_path / filename)
                        outputs.append(output_path / filename)
                    else:
                        outputs.append(results)

            else:
                arg_stack.append(args)
                results_stack = processor.process(input_data)
                results_stack += processor.process(None)
                for results in results_stack:
                    args, *arg_stack = arg_stack

                    results = xr.Dataset({
                        key: (("samples",), tensor.numpy().flatten()) for key, tensor in results.items()
                    })
                    filename = f"results_{cntr}.nc"
                    if hasattr(input_loader, "finalize_results"):
                        results = input_loader.finalize_results(results, *args)
                    if results is not None:
                        if isinstance(results, tuple):
                            results, filename = results
                        if output_path is not None:
                            results.to_netcdf(output_path / filename)
                            outputs.append(output_path / filename)
                        else:
                            outputs.append(results)

            cntr += 1
            progress.update(task, advance=1.0)

    return outputs


@click.argument("model", type=str,)
@click.argument("input_loader", type=str)
@click.argument("input_path", type=str)
@click.option(
    "--input_loader_args",
    type=str,
    help=(
        "A string specifying a Python dictionary that will be passed as additional "
        "arguments to the __init__ call of the input loader."
    )
)
@click.option(
    "--output_path",
    type=str,
    default=None,
    help=(
        "An optional destination to which to write the inference results."
    )
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help=(
        "The device on which to perform inference."
    )
)
@click.option(
    "--dtype",
    type=str,
    default="float32",
    help=(
        "The floating point type to use for inference."
    )
)
def cli(
        model: str,
        input_loader: Any,
        input_path: Path,
        input_loader_args: Dict[str, Any],
        output_path: Optional[Path] = None,
        device: str = "cpu",
        dtype: str = "float32"
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
            model
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
            input_loader_module
        )
        return 1

    if input_loader_args is not None:
        try:
            input_loader_args = eval(input_loader_args)
        except Exception:
            LOGGER.error(
                "Encountered an error when trying to parse the 'input_loader_args' dict."
            )
            return 1
    else:
        input_loader_args = {}

    try:
        input_loader = getattr(module, input_loader_parts[-1])(input_path, **input_loader_args)
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

    run_inference(
        model,
        input_loader,
        InferenceConfig(tile_size=(128, 64), spatial_overlap=16),
        output_path,
        device=device,
        dtype=dtype,
    )
