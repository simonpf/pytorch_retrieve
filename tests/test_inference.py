"""
Tests for the pytorch_retrieve.inference module.
"""
from conftest import MODEL_CONFIG_MLP

import numpy as np
from scipy.stats import norm
import pytest
import toml
import torch
from torch import nn


from pytorch_retrieve.config import OutputConfig, read_config_file
from pytorch_retrieve.architectures import compile_architecture

from pytorch_retrieve.inference import (
    BatchProcessor,
    batch_size_rec,
    cat_n_rec,
    to_rec,
    InferenceConfig,
    run_inference,
    InferenceRunner,
    SequentialInferenceRunner,
)
from pytorch_retrieve.tensors import QuantileTensor
from pytorch_retrieve.modules.output import Quantiles


def test_batch_size_rec():
    """
    Assert that calculation of batch size works for arbitrary collections
    of tensors.
    """
    t = torch.rand(4, 10, 10, 10)

    assert batch_size_rec(t) == 4
    assert batch_size_rec([(t,)]) == 4
    assert batch_size_rec({"x": [(t,)]}) == 4


def test_cat_n_rec():
    """
    Assert that calculation of batch size works for arbitrary collections
    of tensors.
    """
    x = torch.rand(6, 10, 10)
    y = torch.rand(6, 10, 10)

    (x_c, y_c), (x_r, y_r) = cat_n_rec([(x, y), (x, y)], 4)
    assert x_c.shape[0] == 4
    assert x_c.shape[0] == 4
    assert y_r.shape[0] == 8
    assert y_r.shape[0] == 8

    x = {
        "x_1": [torch.rand(6, 10, 10) for _ in range(4)],
        "x_2": [torch.rand(6, 10, 10) for _ in range(4)],
    }
    y = [torch.rand(6, 10, 10) for _ in range(4)]

    (x_c, y_c), (x_r, y_r) = cat_n_rec([(x, y), (x, y)], 4)

    for ind in range(4):
        x_c["x_1"][0].shape[0] == 4
        x_c["x_2"][0].shape[0] == 4
        x_r["x_1"][0].shape[0] == 8
        x_r["x_2"][0].shape[0] == 8


def test_to():
    """
    Assert that calculation of batch size works for arbitrary collections
    of tensors.
    """
    t = torch.rand(4, 10, 10, 10)

    assert to_rec(t, dtype=torch.int32).dtype == torch.int32
    assert to_rec([(t,)], dtype=torch.int32)[0][0].dtype == torch.int32
    assert to_rec({"x": [(t,)]}, dtype=torch.int32)["x"][0][0].dtype == torch.int32


def test_batch_processor(tmp_path):
    """
    Test batched processing of inputs.
    """
    model = nn.Identity()

    x = [
        torch.arange(10).reshape((10, 1)).to(torch.float32),
        torch.arange(10, 20).reshape((10, 1)).to(torch.float32),
        torch.arange(20, 30).reshape((10, 1)).to(torch.float32),
    ]

    processor = BatchProcessor(model=model, batch_size=4, device="cpu")

    results = []
    for inpt in x:
        results.append(processor.process(inpt))
    results.append(processor.process(None))

    assert len(results) == 4
    for ind in range(3):
        assert torch.isclose(x[ind], results[ind + 1][0]["retrieved"]).all()


def test_batch_processor_irregular(tmp_path):
    """
    Test batched processing of inputs with irregular batch sizes.
    """
    model = nn.Identity()

    x = [
        torch.arange(3).reshape((3, 1, 1, 1)).to(torch.float32),
        torch.arange(3, 10).reshape((7, 1, 1, 1)).to(torch.float32),
        torch.arange(10, 30).reshape((20, 1, 1, 1)).to(torch.float32),
    ]

    processor = BatchProcessor(model=model, batch_size=4, device="cpu")

    results = []
    for inpt in x:
        results_step = processor.process(inpt)
        for res in results_step:
            results.append(res["retrieved"])

    results_step = processor.process(None)
    for res in results_step:
        results.append(res["retrieved"])

    x = torch.cat(x, 0)
    results = torch.cat(results, 0)

    assert torch.isclose(x, results).all()


class TabularInputLoader:
    """
    An input loader that loads data that will not be tiled.
    """

    def __init__(self, n_samples: int, n_features: int):
        self.data = np.broadcast_to(
            np.arange(n_samples).reshape(-1, 1), (n_samples, n_features)
        )

    def __len__(self) -> int:
        return 8

    def __getitem__(self, ind):
        batch_size = self.data.shape[0] // len(self)
        start = ind * batch_size
        end = start + batch_size
        return torch.tensor(self.data[start:end])

    def __iter__(self):
        start = 0
        for ind in range(len(self)):
            end = start + (self.data.shape[0] // len(self))
            yield torch.tensor(self.data[start:end])
            start = end


INFERENCE_CONFIG = """
batch_size = 256
"""



def test_run_inference_tabular(tmp_path):
    """
    Test running inference with tabular data.
    """
    inference_config = toml.loads(INFERENCE_CONFIG)
    output_config = {}
    inference_config = InferenceConfig.parse(output_config, inference_config)
    model = nn.Identity()
    input_loader = TabularInputLoader(2048, 12)

    runner = InferenceRunner(model, input_loader, inference_config=inference_config)

    results = runner.run(output_path=tmp_path, device="cpu", dtype=torch.float32)

    assert len(results) == 8
    for res in results:
        assert res.exists()


def test_run_inference_tabular_sequential(tmp_path):
    """
    Test running inference with tabular data.
    """
    inference_config = toml.loads(INFERENCE_CONFIG)
    output_config = {}
    inference_config = InferenceConfig.parse(output_config, inference_config)
    model = nn.Identity()
    input_loader = TabularInputLoader(2048, 12)

    runner = SequentialInferenceRunner(model, input_loader, inference_config=inference_config)

    results = runner.run(output_path=tmp_path, device="cpu", dtype=torch.float32)

    assert len(results) == 8
    for res in results:
        assert res.exists()


def test_run_inference_simple_input(tmp_path):
    """
    Test running inference with tabular data.
    """
    inference_config = toml.loads(INFERENCE_CONFIG)
    output_config = {}
    inference_config = InferenceConfig.parse(output_config, inference_config)
    model = nn.Identity()

    input_loader = TabularInputLoader(2048, 12)
    input_list = [inpt for inpt in input_loader]

    results = run_inference(
        model,
        input_list,
        inference_config=inference_config,
        output_path=tmp_path
    )

    assert len(results) == 8
    for res in results:
        assert res.exists()

    results = run_inference(
        model,
        input_list[0],
        inference_config=inference_config,
        output_path=tmp_path
    )

    assert len(results) == 1
    for res in results:
        assert res.exists()



def test_run_inference_multiple_input_loaders(tmp_path):
    """
    Test running inference with multiple input loaders.
    """
    inference_config = toml.loads(INFERENCE_CONFIG)
    output_config = {}
    inference_config = InferenceConfig.parse(output_config, inference_config)
    model = nn.Identity()
    input_loader = TabularInputLoader(2048, 12)

    runner = InferenceRunner(
        model, input_loader, inference_config=inference_config, n_input_loaders=4
    )

    results = runner.run(output_path=tmp_path, device="cpu", dtype=torch.float32)

    assert len(results) == 8
    for res in results:
        assert res.exists()


class QuantileTensorLoader:
    """
    An input data loader that loads a tensor containing quantile of a
    normal distribution of a given size. To be used in conjunction with a
    identiy retrieval to test the calculation of retrieval outputs.
    """

    def __init__(self, n_inputs: int, n_rows: int, n_cols: int):
        self.n_inputs = n_inputs
        self.n_rows = n_rows
        self.n_cols = n_cols

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        quantiles = np.linspace(0, 1, 33)[1:-1]
        tensor = torch.tensor(norm.ppf(quantiles)).to(torch.float32)[..., None, None]
        tensor = tensor.repeat((1, self.n_rows, self.n_cols))
        offsets = torch.arange(self.n_rows)[None, :, None]
        tensor = QuantileTensor(tensor + offsets, tau=quantiles)
        for ind in range(self.n_inputs):
            yield tensor[None]


QUANTILE_INFERENCE_CONFIG = """
tile_size = 128
spatial_overlap = 32
batch_size = 2

[retrieval_output.surface_precip]
surface_precip_terciles = {retrieval_output="Quantiles", tau=[0.33, 0.67]}
surface_precip_mean = "ExpectedValue"
pop = {retrieval_output="ExceedanceProbability", threshold=0}
"""


def test_run_inference_quantiles():
    """
    Test running inference with several derived outputs.
    """
    inference_config = toml.loads(QUANTILE_INFERENCE_CONFIG)
    output_config = {
        "surface_precip": OutputConfig("surface_precip", kind="Quantiles", shape=(32,))
    }
    inference_config = InferenceConfig.parse(output_config, inference_config)
    model = Quantiles("output", 1, np.linspace(0, 1, 33)[1:-1])
    input_loader = QuantileTensorLoader(4, 234, 453)

    runner = InferenceRunner(model, input_loader, inference_config=inference_config)

    results = runner.run(output_path=None, device="cpu", dtype=torch.float32)

    assert len(results) == 4
    assert "surface_precip_terciles" in results
    assert "surface_precip_mean" in results
    assert "pop" in results

    assert np.isclose(results[0]["pop"].data[0], 0.5).all()
    assert np.isclose(
        results[0]["surface_precip_mean"].data[-1], input_loader.n_rows - 1, atol=1e-3
    ).all()



class TiledDataLoader:
    """
    This data loader loads spatial data and an additonal mask that should not be tiled.
    """

    def __init__(self, n_inputs: int, n_rows: int, n_cols: int):
        self.n_inputs = n_inputs
        self.n_rows = n_rows
        self.n_cols = n_cols

    def __len__(self) -> int:
        return self.n_inputs

    def __iter__(self):
        quantiles = np.linspace(0, 1, 33)[1:-1]
        tensor = norm.ppf(quantiles).astype(np.float32)[..., None, None]
        tensor = np.broadcast_to(tensor, (31, self.n_rows, self.n_cols))
        offsets = np.arange(self.n_rows)[None, :, None]
        tensor = torch.tensor(tensor + offsets)#QuantileTensor(torch.tensor(tensor + offsets), tau=quantiles)
        cntr = 0
        for ind in range(self.n_inputs):
            cntr += 1
            yield {
                "input": tensor[None],
                "mask": torch.isfinite(tensor[None]).any(-1).any(-1)
            }


TILED_INFERENCE_CONFIG = """
tile_size = 128
spatial_overlap = 32
batch_size = 2
exclude_from_tiling = ["mask"]

[retrieval_output.surface_precip]
surface_precip_terciles = {retrieval_output="Quantiles", tau=[0.33, 0.67]}
surface_precip_mean = "ExpectedValue"
pop = {retrieval_output="ExceedanceProbability", threshold=0}
"""

class MaskedQuantileOutput(Quantiles):
    """
    Specialized quantile output class that can handle input dicts.
    """
    def forward(self, inpts):
        return super().forward(inpts["input"])

@pytest.mark.skip(reason="Tensor creation causes deadlock.")
def test_run_inference_tiled():
    """
    This tests running inference on tiled input including inputs. In particular, it ensures
    that input that should be exluded from tiling are excluded.
    """
    inference_config = toml.loads(TILED_INFERENCE_CONFIG)
    output_config = {
        "surface_precip": OutputConfig("surface_precip", kind="Quantiles", shape=(32,))
    }
    inference_config = InferenceConfig.parse(output_config, inference_config)
    model = MaskedQuantileOutput("output", 1, np.linspace(0, 1, 33)[1:-1])
    input_loader = TiledDataLoader(4, 234, 453)

    runner = InferenceRunner(model, input_loader, inference_config=inference_config)

    results = runner.run(output_path=None, device="cpu", dtype=torch.float32)

    assert len(results) == 4
    assert "surface_precip_terciles" in results
    assert "surface_precip_mean" in results
    assert "pop" in results

    assert np.isclose(results[0]["pop"].data[0], 0.5).all()
    assert np.isclose(
        results[0]["surface_precip_mean"].data[-1], input_loader.n_rows - 1, atol=1e-3
    ).all()


def test_run_inference_tiled_sequential(tmp_path):
    """
    Test running inference with tabular data.
    """
    inference_config = toml.loads(TILED_INFERENCE_CONFIG)
    output_config = {
        "surface_precip": OutputConfig("surface_precip", kind="Quantiles", shape=(32,))
    }
    inference_config = InferenceConfig.parse(output_config, inference_config)
    model = MaskedQuantileOutput("output", 1, np.linspace(0, 1, 33)[1:-1])
    input_loader = TiledDataLoader(4, 234, 453)

    runner = SequentialInferenceRunner(model, input_loader, inference_config=inference_config)

    results = runner.run(output_path=None, device="cpu", dtype=torch.float32)

    assert len(results) == 4
    assert "surface_precip_terciles" in results
    assert "surface_precip_mean" in results
    assert "pop" in results

    assert np.isclose(results[0]["pop"].data[0], 0.5).all()
    assert np.isclose(
        results[0]["surface_precip_mean"].data[-1], input_loader.n_rows - 1, atol=1e-3
    ).all()



MLP_INFERENCE_CFG = """
[architecture]
name = "MLP"

[architecture.body]
hidden_channels = 128
n_layers = 4
activation_factory = "GELU"
normalization_factory = "LayerNorm"

[input.x]
n_features = 1

[output.y]
shape = [1,]
kind = "Mean"

[inference]
batch_size = 2048

[inference.retrieval_output.y]
surface_precip_terciles = {retrieval_output="Quantiles", tau=[0.33, 0.67]}
surface_precip_mean = "ExpectedValue"
pop = {retrieval_output="ExceedanceProbability", threshold=0}
"""


@pytest.fixture
def mlp_inference_config(tmp_path):
    """
    A config file defining a MLP retrieval model with inference config.
    """
    cfg_path = tmp_path / "model.toml"

    with open(cfg_path, "w") as output:
        output.write(MLP_INFERENCE_CFG)
    return cfg_path


def test_mlp_inference_config(mlp_inference_config):
    """
    Instantiate the MLP model with inference config.
    """
    cfg_dict = read_config_file(mlp_inference_config)
    model = compile_architecture(cfg_dict)
    inference_config = model.inference_config
    assert inference_config is not None
    assert inference_config.batch_size == 2048
    assert len(inference_config.retrieval_output["y"]) == 3
