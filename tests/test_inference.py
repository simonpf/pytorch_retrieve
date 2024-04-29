"""
Tests for the pytorch_retrieve.inference module.
"""
import toml
import torch
from torch import nn

from pytorch_retrieve.inference import (
    BatchProcessor,
    batch_size_rec,
    cat_n_rec,
    to_rec,
    InferenceConfig
)


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

    processor = BatchProcessor(
        model=model,
        batch_size=4,
    )

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

    processor = BatchProcessor(
        model=model,
        batch_size=4,
    )

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


INFERENCE_CONFIG = """
batch_size = 12
tile_size = [32, 32]
spatial_overlap = 8
temporal_overlap = 0

[output.surface_precip]
surface_precip_mean = "ExpectedValue"
surface_precip_median = {quantity="Quantiles", tau=[0.5]}
surface_precip_heavy = {quantity="ExceedanceProbability", threshold=10.0}
"""

def test_inference_config():
    """
    Test parsing of inference configs.
    """
    inference_config = toml.loads(INFERENCE_CONFIG)
    inference_config = InferenceConfig.parse(inference_config)

    assert inference_config.batch_size == 12
    assert inference_config.tile_size == (32, 32)
    assert inference_config.spatial_overlap == 8
    assert inference_config.temporal_overlap == 0

    qty = inference_config.output["surface_precip"]["surface_precip_mean"]
    assert qty.quantity == "ExpectedValue"

    ic_saved = toml.dumps(inference_config.to_dict())
    inference_config_loaded = toml.loads(ic_saved)
    inference_config_loaded = InferenceConfig.parse(inference_config_loaded)

    assert inference_config_loaded.batch_size == 12
    assert inference_config_loaded.tile_size == (32, 32)
    assert inference_config_loaded.spatial_overlap == 8
    assert inference_config_loaded.temporal_overlap == 0

    qty = inference_config_loaded.output["surface_precip"]["surface_precip_mean"]
    assert qty.quantity == "ExpectedValue"
