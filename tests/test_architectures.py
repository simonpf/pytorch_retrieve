"""
Tests for the pytorch_retrieve.architectures module.
"""
import torch

from pytorch_retrieve.architectures import compile_preset


def test_compile_preset():
    """
    Test compiling a ResNet model and ensure that a tensor can be propagated
    through it.
    """
    input_configs = {"x": {"n_features": 1}}
    output_configs = {"y": {"kind": "Mean", "shape": 1}}
    resnet = compile_preset("EncoderDecoder", "resnet_s", input_configs, output_configs)

    x = {"x": torch.rand(1, 1, 128, 128)}
    y = resnet(x)
    assert y["y"].shape == (1, 1, 128, 128)
