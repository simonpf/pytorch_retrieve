"""
Tests for preset architectures.
"""
import numpy as np
import toml
import torch


from pytorch_retrieve.architectures import compile_architecture


def test_unet():
    """
    Test the UNet architecture preset.
    """
    config_dict = toml.loads(
        """
        [input.x]
        n_features = 1

        [output.y]
        kind = "Mean"
        shape = 1

        [architecture]
        name = "EncoderDecoder"
        preset = "UNet"
        """
    )

    model = compile_architecture(config_dict)
    x = {"x": torch.rand(1, 1, 256, 256)}
    y = model(x)
    assert y["y"].shape == (1, 1, 256, 256)


def test_resnet_s():
    """
    Test the small ResNet encoder-decoder architecture.
    """
    config_dict = toml.loads(
        """
        [input.x]
        n_features = 1

        [output.y]
        kind = "Mean"
        shape = 1

        [architecture]
        name = "EncoderDecoder"
        preset = "resnet_s"
        """
    )

    model = compile_architecture(config_dict)
    x = {"x": torch.rand(1, 1, 256, 256)}
    y = model(x)
    assert y["y"].shape == (1, 1, 256, 256)


def test_resnext_s():
    """
    Test the small ResNeXt encoder-decoder architecture.
    """
    config_dict = toml.loads(
        """
        [input.x]
        n_features = 1

        [output.y]
        kind = "Mean"
        shape = 1

        [architecture]
        name = "EncoderDecoder"
        preset = "resnext_s"
        """
    )

    model = compile_architecture(config_dict)
    x = {"x": torch.rand(1, 1, 256, 256)}
    y = model(x)
    assert y["y"].shape == (1, 1, 256, 256)


def test_metnet_t():
    """
    Test the tiny MetNet architecture.
    """
    config_dict = toml.loads(
        """
        [input.x]
        n_features = 1

        [output.y]
        kind = "Mean"
        shape = 1

        [architecture]
        name = "MetNet"
        preset = "metnet_t"
        """
    )

    model = compile_architecture(config_dict)
    x = {
        "x": [torch.rand(1, 1, 256, 256) for i in range(4)],
        "lead_times": torch.tensor([np.arange(1.0, 5.0).astype(np.float32) * 30.0])
    }
    y = model(x)
    assert isinstance(y["y"], list)
    assert len(y["y"]) == 4
