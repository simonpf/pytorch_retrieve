"""
Tests for preset architectures.
"""
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
        preset = "unet"
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
