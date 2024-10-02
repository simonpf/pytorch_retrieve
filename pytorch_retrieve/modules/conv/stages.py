"""
pytorch_retrieve.modules.conv.stages
====================================

Stage architectures for convolutional neural networks.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn

class SequentialStage:
    """
    A stage consisting of a simple sequence of blocks.
    """

    def __init__(
        self,
        block_factory,
        block_args: Optional[List[object]] = None,
        block_kwargs: Optional[Dict[str, object]] = None,
    ):
        """
        Args:
            block_factory: Factory for generating the convolution block
                making up the stage.
            block_args: Optional list of object that will be passed as
                positional arguments to the block factory.
            block_kwargs: Optional dict mapping parameter names to objects
                that will be passed as additional keyword arguments to
                the block factory.
        """
        self.block_factory = block_factory
        if block_args is None:
            block_args = []
        self.block_args = block_args
        if block_kwargs is None:
            block_kwargs = {}
        self.block_kwargs = block_kwargs

    def __call__(
        self,
        channels_in: int,
        channels_out: int,
        n_blocks: int,
        downsample: Optional[Union[Tuple[int], int]] = None,
        scale: Optional[int] = None,
    ) -> nn.Module:
        """
        Args:
            channels_in: The number of channels in the input to the
                first block.
            channels_out: The number of channels in the input to
                all other blocks an the output from the last block.
            n_blocks: The number of blocks in the stage.
            downsample: Optional scalar of tuple of scalars specifying the
                factors by which to downsample the inputs tensors along the
                height and width dimensions.
        """
        blocks = []
        for block_ind in range(n_blocks):
            blocks.append(
                self.block_factory(
                    channels_in,
                    channels_out,
                    *self.block_args,
                    downsample=downsample,
                    block_index=block_ind,
                    **self.block_kwargs,
                )
            )
            channels_in = channels_out
            downsample = None

        if len(blocks) == 1:
            return blocks[0]

        return nn.Sequential(*blocks)


class SequentialWKeywords(nn.Module):
    """
    A custom sequential layer class that support passing keywors arguments to all
    layers.
    """
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class SequentialWKeywordsStage:
    """
    A stage consisting of a simple sequence of blocks.
    """

    def __init__(
        self,
        block_factory,
        block_args: Optional[List[object]] = None,
        block_kwargs: Optional[Dict[str, object]] = None,
    ):
        """
        Args:
            block_factory: Factory for generating the convolution block
                making up the stage.
            block_args: Optional list of object that will be passed as
                positional arguments to the block factory.
            block_kwargs: Optional dict mapping parameter names to objects
                that will be passed as additional keyword arguments to
                the block factory.
        """
        self.block_factory = block_factory
        if block_args is None:
            block_args = []
        self.block_args = block_args
        if block_kwargs is None:
            block_kwargs = {}
        self.block_kwargs = block_kwargs

    def __call__(
        self,
        channels_in: int,
        channels_out: int,
        n_blocks: int,
        downsample: Optional[Union[Tuple[int], int]] = None,
        scale: Optional[int] = None,
    ) -> nn.Module:
        """
        Args:
            channels_in: The number of channels in the input to the
                first block.
            channels_out: The number of channels in the input to
                all other blocks an the output from the last block.
            n_blocks: The number of blocks in the stage.
            downsample: Optional scalar of tuple of scalars specifying the
                factors by which to downsample the inputs tensors along the
                height and width dimensions.
        """
        blocks = []
        for block_ind in range(n_blocks):
            blocks.append(
                self.block_factory(
                    channels_in,
                    channels_out,
                    *self.block_args,
                    downsample=downsample,
                    block_index=block_ind,
                    **self.block_kwargs,
                )
            )
            channels_in = channels_out
            downsample = None

        if len(blocks) == 1:
            return blocks[0]

        return SequentialWKeywords(*blocks)
