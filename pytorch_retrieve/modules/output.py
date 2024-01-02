"""
pytorch_retrieve.modules.output
===============================

"""
from torch import nn

from pytorch_retrieve.tensors import MeanTensor


class Mean(nn.Module):
    """
    This output module transforms a tensor into an output tensor representing
    the conditional mean of a prediction.
    """

    def forward(self, x):
        return MeanTensor(x)
