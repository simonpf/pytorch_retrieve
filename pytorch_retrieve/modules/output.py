"""
pytorch_retrieve.modules.output
===============================

"""
import torch
from torch import nn

from pytorch_retrieve.tensors import MeanTensor, QuantileTensor



class Mean(nn.Module):
    """
    This output module transforms a tensor into an output tensor representing
    the conditional mean of a prediction.
    """

    def forward(self, x):
        return MeanTensor(x)

class Quantiles(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        return QuantileTensor(x, tau=self.tau)
