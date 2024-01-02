"""
pytorch_retrieve.metrics
========================

Provides TorchMetrics metrics modules that handle pytorch_retrieve's
probabilistic outputs.
"""
import torch
import torchmetrics as tm

from pytorch_retrieve.tensors import MaskedTensor


class ScalarMetric:
    """
    Dummy class to identify metrics that evaluate scalar predictions,
    i.e., predictions that consists of a single value.
    """


class Bias(ScalarMetric, tm.Metric):
    """
    The mean error.
    """

    name = "Bias"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("error", default=torch.tensor(0.0))
        self.add_state("counts", default=torch.tensor(0.0))

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
        """
        pred = pred.squeeze()
        target = target.squeeze()

        if isinstance(target, MaskedTensor):
            mask = target.mask
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
            pred = pred[~mask]
            target = target[~mask]

        self.error += (pred - target).sum()
        self.counts += pred.numel()

    def compute(self) -> torch.Tensor:
        """
        Compute the mean error.

        Return:
            The bias over all data.
        """
        return self.error / self.counts


class CorrelationCoef(ScalarMetric, tm.regression.PearsonCorrCoef):
    """
    Linear correlation coefficient.
    """

    name = "CorrelationCoef"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
        """
        pred = pred.squeeze()
        target = target.squeeze()

        if isinstance(target, MaskedTensor):
            mask = target.mask
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
            pred = pred[~mask]
            target = target[~mask]

        if pred.dim() > 2:
            pred = pred.flatten()
            target = target.flatten()
        super().update(pred, target)


class MSE(ScalarMetric, tm.regression.MeanSquaredError):
    """
    The mean-squared error.
    """

    name = "MSE"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
        """
        pred = pred.squeeze()
        target = target.squeeze()

        if isinstance(target, MaskedTensor):
            mask = target.mask
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
            pred = pred[~mask]
            target = target[~mask]

        if pred.dim() > 2:
            pred = pred.flatten()
            target = target.flatten()
        super().update(pred, target)
