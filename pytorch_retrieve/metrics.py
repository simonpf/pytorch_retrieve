"""
pytorch_retrieve.metrics
========================

Provides TorchMetrics metrics modules that handle pytorch_retrieve's
probabilistic outputs.
"""
import io
import logging
from typing import Any, Dict, Optional, Tuple, Union

from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch import loggers
import numpy as np
import torch
import torchmetrics as tm
from torchvision.utils import make_grid

from pytorch_retrieve.tensors import (
    MaskedTensor,
    RegressionTensor,
    ClassificationTensor,
    DetectionTensor,
)


LOGGER = logging.getLogger(__name__)


BinSpec = Union[int, Tuple[float, float, int]]

class MetricBase:
    """
    Base class for metrics providing shared functionality for logging and the calculation of buffer
    shapes.
    """
    def __init__(self, conditional: Optional[Dict[str, BinSpec]] = None):
        conds = []
        bins = []
        shape = []
        if conditional is None:
            self.conditional = None
        else:
            for name, bin_cfg in conditional.items():
                conds.append(name)
                if isinstance(bin_cfg, int):
                    bins.append(torch.arange(bin_cfg + 1) - 0.5)
                    shape.append(bin_cfg)
                elif isinstance(bin_cfg, (tuple, list)):
                    start, end, n_bins = bin_cfg
                    n_bins = int(n_bins)
                    bins.append(torch.linspace(start, end, n_bins + 1))
                    shape.append(n_bins)
                else:
                    raise RuntimeError(
                        f"Metric received unsupported value of type "
                        f" {type(bin_cfg)} in 'conditional'. Expected a single "
                        " integer or tuple '(start, end, n_bins)'."
                    )
            self.conditional = conds
        self.bins = bins
        self.shape = tuple(shape)

    def log(
        self, lightning_module: LightningModule, output_name: Optional[str] = None
    ) -> None:
        """
        Log metric results.

        Args:
            lightning_module: The lightning module with which the training is performed.
            output_name: The name of the output the accuracy of which this metric measured.
        """
        value = self.compute()
        name = self.name
        if output_name is not None:
            name = f"{name} ({output_name})"

        sync_dist = lightning_module.device != torch.device("cpu")

        # Log multiple scalars for conditional norms.
        if self.conditional is not None and len(self.bins) == 1:
            cntrs = self.bins[0]
            cntrs = 0.5 * (cntrs[1:] + cntrs[:-1])
            vals = {
                str(key): val for key, val in zip(cntrs, value)
            }
            lightning_module.logger.experiment.add_scalars(name, vals, global_step=lightning_module.global_step)
            return None

        lightning_module.log(
            name, value, on_step=False, on_epoch=True, sync_dist=sync_dist
        )


class ScalarMetric(MetricBase):
    """
    Helper class to identify metrics that evaluate deterministic scalar predictions,
    i.e., predictions that consists of a single value.
    """

class CategoricalDetectionMetric(MetricBase):
    """
    Helper class to identify metrics that evaluate deterministic detection tasks.
    """

class ProbabilisticDetectionMetric(MetricBase):
    """
    Helper class to identify metrics that evaluate probabilistic detection tasks.
    """


class Bias(ScalarMetric, tm.Metric):
    """
    The mean error.
    """

    name = "Bias"

    def __init__(self, conditional: Optional[Dict[str, BinSpec]] = None, **kwargs):
        ScalarMetric.__init__(self, conditional)
        tm.Metric.__init__(self, **kwargs)
        error = torch.zeros(self.shape)
        counts = torch.zeros(self.shape)
        self.add_state("error", default=error, dist_reduce_fx="sum")
        self.add_state("counts", default=counts, dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        conditional: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            weights: Optional weight tensor used to weigh the contribution of each
                sample to the overall bias.
            conditional: An optional dictionary containing values to
                condition the calculation of the bias onto.
        """
        pred = pred.squeeze()
        target = target.squeeze()
        if weights is None:
            weights = torch.ones_like(target)
        else:
            weights = weights.squeeze()

        mask = None
        if isinstance(target, MaskedTensor):
            mask = target.mask
            target = target.base
            if isinstance(weights, MaskedTensor):
                weights = weights.base
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
                pred = pred.base
            pred = pred[~mask]
            target = target[~mask]
            weights = weights[~mask]

        if self.conditional is None:
            self.error += ((pred - target) * weights).sum()
            self.counts += weights.sum()
            return None

        device = torch.device("cpu")
        dtype = pred.dtype
        self.to(device=device, dtype=dtype)

        coords = []
        for cond in self.conditional:

            coords_c = conditional[cond].squeeze()

            if mask is None:
                if coords_c.ndim < target.ndim:
                    coords_c = coords_c.reshape(coords_c.shape + (1,) * (target.ndim - coords_c.ndim))
                    coords_c = torch.broadcast_to(coords_c, target.shape)
                coords.append(coords_c)
            else:
                mask = mask.to(device=device)
                mask_s = mask.squeeze()
                # Expand channel dimension if necessary
                if coords_c.ndim < mask_s.ndim:
                    coords_c = coords_c.reshape(coords_c.shape + (1,) * (mask_s.ndim - coords_c.ndim))
                    coords_c = torch.broadcast_to(coords_c, mask_s.shape)
                coords.append(coords_c[~mask_s])

        coords = torch.stack(coords, -1).to(device=device, dtype=dtype)
        wgts = ((pred - target) * weights).to(device=device, dtype=dtype)
        bins = tuple([bns.to(device=device, dtype=pred.dtype) for bns in self.bins])
        self.error += torch.histogramdd(coords, bins=bins, weight=wgts)[0]
        self.counts += torch.histogramdd(coords, bins=bins, weight=weights.to(device=device, dtype=dtype))[0]


    def compute(self) -> torch.Tensor:
        """
        Compute the mean error.

        Return:
            The bias over all data.
        """
        return self.error / self.counts


class RelativeBias(ScalarMetric, tm.Metric):
    """
    The relative mean error.
    """

    name = "Bias"

    def __init__(self, conditional: Optional[Dict[str, BinSpec]] = None, **kwargs):
        ScalarMetric.__init__(self, conditional)
        tm.Metric.__init__(self, **kwargs)
        error = torch.zeros(self.shape)
        mean = torch.zeros(self.shape)
        counts = torch.zeros(self.shape)
        self.add_state("error", default=error, dist_reduce_fx="sum")
        self.add_state("mean", default=mean, dist_reduce_fx="sum")
        self.add_state("counts", default=counts, dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        conditional: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            weights: Optional weight tensor used to weigh the contribution of each
                sample to the overall bias.
            conditional: An optional dictionary containing values to
                condition the calculation of the bias onto.
        """
        pred = pred.squeeze()
        target = target.squeeze()
        if weights is None:
            weights = torch.ones_like(target)
        else:
            weights = weights.squeeze()

        mask = None
        if isinstance(target, MaskedTensor):
            mask = target.mask
            target = target.base
            if isinstance(weights, MaskedTensor):
                weights = weights.base
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
                pred = pred.base
            pred = pred[~mask]
            target = target[~mask]
            weights = weights[~mask]

        if self.conditional is None:
            self.error += ((pred - target) * weights).sum()
            self.mean += (target * weights).sum()
            self.counts += weights.sum()
            return None

        device = torch.device("cpu")
        self.to(device=device)

        coords = []
        for cond in self.conditional:

            coords_c = conditional[cond].squeeze()

            if mask is None:
                if coords_c.ndim < target.ndim:
                    coords_c = coords_c.reshape(coords_c.shape + (1,) * (target.ndim - coords_c.ndim))
                    coords_c = torch.broadcast_to(coords_c, target.shape)
                coords.append(coords_c)
            else:
                mask = mask.to(device=device)
                mask_s = mask.squeeze()
                # Expand channel dimension if necessary
                if coords_c.ndim < mask_s.ndim:
                    coords_c = coords_c.reshape(coords_c.shape + (1,) * (mask_s.ndim - coords_c.ndim))
                    coords_c = torch.broadcast_to(coords_c, mask_s.shape)
                coords.append(coords_c[~mask_s])

        coords = torch.stack(coords, -1).to(device=device)

        wgts = ((pred - target) * weights).to(device=device)
        bins = tuple([bns.to(device=device, dtype=pred.dtype) for bns in self.bins])
        self.error += torch.histogramdd(coords, bins=bins, weight=wgts)[0]
        wgts = (target * weights).to(device=device)
        self.mean += torch.histogramdd(coords, bins=bins, weight=wgts)[0]
        self.counts += torch.histogramdd(coords, bins=bins, weight=weights.to(device=device))[0]

    def compute(self) -> torch.Tensor:
        """
        Compute the mean error.

        Return:
            The relative bias in percent.
        """
        return 100.0 * (self.error / self.mean)


class CorrelationCoef(ScalarMetric, tm.regression.PearsonCorrCoef):
    """
    Linear correlation coefficient.
    """

    name = "CorrelationCoef"

    def __init__(self, conditional: Optional[Dict[str, BinSpec]] = None):
        ScalarMetric.__init__(self, conditional)
        tm.regression.PearsonCorrCoef.__init__(self)

        error = torch.zeros(self.shape)
        self.add_state("x", default=torch.zeros(self.shape), dist_reduce_fx="sum")
        self.add_state("x2", default=torch.zeros(self.shape), dist_reduce_fx="sum")
        self.add_state("xy", default=torch.zeros(self.shape), dist_reduce_fx="sum")
        self.add_state("y", default=torch.zeros(self.shape), dist_reduce_fx="sum")
        self.add_state("y2", default=torch.zeros(self.shape), dist_reduce_fx="sum")
        self.add_state("counts", default=torch.zeros(self.shape), dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        conditional: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            weights: An optional tensor containing weights to apply to each
                sample.
            conditional: An optional dictionary containing values to
                condition the calculation of the bias onto.
        """
        pred = pred.squeeze()
        target = target.squeeze()
        if weights is None:
            weights = torch.ones_like(target)
        else:
            weights = weights.squeeze()

        mask = None
        if isinstance(target, MaskedTensor):
            mask = target.mask
            target = target.base
            weights = weights.base
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
                pred = pred.base
            pred = pred[~mask]
            target = target[~mask]
            weights = weights[~mask]

        if self.conditional is None:

            if pred.dim() >= 2:
                pred = pred.flatten()
                target = target.flatten()
                weights = weights.flatten()

            self.x += (pred * weights).sum()
            self.x2 += ((pred**2) * weights).sum()
            self.xy += (pred * target * weights).sum()
            self.y += (target * weights).sum()
            self.y2 += ((target**2) * weights).sum()
            self.counts += weights.sum()
        else:
            device = torch.device("cpu")
            dtype = pred.dtype
            self.to(device=device, dtype=dtype)

            coords = []
            for cond in self.conditional:

                coords_c = conditional[cond].squeeze()

                if mask is None:
                    if coords_c.ndim < target.ndim:
                        coords_c = coords_c.reshape(coords_c.shape + (1,) * (target.ndim - coords_c.ndim))
                        coords_c = torch.broadcast_to(coords_c, target.shape)
                    coords.append(coords_c)
                else:
                    mask = mask.to(device=device)
                    mask_s = mask.squeeze()
                    # Expand channel dimension if necessary
                    if coords_c.ndim < mask_s.ndim:
                        coords_c = coords_c.reshape(coords_c.shape + (1,) * (mask_s.ndim - coords_c.ndim))
                        coords_c = torch.broadcast_to(coords_c, mask_s.shape)
                    coords.append(coords_c[~mask_s])

            coords = torch.stack(coords, -1).to(device=device, dtype=dtype)
            bins = tuple([bns.to(device=device, dtype=pred.dtype) for bns in self.bins])
            pred = pred.to(device=device, dtype=dtype)
            target = target.to(device=device, dtype=dtype)
            weights = weights.to(device=device, dtype=dtype)

            self.x += torch.histogramdd(coords, bins=bins, weight=pred * weights)[0]
            self.x2 += torch.histogramdd(
                coords, bins=bins, weight=(pred**2) * weights
            )[0]
            self.xy += torch.histogramdd(
                coords, bins=bins, weight=pred * target * weights
            )[0]
            self.y += torch.histogramdd(coords, bins=bins, weight=target * weights)[0]
            self.y2 += torch.histogramdd(
                coords, bins=bins, weight=(target**2) * weights
            )[0]
            self.counts += torch.histogramdd(coords, bins=bins, weight=weights)[0]

    def compute(self) -> torch.Tensor:
        """
        Calculate the correlation coefficient.
        """
        x_mean = self.x / self.counts
        y_mean = self.y / self.counts
        x_var = self.x2 / self.counts - x_mean**2
        y_var = self.y2 / self.counts - y_mean**2
        corr = (self.xy / self.counts - x_mean * y_mean) / torch.sqrt(x_var * y_var)
        return corr

    def compute(self) -> torch.Tensor:
        """
        Calculate the correlation coefficient.
        """
        x_mean = self.x / self.counts
        y_mean = self.y / self.counts
        x_var = self.x2 / self.counts - x_mean**2
        y_var = self.y2 / self.counts - y_mean**2
        corr = (self.xy / self.counts - x_mean * y_mean) / torch.sqrt(x_var * y_var)
        return corr


class MSEBase(tm.Metric):
    """
    The mean-squared error.
    """
    def __init__(self, conditional: Optional[Dict[str, BinSpec]] = None):
        tm.Metric.__init__(self)
        error = torch.zeros(self.shape)
        self.add_state("error", default=error, dist_reduce_fx="sum")
        counts = torch.zeros(self.shape)
        self.add_state("counts", default=counts, dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        conditional: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            weights: An optional tensor of weights to apply to all validation samples.
            conditional: An optional dictionary containing values to
                condition the calculation of the bias onto.
        """
        pred = pred.squeeze()
        target = target.squeeze()
        if weights is None:
            weights = torch.ones_like(target)
        else:
            weights = weights.squeeze()

        mask = None
        if isinstance(target, MaskedTensor):
            mask = target.mask
            target = target.base
            if isinstance(weights, MaskedTensor):
                weights = weights.base
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
                pred = pred.base
            pred = pred[~mask]
            target = target[~mask]
            weights = weights[~mask]

        if self.conditional is None:

            if pred.dim() > 2:
                pred = pred.flatten()
                target = target.flatten()
                weights = weights.flatten()

            self.error += (((pred - target) ** 2) * weights).sum()
            self.counts += weights.sum()
        else:

            device = torch.device("cpu")
            dtype = pred.dtype

            self.to(device, dtype=dtype)

            coords = []
            for cond in self.conditional:

                coords_c = conditional[cond].squeeze()

                if mask is None:
                    if coords_c.ndim < target.ndim:
                        coords_c = coords_c.reshape(coords_c.shape + (1,) * (target.ndim - coords_c.ndim))
                        coords_c = torch.broadcast_to(coords_c, target.shape)
                    coords.append(coords_c)
                else:
                    mask = mask.to(device=device)
                    mask_s = mask.squeeze()
                    # Expand channel dimension if necessary
                    if coords_c.ndim < mask_s.ndim:
                        coords_c = coords_c.reshape(coords_c.shape + (1,) * (mask_s.ndim - coords_c.ndim))
                        coords_c = torch.broadcast_to(coords_c, mask_s.shape)
                    coords.append(coords_c[~mask_s])

            coords = torch.stack(coords, -1).to(device=device, dtype=dtype)
            bins = tuple([bns.to(device=device, dtype=pred.dtype) for bns in self.bins])
            pred = pred.to(device=device, dtype=dtype)
            target = target.to(device=device, dtype=dtype)
            weights = weights.to(device=device, dtype=dtype)
            self.error += torch.histogramdd(
                coords, bins=bins, weight=((pred - target) ** 2) * weights
            )[0]
            self.counts += torch.histogramdd(coords, bins=bins, weight=weights)[0]

    def compute(self) -> torch.Tensor:
        """
        Calculate the MSE.
        """
        return self.error / self.counts


class MSE(ScalarMetric, MSEBase):
    """
    The mean-squared error.
    """
    name = "MSE"
    def __init__(self, conditional: Optional[Dict[str, BinSpec]] = None):
        ScalarMetric.__init__(self, conditional=conditional)
        MSEBase.__init__(self)


class MAE(ScalarMetric, tm.Metric):
    """
    The mean absolute error.
    """

    name = "MAE"

    def __init__(self, conditional: Optional[Dict[str, BinSpec]] = None):
        ScalarMetric.__init__(self, conditional=conditional)
        tm.Metric.__init__(self)
        error = torch.zeros(self.shape)
        counts = torch.zeros(self.shape)
        self.add_state("error", default=error, dist_reduce_fx="sum")
        self.add_state("counts", default=counts, dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        conditional: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            weights: An optional tensor of weights to apply to all validation samples.
            conditional: An optional dictionary containing values to
                condition the calculation of the bias onto.
        """
        pred = pred.squeeze()
        target = target.squeeze()
        if weights is None:
            weights = torch.ones_like(target)
        else:
            weights = weights.squeeze()

        if isinstance(target, MaskedTensor):
            mask = target.mask
            target = target.base
            if isinstance(weights, MaskedTensor):
                weights = weights.base

            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
                pred = pred.base
            pred = pred[~mask]
            target = target[~mask]
            weights = weights[~mask]

        if self.conditional is None:

            if pred.dim() > 2:
                pred = pred.flatten()
                target = target.flatten()
                weights = weights.flatten()

            self.error += (torch.abs(pred - target) * weights).sum()
            self.counts += weights.sum()
        else:
            device = torch.device("cpu")
            self.to(device)

            coords = []
            for cond in self.conditional:

                coords_c = conditional[cond].squeeze()

                if mask is None:
                    if coords_c.ndim < target.ndim:
                        coords_c = coords_c.reshape(coords_c.shape + (1,) * (target.ndim - coords_c.ndim))
                        coords_c = torch.broadcast_to(coords_c, target.shape)
                    coords.append(coords_c)
                else:
                    mask = mask.to(device=device)
                    mask_s = mask.squeeze()
                    # Expand channel dimension if necessary
                    if coords_c.ndim < mask_s.ndim:
                        coords_c = coords_c.reshape(coords_c.shape + (1,) * (mask_s.ndim - coords_c.ndim))
                        coords_c = torch.broadcast_to(coords_c, mask_s.shape)
                    coords.append(coords_c[~mask_s])

            coords = torch.stack(coords, -1).to(device=device)
            bins = tuple([bns.to(device=device, dtype=pred.dtype) for bns in self.bins])
            pred = pred.to(device=device)
            target = target.to(device=device)
            weights = weights.to(device=device)
            self.error += torch.histogramdd(
                coords, bins=bins, weight=torch.abs(pred - target) * weights
            )[0]
            self.counts += torch.histogramdd(coords, bins=bins, weight=weights)[0]

    def compute(self) -> torch.Tensor:
        """
        Calculate the MAE.
        """
        return self.error / self.counts


class SMAPE(ScalarMetric, tm.Metric):
    """
    The Symmetric Mean Absolute Percentage Error (SMAPE)
    """

    name = "SMAPE"

    def __init__(
        self, threshold: float = 1e-3, conditional: Optional[Dict[str, BinSpec]] = None
    ):
        ScalarMetric.__init__(self, conditional=conditional)
        tm.Metric.__init__(self)
        self.threshold = threshold
        error = torch.zeros(self.shape)
        counts = torch.zeros(self.shape)
        self.add_state("error", default=error, dist_reduce_fx="sum")
        self.add_state("counts", default=counts, dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        conditional: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            weights: An optional tensor of weights to apply to all validation samples.
            conditional: An optional dictionary containing values to
                condition the calculation of the bias onto.
        """
        pred = pred.squeeze()
        target = target.squeeze()
        if weights is None:
            weights = torch.ones_like(target)
        else:
            weights = weights.squeeze()

        if isinstance(target, MaskedTensor):
            mask = target.mask
            target = target.base
            if isinstance(weights, MaskedTensor):
                weights = weights.base
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
                pred = pred.base
            pred = pred[~mask]
            target = target[~mask]
            weights = weights[~mask]

        if self.conditional is None:

            if pred.dim() > 2:
                pred = pred.flatten()
                target = target.flatten()
                weights = weights.flatten()

            valid = target >= self.threshold
            pred = pred[valid]
            target = target[valid]
            weights = weights[valid]
            smape = torch.abs(pred - target) / (
                0.5 * (torch.abs(pred) + torch.abs(target))
            )
            self.error += (smape * weights).sum()
            self.counts += weights.sum()
            return None

        device = torch.device("cpu")
        self.to(device)

        coords = []
        for cond in self.conditional:

            coords_c = conditional[cond].squeeze()

            if mask is None:
                if coords_c.ndim < target.ndim:
                    coords_c = coords_c.reshape(coords_c.shape + (1,) * (target.ndim - coords_c.ndim))
                    coords_c = torch.broadcast_to(coords_c, target.shape)
                coords.append(coords_c)
            else:
                mask = mask.to(device=device)
                mask_s = mask.squeeze()
                # Expand channel dimension if necessary
                if coords_c.ndim < mask_s.ndim:
                    coords_c = coords_c.reshape(coords_c.shape + (1,) * (mask_s.ndim - coords_c.ndim))
                    coords_c = torch.broadcast_to(coords_c, mask_s.shape)
                coords.append(coords_c[~mask_s])

        coords = torch.stack(coords, -1).to(device=device)
        bins = tuple([bns.to(device=device, dtype=pred.dtype) for bns in self.bins])
        pred = pred.to(device=device)
        target = target.to(device=device)
        weights = weights.to(device=device)

        valid = target >= self.threshold
        pred = pred[valid]
        target = target[valid]
        coords = coords[valid]
        weights = weights[valid]
        smape = torch.abs(pred - target) / (
            0.5 * (torch.abs(pred) + torch.abs(target))
        )
        self.error += torch.histogramdd(coords, bins=bins, weight=smape * weights)[
            0
        ]
        self.counts += torch.histogramdd(coords, bins=bins, weight=weights)[0]

    def compute(self) -> torch.Tensor:
        """
        Calculate the SMAPE.
        """
        return self.error / self.counts


class HeidkeSkillScore(CategoricalDetectionMetric, tm.Metric):
    """
    The Heidke Skill score.
    """

    name = "Bias"

    def __init__(self, conditional: Optional[Dict[str, BinSpec]] = None, **kwargs):
        ScalarMetric.__init__(self, conditional)
        tm.Metric.__init__(self, **kwargs)
        counts = torch.zeros(self.shape)
        self.add_state("hits", default=torch.zeros(self.shape), dist_reduce_fx="sum")
        self.add_state("false_alarms", default=torch.zeros(self.shape), dist_reduce_fx="sum")
        self.add_state("misses", default=torch.zeros(self.shape), dist_reduce_fx="sum")
        self.add_state("correct_rejections", default=torch.zeros(self.shape), dist_reduce_fx="sum")

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        conditional: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            pred: A tensor containing the predicted classes as a float tensor containing
                ones and zeros.
            target: The true classes.
            weights: Optional weight tensor used to weigh the contribution of each
                sample to the overall bias.
            conditional: An optional dictionary containing values to
                condition the calculation of the bias onto.
        """
        pred = pred.squeeze()
        target = target.squeeze()
        if weights is None:
            weights = torch.ones_like(target)
        else:
            weights = weights.squeeze()

        mask = None
        if isinstance(target, MaskedTensor):
            mask = target.mask
            target = target.base
            if isinstance(weights, MaskedTensor):
                weights = weights.base
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
                pred = pred.base
            pred = pred[~mask]
            target = target[~mask]
            weights = weights[~mask]

        if self.conditional is None:
            self.hits += (weights * (0.5 < pred) * (0.5 < target)).sum()
            self.false_alarms += (weights * (0.5 < pred) * (target < 0.5)).sum()
            self.misses += (weights * (pred < 0.5) * (0.5 < target)).sum()
            self.correct_rejections += (weights * (pred < 0.5) * (target < 0.5)).sum()
            return None

        device = torch.device("cpu")
        self.to(device=device)

        coords = []
        for cond in self.conditional:

            coords_c = conditional[cond].squeeze()

            if mask is None:
                if coords_c.ndim < target.ndim:
                    coords_c = coords_c.reshape(coords_c.shape + (1,) * (target.ndim - coords_c.ndim))
                    coords_c = torch.broadcast_to(coords_c, target.shape)
                coords.append(coords_c)
            else:
                mask = mask.to(device=device)
                mask_s = mask.squeeze()
                # Expand channel dimension if necessary
                if coords_c.ndim < mask_s.ndim:
                    coords_c = coords_c.reshape(coords_c.shape + (1,) * (mask_s.ndim - coords_c.ndim))
                    coords_c = torch.broadcast_to(coords_c, mask_s.shape)
                coords.append(coords_c[~mask_s])

        coords = torch.stack(coords, -1).to(device=device)

        bins = tuple([bns.to(device=device, dtype=pred.dtype) for bns in self.bins])
        wgts = ((pred - target) * weights).to(device=device)

        hits = (weights * (0.5 < pred) * (0.5 < target)).sum()
        self.hits += torch.histogramdd(hits, bins=bins, weight=wgts)[0]

        false_alarms = (weights * (0.5 < pred) * (target < 0.5)).sum()
        self.false_alarms += torch.histogramdd(false_alarms, bins=bins, weight=wgts)[0]

        misses = (weights * (pred < 0.5) * (0.5 < target)).sum()
        self.misses += torch.histogramdd(misses, bins=bins, weight=wgts)[0]

        correct_rejections = (weights * (pred < 0.5) * (target < 0.5)).sum()
        self.correct_rejection += torch.histogramdd(correct_rejections, bins=bins, weight=weights.to(device=device))[0]

    def compute(self) -> torch.Tensor:
        """
        Compute the mean error.

        Return:
            The bias over all data.
        """
        n_tp = self.hits
        n_fp = self.false_alarms
        n_tn = self.correct_rejections
        n_fn = self.misses

        hss = 2.0 * (n_tp * n_tn - n_fp * n_fn) / ((n_tp + n_fn) * (n_fn + n_tn) + (n_tp + n_fp) * (n_fp + n_tn))
        return hss




class PlotSamples(tm.Metric):
    """
    Plots images of retrieved 2D fields for the n samples with the highest validation
    loss.
    """

    name = "Validation samples"

    def __init__(self, n_samples: int = 8, conditional: Any = None):
        """
        Args:
            n_samples: The number of samples to display.

        """
        super().__init__()
        self.n_samples = n_samples

        self.indices = []
        self.preds = []
        self.targets = []
        self.sample_indices = None
        self.batch_start = 0
        self.step = 0

    def reset(self):
        """
        Reset metric.
        """
        if self.sample_indices is None:
            permutation = np.random.permutation(len(self.indices))[: self.n_samples]
            permutation = sorted(permutation)
            self.sample_indices = [self.indices[ind] for ind in permutation]
        self.batch_start = 0
        self.preds = []
        self.targets = []
        self.step += 1

    @rank_zero_only
    def update(self, pred: torch.Tensor, target: torch.Tensor, conditional: Any = None):
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            conditional: Ignored.
        """
        if isinstance(pred, list):
            batch_size = pred[0].shape[0]
        else:
            batch_size = pred.shape[0]

        if self.sample_indices is None:
            ind = np.random.randint(0, batch_size)
            sample_ind = self.batch_start + ind
            self.indices.append(sample_ind)
            if isinstance(target, list):
                self.targets.append([elem[ind : ind + 1].cpu() for elem in target])
                self.preds.append([elem[ind : ind + 1].cpu() for elem in pred])
            else:
                self.targets.append(target[ind : ind + 1].cpu())
                self.preds.append(pred[ind : ind + 1].cpu())
        else:
            batch_indices = [
                ind - self.batch_start
                for ind in self.sample_indices
                if (ind >= self.batch_start) and (ind < self.batch_start + batch_size)
            ]
            for batch_ind in batch_indices:
                if isinstance(target, list):
                    self.preds.append(
                        [elem[batch_ind : batch_ind + 1] for elem in pred]
                    )
                    self.targets.append(
                        [elem[batch_ind : batch_ind + 1] for elem in target]
                    )
                else:
                    self.preds.append(pred[batch_ind : batch_ind + 1])
                    self.targets.append(target[batch_ind : batch_ind + 1])
        self.batch_start += batch_size

    @rank_zero_only
    def log(
        self, lightning_module: LightningModule, output_name: Optional[str] = None
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import colormaps
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import LogNorm, Normalize
        except ImportError:
            LOGGER.warning(
                "Could not import 'matplotlib', which is required by the PlotSamples"
                "metric. Not producing any plots."
            )
            return {}
        sequences = self.compute()
        if len(sequences) == 0:
            LOGGER.warning(
                "PlotSamples metric did not return any images for output '%s'.",
                output_name,
            )
            return None

        if not isinstance(sequences, list):
            sequences = [sequences]

        images = []
        for sequence in sequences:
            images += sequence["pred"] + sequence["target"]
            n_rows = len(sequence["pred"])

        img = make_grid(images, nrow=n_rows)
        name = self.name + f" ({output_name}, {self.sample_indices})"
        if hasattr(lightning_module.logger.experiment, "log_image"):
            lightning_module.logger.experiment.log_image(name, img)
        elif isinstance(lightning_module.logger, loggers.wandb.WandbLogger):
            lightning_module.logger.log_image(
                key=name, images=[(255 * img[:3]).to(dtype=torch.uint8)]
            )
        elif hasattr(lightning_module.logger.experiment, "add_image"):
            lightning_module.logger.experiment.add_image(name, img, self.step)
        else:
            LOGGER.warning(
                "Logger has no image logging functionality. Not logging output "
                "from PlotSamples metric."
            )

    @rank_zero_only
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Plots images of all predictions in the validation batch with the highest
        loss.

        Return:
            A dictionary containing lists of plots of predicted 2D fields for
            the predictions ('pred') and targets ('target') in the batch with
            the highest validation loss.
        """
        try:
            from matplotlib import colormaps
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import LogNorm, Normalize
        except ImportError:
            LOGGER.warning(
                "Could not import 'matplotlib', which is required by the PlotSamples"
                "metric. Not producing any plots."
            )
            return {}

        if self.sample_indices is None:
            permutation = np.random.permutation(len(self.indices))[: self.n_samples]
            permutation = sorted(permutation)
            self.sample_indices = [self.indices[ind] for ind in permutation]
            self.targets = [self.targets[ind] for ind in permutation]
            self.preds = [self.preds[ind] for ind in permutation]

        images = {}

        if isinstance(self.targets[0], torch.Tensor):
            target_min = np.nanmin(
                torch.stack(self.targets).to(dtype=torch.float32).cpu().numpy()
            )
            target_max = np.nanmax(
                torch.stack(self.targets).to(dtype=torch.float32).cpu().numpy()
            )
            target_max *= 0.2
            cmap = colormaps["magma"]
            cmap.set_bad("grey")

            for pred, target in zip(self.preds, self.targets):
                if isinstance(pred, RegressionTensor):
                    pred = pred.expected_value()[0]
                elif isinstance(pred, ClassificationTensor):
                    target_min = 0
                    target_max = pred.shape[1] - 1
                    pred = pred.to(dtype=torch.float32).most_likely_class()[0]
                    cmap = colormaps["Set1"]
                elif isinstance(pred, DetectionTensor):
                    target_min = 0
                    target_max = 1
                    pred = pred.to(dtype=torch.float32).probability().to(dtype=torch.float32)[0]
                else:
                    continue

                if pred.dim() == 3:
                    pred = pred[0]
                if pred.dim() != 2:
                    LOGGER.warning(
                        "PlotSamples metric received targets whose expected value "
                        "is not 2D and thus can't be plotted."
                    )
                    return {}

                target = target[0]
                if target.dim() == 3:
                    target = target[0]

                pred = pred.to(dtype=torch.float32).cpu().numpy()
                target = target.to(dtype=torch.float32).cpu().numpy()

                mappable = ScalarMappable(
                    cmap=cmap, norm=Normalize(target_min, target_max)
                )
                img_target = np.transpose(mappable.to_rgba(pred), [2, 0, 1])
                img_pred = np.transpose(mappable.to_rgba(target), [2, 0, 1])

                images.setdefault("pred", []).append(torch.tensor(img_pred))
                images.setdefault("target", []).append(torch.tensor(img_target))

            return images

        sequences = []
        for pred, target in zip(self.preds, self.targets):
            images = {}

            target_min = np.nanmin(torch.stack(target).float().cpu().numpy())
            target_max = np.nanmax(torch.stack(target).float().cpu().numpy())
            target_max = max(target_min * 1.5, target_max)
            cmap = colormaps["magma"]
            cmap.set_bad("grey")

            for pred_s, target_s in zip(pred, target):
                pred_s = pred_s.expected_value()[0]
                if pred_s.dim() == 3:
                    pred_s = pred_s[0]
                if pred_s.dim() != 2:
                    LOGGER.warning(
                        "PlotSamples metric received targets whose expected value "
                        "is not 2D and thus can't be plotted."
                    )
                    return {}

                target_s = target_s[0]
                if target_s.dim() == 3:
                    target_s = target_s[0]

                pred_s = pred_s.to(dtype=torch.float32).cpu().numpy()
                target_s = target_s.to(dtype=torch.float32).cpu().numpy()

                mappable = ScalarMappable(
                    cmap=cmap, norm=Normalize(target_min, target_max)
                )
                img_pred = np.transpose(mappable.to_rgba(pred_s), [2, 0, 1])
                img_target = np.transpose(mappable.to_rgba(target_s), [2, 0, 1])

                images.setdefault("pred", []).append(torch.tensor(img_pred))
                images.setdefault("target", []).append(torch.tensor(img_target))

            sequences.append(images)

        return sequences


class PlotMeans(tm.Metric):
    """
    Plots images of the mean retrieved quantities calculated along the spatial dimension.
    """
    name = "Means"

    def __init__(self):
        """
        Args:
            n_samples: The number of samples to display.

        """
        super().__init__()
        self.sum_v = None
        self.sum_h = None
        self.cts_v = None
        self.cts_h = None
        self.step = 0

    def reset(self):
        """
        Reset metric.
        """
        self.sum_v = None
        self.sum_h = None
        self.cts_v = None
        self.cts_h = None
        self.step += 1

    @rank_zero_only
    def update(self, pred: torch.Tensor, target: torch.Tensor, conditional: Any = None):
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            conditional: Ignored.
        """
        if not isinstance(pred, list):
            pred = [pred]

        for pred_s in pred:

            dim_v = pred_s.dim() - 2
            dim_h = pred_s.dim() - 1

            pred_s_f = torch.nan_to_num(pred_s, nan=0.0).float()

            dims_v = [dim for dim in range(pred_s.dim()) if dim != dim_v]
            cts_v = torch.isfinite(pred_s).to(torch.float32).sum(dim=dims_v)
            sum_v = pred_s_f.sum(dim=dims_v)

            dims_h = [dim for dim in range(pred_s.dim()) if dim != dim_h]
            cts_h = torch.isfinite(pred_s).to(torch.float32).sum(dim=dims_h)
            sum_h = pred_s_f.sum(dim=dims_h)

            if self.sum_v is None:
                self.sum_v = sum_v.cpu()
                self.cts_v = cts_v.cpu()
                self.sum_h = sum_h.cpu()
                self.cts_h = cts_h.cpu()
            else:
                self.sum_v += sum_v.cpu()
                self.cts_v += cts_v.cpu()
                self.sum_h += sum_h.cpu()
                self.cts_h += cts_h.cpu()


    @rank_zero_only
    def log(
        self, lightning_module: LightningModule, output_name: Optional[str] = None
    ) -> None:

        try:
            import matplotlib.pyplot as plt
            from PIL import Image
        except ImportError:
            LOGGER.warning(
                "Could not import 'matplotlib', which is required by the PlotMeans"
                "metric. Not producing any plots."
            )
            return {}

        if self.sum_v is None:
            LOGGER.warning(
                "PlotMeans metric did not return any images for output '%s'.",
                output_name,
            )
            return None

        fig = plt.Figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.sum_v / self.cts_v, label="Vertical mean")
        ax.plot(self.sum_h / self.cts_h, label="Horizontal mean")
        ax.legend()
        name = f"{self.name}  ({output_name})"

        buf = io.BytesIO()
        fig.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        img = np.array(Image.open(buf)) / 255.0
        img = np.transpose(img, [2, 0, 1])
        lightning_module.logger.experiment.add_image(name, img, self.step)
        del fig
        del ax

        if hasattr(lightning_module.logger.experiment, "log_image"):
            lightning_module.logger.experiment.log_image(name, img)
        elif isinstance(lightning_module.logger, loggers.wandb.WandbLogger):
            lightning_module.logger.log_image(
                key=name, images=[(255 * img[:3]).to(dtype=torch.uint8)]
            )
        elif hasattr(lightning_module.logger.experiment, "add_image"):
            lightning_module.logger.experiment.add_image(name, img, self.step)
        else:
            LOGGER.warning(
                "Logger has no image logging functionality. Not logging output "
                "from PlotMeans metric."
            )

    @rank_zero_only
    def compute(self) -> Dict[str, torch.Tensor]:
        pass


class Histogram(ScalarMetric, tm.Metric):
    """
    Plots a histogram of the predicted and reference data.
    """
    name = "Histogram"

    def __init__(self, v_min: float = 0.0, v_max: float = 100.0):
        """
        Args:
            n_samples: The number of samples to display.

        """
        ScalarMetric.__init__(self)
        tm.Metric.__init__(self)

        self.v_min = v_min
        self.v_max = v_max

        self.bins = torch.tensor(np.linspace(self.v_min, self.v_max).astype(np.float32))
        counts = torch.zeros((2, self.bins.shape[0] - 1))
        self.add_state("counts", default=counts, dist_reduce_fx="sum")
        self.step = 0

    def reset(self):
        """
        Reset metric.
        """
        super().reset()
        self.step += 1

    @rank_zero_only
    def update(self, pred: torch.Tensor, target: torch.Tensor, conditional: Any = None):
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            conditional: Ignored.
        """
        device = torch.device("cpu")
        dtype = torch.float32

        self.to(device=device)
        pred = pred.squeeze().to(device=device, dtype=dtype)
        target = target.squeeze().to(device=device, dtype=dtype)

        mask = None
        if isinstance(target, MaskedTensor):
            mask = target.mask
            target = target.base
            if isinstance(pred, MaskedTensor):
                mask = mask | pred.mask
                pred = pred.base
            pred = pred[~mask]
            target = target[~mask]

        self.counts[0] += torch.histogram(target, bins=self.bins)[0]
        self.counts[1] += torch.histogram(pred, bins=self.bins)[0]


    @rank_zero_only
    def log(
        self, lightning_module: LightningModule, output_name: Optional[str] = None
    ) -> None:

        try:
            import matplotlib.pyplot as plt
            from PIL import Image
        except ImportError:
            LOGGER.warning(
                "Could not import 'matplotlib', which is required by the PlotMeans"
                "metric. Not producing any plots."
            )
            return {}

        fig = plt.Figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)
        x = 0.5 * (self.bins[1:] + self.bins[:-1])
        ax.plot(x, self.counts[0].float().cpu().numpy(), label="Target")
        ax.plot(x, self.counts[1].float().cpu().numpy(), label="Predicted")
        ax.set_yscale("log")
        ax.legend()
        name = f"{self.name}  ({output_name})"

        buf = io.BytesIO()
        fig.savefig("test.jpg", format='jpeg', bbox_inches='tight')
        fig.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        img = np.array(Image.open(buf)) / 255.0
        img = np.transpose(img, [2, 0, 1])
        lightning_module.logger.experiment.add_image(name, img, self.step)
        del fig
        del ax

        if hasattr(lightning_module.logger.experiment, "log_image"):
            lightning_module.logger.experiment.log_image(name, img)
        elif isinstance(lightning_module.logger, loggers.wandb.WandbLogger):
            lightning_module.logger.log_image(
                key=name, images=[(255 * img[:3]).to(dtype=torch.uint8)]
            )
        elif hasattr(lightning_module.logger.experiment, "add_image"):
            lightning_module.logger.experiment.add_image(name, img, self.step)
        else:
            LOGGER.warning(
                "Logger has no image logging functionality. Not logging output "
                "from PlotMeans metric."
            )


    @rank_zero_only
    def compute(self) -> Dict[str, torch.Tensor]:
        pass


class BrierScore(ProbabilisticDetectionMetric, MSEBase):
    """
    The Brier score, which is just the MSE applied to probabilities.
    """
    name = "Brier Score"

    def __init__(self, conditional: Optional[Dict[str, BinSpec]] = None):
        ProbabilisticDetectionMetric.__init__(self, conditional=conditional)
        MSEBase.__init__(self)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        conditional: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
            weights: An optional tensor of weights to apply to all validation samples.
            conditional: An optional dictionary containing values to
                condition the calculation of the bias onto.
        """
        super().update(
            pred,
            target.to(dtype=pred.dtype),
            weights,
            conditional
        )
