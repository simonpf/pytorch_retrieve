"""
pytorch_retrieve.metrics
========================

Provides TorchMetrics metrics modules that handle pytorch_retrieve's
probabilistic outputs.
"""
import logging
from typing import Dict, Optional

from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_only
import numpy as np
import torch
import torchmetrics as tm
from torchvision.utils import make_grid

from pytorch_retrieve.tensors import MaskedTensor


LOGGER = logging.getLogger(__name__)


class ScalarMetric:
    """
    Dummy class to identify metrics that evaluate scalar predictions,
    i.e., predictions that consists of a single value.
    """

    def log(
        self, lightning_module: LightningModule, output_name: Optional[str] = None
    ) -> None:
        value = self.compute()
        name = self.name
        if output_name is not None:
            name += f" ({output_name})"
        lightning_module.log(self.name, value, on_step=False, on_epoch=True, sync_dist=True)


class Bias(ScalarMetric, tm.Metric):
    """
    The mean error.
    """

    name = "Bias"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("counts", default=torch.tensor(0.0), dist_reduce_fx="sum")

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

        if pred.dim() >= 2:
            pred = pred.flatten()
            target = target.flatten()
        super().update(pred.to(dtype=torch.float32), target.to(dtype=torch.float32))


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


class PlotSamples(tm.Metric):
    """
    Plots images of retrieved 2D fields for the n samples with the highest validation
    loss.
    """

    name = "Validation samples"

    def __init__(self, n_samples: int = 8):
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
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred: A tensor containing the point predictions from the
                retrieval model.
            target: A tensor containing the reference values corresponding
                to 'pred'.
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
                self.targets.append([elem[ind : ind + 1] for elem in target])
                self.preds.append([elem[ind : ind + 1] for elem in pred])
            else:
                self.targets.append(target[ind : ind + 1])
                self.preds.append(pred[ind : ind + 1])
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
        sequences = self.compute()
        if len(sequences) == 0:
            LOGGER.warning(
                "PlotSamples metric did not return any images for output '%s'.",
                output_name
            )
            return None

        if not isinstance(sequences, list):
            sequences = [sequences]

        images = []
        for sequence in sequences:
            images += sequence["pred"] + sequence["target"]
            n_rows = len(sequence["pred"])

        img = make_grid(images, nrow=n_rows)
        name = self.name + f" ({output_name})"
        if hasattr(lightning_module.logger.experiment, "log_image"):
            lightning_module.logger.experiment.log_image(name, img)
        if hasattr(lightning_module.logger.experiment, "add_image"):
            lightning_module.logger.experiment.add_image(name, img, self.step)
        else:
            LOGGER.warning(
                "Logger has not image logging functionality. Not logging output "
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
            from matplotlib.cm import ScalarMappable, get_cmap
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
            cmap = get_cmap("magma")
            cmap.set_bad("grey")

            for pred, target in zip(self.preds, self.targets):
                pred = pred.expected_value()[0]
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

            target_min = np.nanmin(torch.stack(target).cpu().numpy())
            target_max = np.nanmax(torch.stack(target).cpu().numpy())
            target_max *= 0.2
            cmap = get_cmap("magma")
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
