"""
pytorch_retrieve.lightning

"""

import copy
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler
from torch import nn
from torchmetrics import Metric
import lightning as L
from lightning.pytorch import loggers

from pytorch_retrieve.tensors import DetectionTensor, MaskedTensor
from pytorch_retrieve.metrics import (
    CategoricalDetectionMetric,
    ProbabilisticDetectionMetric,
    ScalarMetric
)
from pytorch_retrieve.utils import WarmupLR

RetrievalInput = Union[torch.Tensor, list, dict]


LOGGER = logging.getLogger(__name__)


class LightningRetrieval(L.LightningModule):
    """
    The RetrievalModule implements a LightningModule for the training
    of PytorchRetrieve models.
    """

    def __init__(
        self,
        model: nn.Module,
        training_schedule: Dict[str, "TrainingConfig"] = None,
        name: Optional[str] = None,
        model_dir: Optional[Path] = None,
        logger: Optional[Callable[[Path, str], L.pytorch.loggers.Logger]] = None,
    ):
        """
        Create lightning retrieval module to train a retrieval using a
        given PyTorch model.

        Args:
            model: A PyTorch module implementing the neural network to
                be used for the retrieval.
        """
        super().__init__()
        self.model = model

        if name is None:
            name = "retrieval_model"
            if hasattr(model, "config_dict"):
                name = model.config_dict.get("name", "retrieval_model")
        self.name = name

        self.training_schedule = training_schedule
        if self.training_schedule is None:
            self.stage_names = None
        else:
            self.stage_names = list(self.training_schedule.keys())

        self.stage = 0
        self.prev_optim = None

        if model_dir is None:
            model_dir = Path(".")
        self.model_dir = model_dir
        self.log_dir = Path(model_dir) / "logs"
        if logger is None:
            logger = loggers.TensorBoardLogger
        self.logger_class = logger

        self.inputs_prev = None
        self.pred_prev = None
        self.target_prev = None
        self.mean_loss = None
        self.alpha = 0.99

    @property
    def training_finished(self):
        if self.training_schedule is None:
            return self.stage > 0
        return self.stage >= len(self.training_schedule)

    @property
    def stage(self) -> int:
        """
        The index of the current stage.
        """
        return self._stage

    def save_model(self, path: Optional[Path] = None) -> Path:
        """
        Save retrieval module as a PyTorch state dict.

        Args:
            path: A path pointing to a folder or file to which to write
                the retrieval module.

        Return:
            A Path object pointing to the path to which the model was written.
        """
        if path is None:
            path = Path(".")
        else:
            path = Path(path)

        if path.is_dir():
            name = self.name
            if name is None:
                name = "retrieval_model"
            path = path / (name + ".pt")
        self.model.save(path)
        return path

    @stage.setter
    def stage(self, stage: int) -> None:
        self._stage = stage

    @property
    def stage_name(self) -> str:
        """
        The name of the current training stage.
        """
        return self.stage_names[self.stage]

    @property
    def current_training_config(self) -> "TrainingConfig":
        """
        Returns the configuration object specifying the currently active
        training stage.
        """
        if self.training_finished or self.training_schedule is None:
            return None
        stage_name = self.stage_names[self.stage]
        return self.training_schedule[stage_name]

    @property
    def debug(self) -> bool:
        try:
            if self.current_training_config is None:
                return False
        except AttributeError:
            return False
        return self.current_training_config.debug

    def training_step_single_pred(
        self, pred: torch.Tensor, target: Union[torch.Tensor, dict]
    ) -> torch.Tensor:
        """
        Loss calculation for a prediction with just a single output.

        Args:
            pred: A torch.Tensor containing the single prediction from the
                retrieval module.
            target: A torch.Tensor or a dict with a single entry containing
                the target corresponding to the prediction in 'pred'.

        Return:
            The loss of the prediction with respect to the given target.
        """
        weights = None
        if isinstance(target, dict):
            if len(target) > 1:
                raise RuntimeError(
                    "The model prediction is only a tensor but the target is"
                    " a dict  with more than 1 entry. Therefore, it is not "
                    " clear which  target element the predictions should "
                    " be associated with."
                )
            name, target = next(iter(target.items()))

        else:
            name = None

        if isinstance(target, tuple):
            target, weights = target
        else:
            weights = None

        if isinstance(pred, list):
            if not isinstance(target, list):
                raise RuntimeError(
                    "Model predicts a sequence but the reference data is not."
                )
            loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            if weights is None:
                weights = [None] * len(target)

            for pred_s, target_s, weights_s in zip(pred, target, weights):
                mask = torch.isnan(target_s)
                if mask.any():
                    target_s = torch.nan_to_num(target_s, 0.0)
                    target_s = MaskedTensor(target_s, mask=mask)
                    if weights_s is not None:
                        weights_s = MaskedTensor(weights_s, mask=mask)
                loss += pred_s.loss(target_s, weights=weights_s)

            if name is not None:
                self.log("Training loss", loss, prog_bar=True)
            else:
                self.log(f"Training loss ({name})", loss, prog_bar=True)
            return loss

        mask = torch.isnan(target)
        if mask.any():
            target = torch.nan_to_num(target, 0.0)
            target = MaskedTensor(target, mask=mask)
            if weights is not None:
                weights = MaskedTensor(weights, mask=mask)

        loss = pred.loss(target, weights=weights)

        if name is not None:
            self.log("Training loss", loss)
        else:
            self.log(f"Training loss ({name})", loss)

        return loss

    def training_step(self, batch: tuple, batch_idx: int):
        """
        Evaluate module prediction for a single step.

        Args:
            batch: A tuple ``(x, y)`` containing the inputs and output targets
                for a given training batch.
            batch_idx: An integer specifying the index of the current batch.

        Return:
            A single scalar value containing the loss of the module prediction
            w.r.t. the given target for the current batch if the module has only
            a single output. Otherwise, a dictionary of loss values is returned
            with the entry 'loss' containing the total loss, which is just the
            sum of all indiviual losses.
        """
        inputs, target = batch
        pred = self.model(inputs)

        if not isinstance(pred, dict):
            loss = self.training_step_single_pred(pred, target)
            self.track_mean_loss(inputs, pred, target, loss)
            return loss

        if not isinstance(target, dict):
            if len(pred) > 1:
                raise RuntimeError(
                    "The model prediction is a dict with more than one "
                    " entry but the target is a single tensor. Therefore, it "
                    " is not clear which target element the predictions should "
                    " be associated with."
                )
            pred = next(iter(pred.values()))
            loss = self.training_step_single_pred(pred, target)
            self.track_mean_loss(inputs, pred, target, loss)

            return loss

            raise RuntimeError(
                "If the model output is a 'dict' the reference data must also "
                "be a 'dict'."
            )

        losses = {}

        if isinstance(inputs, dict):
            inpt = next(iter(inputs.values()))
            if isinstance(inpt, list):
                inpt = inpt[0]
            device = inpt.device
            dtype = inpt.dtype
        else:
            inpt = inputs
            if isinstance(inpt, list):
                inpt = inpt[0]
            device = inputs.device
            dtype = inputs.dtype

        tot_loss = torch.tensor(0.0, requires_grad=True, device=device, dtype=dtype)
        tot_samples = 0

        for name in pred:
            key = name.split("::")[-1]

            pred_k = pred[key]
            target_k = target[key]
            weights_k = target.get(key + "_weights", None)

            if isinstance(pred_k, list):
                if not isinstance(target_k, list):
                    raise RuntimeError(
                        "Model predicts a sequence but the reference data is not."
                    )

                if weights_k is None:
                    weights_k = [None] * len(target_k)

                losses[key] = 0.0
                for pred_k_s, target_k_s, weights_k_s in zip(
                    pred_k, target_k, weights_k
                ):
                    mask = torch.isnan(target_k_s)
                    if mask.any():
                        target_k_s = torch.nan_to_num(target_k_s, 0.0)
                        target_k_s = MaskedTensor(target_k_s, mask=mask)
                        if weights_k_s is not None:
                            weights_k_s = MaskedTensor(weights_k_s, mask=mask)

                    if mask.all() or torch.isnan(pred_k_s).any():
                        continue

                    if weights_k_s is None:
                        n_samples = (~mask).sum()
                    else:
                        n_samples = weights_k_s.sum()

                    if n_samples == 0:
                        pred_k_s = 0.0 * pred_k_s
                        target_k_s = 0.0 * target_k_s

                    tot_samples += n_samples
                    loss_k_s = pred_k_s.loss(target_k_s, weights=weights_k_s)
                    tot_loss = tot_loss + n_samples * loss_k_s
                    losses[name] += loss_k_s.item()

            else:
                mask = torch.isnan(target_k)
                if mask.any():
                    target_k = torch.nan_to_num(target_k, 0.0)
                    target_k = MaskedTensor(target_k, mask=mask)
                    if weights_k is not None:
                        weights_k = MaskedTensor(weights_k, mask=mask)

                if weights_k is None:
                    n_samples = (~mask).sum()
                else:
                    n_samples = weights_k.sum()

                loss_k = pred[name].loss(target_k, weights=weights_k)
                tot_loss = tot_loss + n_samples * loss_k
                tot_samples += n_samples
                losses[name] = loss_k.item()
                pred_k_s = pred[name]

        if tot_samples > 0:
            tot_loss = tot_loss / tot_samples
        else:
            tot_loss = tot_loss + 0.0 * pred_k_s.sum()

        self.track_mean_loss(inputs, pred, target, tot_loss)

        log_dict = {}
        for name, loss in losses.items():
            log_dict[f"Training loss ({name})"] = loss
        self.log_dict(log_dict)
        self.log("Training loss", tot_loss)
        losses["loss"] = tot_loss

        return losses

    def track_mean_loss(
            self,
            inputs: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
            target: Dict[str, torch.Tensor],
            loss: torch.Tensor
    ) -> None:
        """
        Track mean loss.

        Args:
            inputs: The model input for this training step.
            pred: The model prediction.
            target: The target data for this training step.
            loss: The calculated loss.

        """

        if self.mean_loss is not None:
            self.log("Mean loss", self.mean_loss, on_step=True, prog_bar=True)

        if self.mean_loss is None:
            self.mean_loss = loss.item()
        else:
            # Check if loss is anomalous
            try:
                if self.debug and (
                    loss > 5 * self.mean_loss or torch.isnan(loss).all()
                ):
                    filename = f"inputs_prev_{self.global_rank}_{self.global_step}_{loss:.2f}.pt"
                    torch.save(self.inputs_prev, filename)
                    filename = f"targets_prev_{self.global_rank}_{self.global_step}_{loss:.2f}.pt"
                    torch.save(self.target_prev, filename)
                    filename = f"preds_prev_{self.global_rank}_{self.global_step}_{loss:.2f}.pt"
                    torch.save(self.pred_prev, filename)
                    filename = f"inputs_{self.global_rank}_{self.global_step}_{loss:.2f}.pt"
                    torch.save(inputs, filename)
                    filename = f"targets_{self.global_rank}_{self.global_step}_{loss:.2f}.pt"
                    torch.save(target, filename)
                    filename = (
                        f"preds_{self.global_rank}_{self.global_step}_{loss:.2f}.pt"
                    )
                    torch.save(pred, filename)

                self.mean_loss = (
                    self.alpha * self.mean_loss + (1.0 - self.alpha) * loss.item()
                )

                if self.debug:
                    if torch.isnan(loss).all():
                        raise ValueError("NAN encountered in training.")
                    self.inputs_prev = inputs
                    self.target_prev = target
                    self.pred_prev = pred
            except AttributeError:
                pass

    def on_train_start(self):
        """
        Initialize metrics.
        """
        current_config = self.current_training_config
        if current_config is not None:
            self.metrics = self.current_training_config.get_metrics_dict(
                self.model.output_names
            )
        else:
            self.metrics = []

        # Print number of FSDP wrapped modules.
        n_wrapped = sum(1 for m in self.trainer.strategy.model.modules() if getattr(m, "_fsdp_wrapped_module", None))
        if n_wrapped > 0:
            LOGGER.info(f"FSDP-wrapped modules: %s.", n_wrapped)

    def on_before_optimizer_step(self, optimizer):
        from lightning.pytorch.utilities import grad_norm

        pass
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # norms = grad_norm(self.model, norm_type=2)
        # self.log_dict(norms)

    def validation_step_single_sequence(
            self,
            pred: List[torch.Tensor],
            target: Union[List[torch.Tensor], dict],
            inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Validation step for a sequence prediction with a single output.

        Sequence predictions are handled as lists of tensors with each element representing a separate
        step of the sequence. This function collects validation statistic by comparing the predicted
        sequence 'pred' with the target sequence 'target'.

        Args:
            pred: A list of tensors containing the predicted sequence.
            target: A list of tensors containing the targets.
            inputs: A dictionary containing the model inputs used to condition certain metrics.

        Return:
            The total validation loss.
        """
        weights = None
        if isinstance(target, dict):
            if len(target) > 1:
                raise RuntimeError(
                    "The model prediction is a single, unnamed sequence "
                    " but the target is a dict  with more than 1 entry. "
                    " Therefore, it is not clear which  target element "
                    " the predictions should be associated with."
                )
            target = next(iter(target.values()))
            if isinstance(target, tuple):
                target, weights = target
            else:
                weights = None

        metrics = self.metrics
        if len(metrics) > 1:
            raise RuntimeError(
                "Metrics for more than two outputs are defined but "
                "the model provided only a single, unnamed output."
            )
        metrics = next(iter(metrics.values()))

        if not isinstance(target, list):
            raise ValueError("Prediction is a sequence but target is not.")

        scalar_metrics = [
            metric for metric in metrics if isinstance(metric, ScalarMetric)
        ]
        prob_detection_metrics = [
            metric for metric in metrics if isinstance(metric, ProbabilisticDetectionMetric)
        ]
        cat_detection_metrics = [
            metric for metric in metrics if isinstance(metric, CategoricalDetectionMetric)
        ]
        # Other metrics
        other_metrics = [
            metric for metric in metrics if not isinstance(metric, (ScalarMetric, DetectionMetric))
        ]

        if weights is None:
            weights = [None] * len(target)

        loss = 0.0

        for step, (pred_s, target_s, weights_s) in enumerate(zip(pred, target, weights)):

            # Conditional data for metrics
            if inputs is None:
                cond = {"step": step}
            else:
                cond = inputs | {"step": step}

            mask = torch.isnan(target_s)
            if mask.any():
                target_s = MaskedTensor(target_s, mask=mask)
            if mask.all():
                continue

            loss += pred_s.loss(target_s, weights=weights_s)

            if hasattr(pred_s, "expected_value"):
                scalar_pred = pred_s.expected_value()
                for metric in scalar_metrics:
                    metric = metric.to(device=scalar_pred.device)
                    metric.update(scalar_pred, target_s, conditional=cond)

            if isinstance(pred_s, DetectionTensor):
                mlc = pred_s.most_likely_class()
                for metric in cat_detection_metrics:
                    metric = metric.to(device=mlc.device, dtype=mlc.dtype)
                    metric.update(mlc, target_s, conditional=cond)
                prob = pred_s.probability()
                for metric in porb_detection_metrics:
                    metric = metric.to(device=mlc.device, dtype=prob.dtype)
                    metric.update(prob, target_s, conditional=cond)

        for metric in other_metrics:
            metric = metric.to(device=pred_s.device)
            metric.update(pred, target, conditional=cond)

        self.log("Validation loss", loss)
        return loss

    def validation_step_single_pred(
            self,
            pred: torch.Tensor,
            target: Union[torch.Tensor, dict],
            inputs: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Validation step for a prediction with just a single output.

        Args:
            pred: A torch.Tensor containing the single prediction from the
                retrieval module.
            target: A torch.Tensor or a dict with a single entry containing
                the target corresponding to the prediction in 'pred'.
            inputs: Optional dictionary containing the model input, which is used to condition
                the validation metrics metrics.

        Return:
            The loss of the prediction with respect to the given target.
        """
        weights = None
        if isinstance(target, dict):
            if len(target) > 1:
                raise RuntimeError(
                    "The model prediction is only a tensor but the target is"
                    " a dict  with more than 1 entry. Therefore, it is not "
                    " clear which  target element the predictions should "
                    " be associated with."
                )
            target = next(iter(target.values()))

        if isinstance(target, tuple):
            target, weights = target

        metrics = self.metrics
        if len(metrics) > 1:
            raise RuntimeError(
                "Metrics for more than two outputs are defined but "
                "the model provided only a single, unnamed output."
            )
        metrics = next(iter(metrics.values()))

        mask = torch.isnan(target)
        if mask.any():
            target = MaskedTensor(target, mask=mask)

        loss = pred.loss(target, weights=weights)

        scalar_metrics = [
            metric for metric in metrics if isinstance(metric, ScalarMetric)
        ]
        # Other metrics
        other_metrics = [
            metric for metric in metrics if not isinstance(metric, ScalarMetric)
        ]

        if isinstance(pred, list):
            tot_samples = 0
            tot_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            for step, (pred_s, target_s) in enumerate(zip(pred, target)):

                if inputs is None:
                    cond = {"step": step * torch.ones_like(pred_s)}
                else:
                    cond = inputs | {"step": step * torch.ones_like(pred_s)}

                mask = torch.isnan(target_s)
                if mask.any():
                    target_s = MaskedTensor(target_s, mask=mask)
                if mask.all():
                    continue

                if weights is  None:
                    n_samples = (~mask).sum()
                else:
                    n_samples = weights.sum()

                if n_samples == 0:
                    pred_s = 0.0 * pred_s
                    target_s = 0.0 * target_s
                tot_samples += n_samples

                loss_s = pred_s.loss(target_s)
                tot_loss = tot_loss + loss_s * n_samples

                if hasattr(pred_s, "expected_value") and len(scalar_metrics) > 0:
                    pred_s = pred_s.expected_value()
                    for metric in scalar_metrics:
                        metric = metric.to(device=pred_s.device)
                        metric.update(pred_s, target_s, cond=cond)

            if tot_samples > 0:
                tot_loss = tot_loss / tot_samples

            for metric in other_metrics:
                metric = metric.to(device=pred_s.device)
                metric.update(pred, target, cond=cond)

        else:

            mask = torch.isnan(target)
            if mask.any():
                target = MaskedTensor(target, mask=mask)
                n_samples = (~mask).sum()
            else:
                n_samples = target.numel()

            tot_loss = pred.loss(target)

            for metric in other_metrics:
                metric = metric.to(device=pred.device)
                metric.update(pred, target, conditional=inputs)

            if hasattr(pred, "expected_value") and len(scalar_metrics) > 0:
                pred = pred.expected_value()
                for metric in scalar_metrics:
                    metric = metric.to(device=pred.device)
                    metric.update(pred, target, conditional=inputs)

        self.log("Validation loss", tot_loss)
        return tot_loss

    def validation_step(self, batch: tuple, batch_idx: int, dataloader_idx=0):
        """
        Perform validation step.

        Args:
            batch: A tuple ``(x, y)`` containing the inputs and output targets
                for a given training batch.
            batch_idx: An integer specifying the index of the current batch.

        Return:
            A single scalar value containing the loss of the module prediction
            w.r.t. the given target for the current batch if the module has only
            a single output. Otherwise, a dictionary of loss values is returned
            with the entry 'loss' containing the total loss, which is just the
            sum of all indiviual losses.
        """
        inputs, target = batch
        pred = self.model(inputs)

        if not isinstance(pred, dict):
            if isinstance(pred, list):
                return self.validation_step_single_sequence(pred, target, inputs=inputs)
            else:
                return self.validation_step_single_pred(pred, target, inputs=inputs)

        if not isinstance(target, dict):
            if len(pred) > 1:
                raise RuntimeError(
                    "The model prediction is a dict with more than one "
                    " entry but the target is a single tensor. Therefore, it "
                    " is not clear which target element the predictions should "
                    " be associated with."
                )
            pred = next(iter(pred.values()))
            if isinstance(pred, list):
                return self.validation_step_single_sequence(pred, target, inputs=inputs)
            else:
                return self.validation_step_single_pred(pred, target, inputs=inputs)

            raise RuntimeError(
                "If the model output is a 'dict' the reference data must also "
                "be a 'dict'."
            )

        metrics = self.metrics

        losses = {}
        tot_loss = 0.0
        for name, pred_k in pred.items():
            key = name.split("::")[-1]
            target_k = target[key]
            losses[name] = 0.0

            weights_k = target.get(key + "_weights", None)

            # Determine scalar metrics for this outout
            metrics_k = metrics.get(name, [])
            scalar_metrics = [
                metric for metric in metrics_k if isinstance(metric, ScalarMetric)
            ]
            # Other metrics
            other_metrics = [
                metric for metric in metrics_k if not isinstance(metric, ScalarMetric)
            ]

            if isinstance(pred_k, list):

                if weights_k is None:
                    weights_k = [None] * len(target_k)

                tot_samples = 0

                for step, (pred_k_s, target_k_s, weights_k_s) in enumerate(zip(
                    pred_k, target_k, weights_k
                )):

                    cond = inputs | {"step": step * torch.ones_like(pred_k_s)}

                    mask = torch.isnan(target_k_s)
                    if mask.any():
                        target_k_s = MaskedTensor(target_k_s, mask=mask)
                        if weights_k_s is not None:
                            weights_k_s = MaskedTensor(weights_k_s, mask=mask)
                    if mask.all():
                        continue

                    if weights_k_s is None:
                        n_samples = (~mask).sum()
                    else:
                        n_samples = weights_k_s.sum()

                    if n_samples == 0:
                        pred_k_s = 0.0 * pred_k_s
                        target_k_s = 0.0 * target_k_s
                    tot_samples += n_samples

                    loss_k_s = pred_k_s.loss(target_k_s, weights=weights_k_s)
                    tot_loss = tot_loss + loss_k_s * n_samples
                    losses[name] += loss_k_s.item()

                    if hasattr(pred_k_s, "expected_value") and len(scalar_metrics) > 0:
                        pred_k_s = pred_k_s.expected_value()
                        for metric in scalar_metrics:
                            metric = metric.to(device=pred_k_s.device)
                            metric.update(pred_k_s, target_k_s, conditional=cond)

                    for metric in other_metrics:
                        metric = metric.to(device=pred_k_s.device)
                        metric.update(pred_k, target_k, conditional=cond)

                if tot_samples > 0:
                    tot_loss = tot_loss / tot_samples

            else:
                mask = torch.isnan(target_k)
                if mask.any():
                    target_k = MaskedTensor(target_k, mask=mask)
                    if weights_k is not None:
                        weights_k = MaskedTensor(weights_k, mask=mask)
                        n_samples = weights_k.sum()
                    else:
                        n_samples = (~mask).sum()
                else:
                    if weights_k is None:
                        n_samples = target_k.numel()
                    else:
                        n_samples = weights_k.sum()

                loss_k = pred_k.loss(target_k, weights=weights_k)
                tot_loss += loss_k if n_samples > 0 else 0
                losses[name] += loss_k.item() if n_samples > 0 else 0

                for metric in other_metrics:
                    metric = metric.to(device=pred_k.device)
                    metric.update(pred_k, target_k, conditional=inputs)

                if hasattr(pred_k, "expected_value") and len(scalar_metrics) > 0:
                    pred_k = pred_k.expected_value()
                    for metric in scalar_metrics:
                        metric = metric.to(device=pred_k.device)
                        metric.update(pred_k, target_k, conditional=inputs)

        log_dict = {}
        for name, loss in losses.items():
            log_dict[f"Validation loss ({name})"] = loss
        self.log_dict(log_dict)
        self.log("Validation loss", tot_loss)

    def on_validation_epoch_end(self):
        """
        Log metrics and learning rate at the end of the validation epoch.
        """
        for output_name, metrics in self.metrics.items():
            for metric in metrics:
                metric = metric.to(self.device)
                metric.log(self, output_name=output_name)
                metric.reset()

        try:
            self.log(
                "Learning rate",
                self.trainer.optimizers[0].param_groups[0]["lr"],
            )
        except IndexError:
            pass

    def configure_optimizers(self):
        if self.training_schedule is None:
            optimizer = torch.optim.Adam(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=1e-3
            )
            return optimizer

        curr_config = self.current_training_config
        curr_name = self.stage_name

        # If set, use 'lr' attribute. This is done to support the automatic
        # LR search provided by Lightning.
        if hasattr(self, "lr"):
            curr_config = copy.copy(curr_config)
            curr_config.optimizer_args["lr"] = self.lr

        optimizer, scheduler = curr_config.get_optimizer_and_scheduler(
            curr_name,
            self,
            previous_optimizer=self.prev_optim,
            steps_per_epoch=self.trainer.estimated_stepping_batches
        )

        conf = {"optimizer": optimizer}
        if scheduler is None:
            return conf
        scheduler_config = {
            "scheduler": scheduler,
            "monitor": "Validation loss",
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
            "name": "learning_rate",
        }
        if hasattr(scheduler, "stepwise"):
            if scheduler.stepwise:
                scheduler_config["interval"] = "step"

        conf["lr_scheduler"] = scheduler_config
        return conf

    def fit(self, trainer: L.Trainer):
        training_config = self.current_training_config
        trainer.fit()

    @property
    def current_logger(self):
        logger = self.logger_class(
            save_dir=self.log_dir,
            name=self.name,
            version=self.stage_name,
        )
        return logger

    def on_fit_end(self):
        self._tensorboard = None
        self.prev_optim = self.optimizers().optimizer
        self.stage += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.trainer.strategy.barrier()

    def on_save_checkpoint(self, checkpoint) -> None:
        """
        Hook used to store 'stage' attribute in checkpoint.
        """
        checkpoint["stage"] = self.stage
        if hasattr(self.model, "config_dict"):
            checkpoint["model_config"] = self.model.config_dict

    def on_load_checkpoint(self, checkpoint) -> None:
        """
        Sets the 'prev_optim' attribute to avoid issues when training with 'reuse_optimizer' option.
        """
        self.stage = checkpoint["stage"]
        curr_config = copy.copy(self.current_training_config)
        curr_config.reuse_optimizer = False
        self.prev_optim, _ = curr_config.get_optimizer_and_scheduler(
            self.stage_name, self, previous_optimizer=self.prev_optim
        )
