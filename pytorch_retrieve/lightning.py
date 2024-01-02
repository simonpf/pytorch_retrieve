from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler
from torch import nn
from torchmetrics import Metric
import lightning as L

from pytorch_retrieve.tensors.masked_tensor import MaskedTensor
from pytorch_retrieve.metrics import ScalarMetric

RetrievalInput = Union[torch.Tensor, list, dict]


class LightningRetrieval(L.LightningModule):
    """
    The RetrievalModule implements a LightningModule for the training
    of PytorchRetrieve models.
    """

    def __init__(
        self,
        model: nn.Module,
        name: Optional[str] = None,
        training_schedule: Dict[str, "TrainingConfig"] = None,
        log_dir: Optional[Path] = None,
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
        self.name = name

        self.training_schedule = training_schedule
        if self.training_schedule is None:
            self.stage_names = None
        else:
            self.stage_names = list(self.training_schedule.keys())

        self.stage = 0

        if log_dir is None:
            log_dir = Path(".")
        self.log_dir = log_dir
        self.log_dir = Path(self.log_dir)
        if logger is None:
            logger = L.pytorch.loggers.TensorBoardLogger
        self.logger_class = logger

    @property
    def training_finished(self):
        if self.training_schedule is None:
            return self.stage > 0
        return self.stage >= len(self.training_schedule)

    @property
    def tensorboard(self):
        if self._tensorboard is None:
            self._tensorboard = pl.loggers.TensorBoardLogger(
                self.log_dir, name=self.name + f" ({self.stage_name})"
            )
        return self._tensorboard

    @property
    def stage(self) -> int:
        """
        The index of the current stage.
        """
        return self._stage

    def save_model(self, path: Optional[Path] = None) -> None:
        """
        Save retrieval module as a PyTorch state dict.
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
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "model_config": self.model.to_config_dict(),
            },
            path,
        )

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
        if isinstance(target, dict):
            if len(target) > 1:
                raise RuntimeError(
                    "The model prediction is only a tensor but the target is"
                    " a dict  with more than 1 entry. Therefore, it is not "
                    " clear which  target element the predictions should "
                    " be associated with."
                )
            target = next(iter(target.values()))
        loss = pred.loss(target)
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
            return self.training_step_single_pred(pred, target)

        if not isinstance(target, dict):
            if len(pred) > 1:
                raise RuntimeError(
                    "The model prediction is a dict with more than one "
                    " entry but the target is a single tensor. Therefore, it "
                    " is not clear which target element the predictions should "
                    " be associated with."
                )
            pred = next(iter(pred.values()))
            return self.training_step_single_pred(pred, target)

            raise RuntimeError(
                "If the model output is a 'dict' the reference data must also "
                "be a 'dict'."
            )

        losses = {}
        tot_loss = 0.0
        for name in pred:
            key = name.split("::")[-1]
            y_t = target[key]
            mask = torch.isnan(y_t)
            if mask.any():
                y_t = torch.nan_to_num(y_t, 0.0)
                y_t = MaskedTensor(y_t, mask=mask)
            loss_k = pred[name].loss(y_t)
            tot_loss += loss_k
            losses[name] = loss_k.item()

        log_dict = {}
        for name, loss in losses.items():
            log_dict[f"Training loss ({name})"] = loss
        self.log_dict(log_dict)
        self.log("Training loss", tot_loss)

        losses["loss"] = tot_loss
        return losses

    def on_validation_epoch_start(self):
        self.metrics = self.current_training_config.get_metrics_dict(
            self.model.output_names
        )

    def validation_step_single_pred(
        self, pred: torch.Tensor, target: Union[torch.Tensor, dict]
    ) -> torch.Tensor:
        """
        Validation step for a prediction with just a single output.

        Args:
            pred: A torch.Tensor containing the single prediction from the
                retrieval module.
            target: A torch.Tensor or a dict with a single entry containing
                the target corresponding to the prediction in 'pred'.

        Return:
            The loss of the prediction with respect to the given target.
        """
        if isinstance(target, dict):
            if len(target) > 1:
                raise RuntimeError(
                    "The model prediction is only a tensor but the target is"
                    " a dict  with more than 1 entry. Therefore, it is not "
                    " clear which  target element the predictions should "
                    " be associated with."
                )
            target = next(iter(target.values()))

        metrics = self.metrics
        if len(metrics) > 1:
            raise RuntimeError(
                "Metrics for more than two outputs are defined but "
                "the model provided only a single, non-named output."
            )
            metrics = next(iter(metrics.values()))

        loss = pred.loss(target)

        scalar_metrics = [
            metric for metric in metrics if isinstance(metric, ScalarMetric)
        ]
        scalar_pred = pred.expected_value()
        for metric in scalar_metrics:
            metric.update(scalar_pred, target)

        return loss

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
            return self.validation_step_single_pred(pred, target)

        if not isinstance(target, dict):
            if len(pred) > 1:
                raise RuntimeError(
                    "The model prediction is a dict with more than one "
                    " entry but the target is a single tensor. Therefore, it "
                    " is not clear which target element the predictions should "
                    " be associated with."
                )
            pred = next(iter(pred.values()))
            return self.validation_step_single_pred(pred, target)

            raise RuntimeError(
                "If the model output is a 'dict' the reference data must also "
                "be a 'dict'."
            )

        metrics = self.metrics

        losses = {}
        tot_loss = 0.0
        for name in pred:
            key = name.split("::")[-1]
            y_t = target[key]
            mask = torch.isnan(y_t)
            if mask.any():
                y_t = torch.nan_to_num(y_t, 0.0)
                y_t = MaskedTensor(y_t, mask=mask)

            loss_k = pred[name].loss(y_t)
            tot_loss += loss_k
            losses[name] = loss_k.item()

            if name in metrics:
                metrics_t = metrics[name]
                scalar_metrics = [
                    metric for metric in metrics_t if isinstance(metric, ScalarMetric)
                ]
                pred_t = pred[name].expected_value()
                for metric in scalar_metrics:
                    metric.update(pred_t, y_t)

        log_dict = {}
        for name, loss in losses.items():
            log_dict[f"Validation loss ({name})"] = loss
        self.log_dict(log_dict)
        self.log("Validation loss", tot_loss)

    @L.pytorch.utilities.rank_zero_only
    def on_validation_epoch_end(self):
        for output_name, metrics in self.metrics.items():
            for metric in metrics:
                self.log(
                    metric.name + f" (output_name)",
                    metric.compute(),
                    on_step=False,
                    on_epoch=True,
                )
                metric.reset()

    #     validation_step_output = self.validation_step_outputs

    #     i_epoch = self.trainer.current_epoch
    #     writer = self.tensorboard.experiment

    #     # if self.trainer.is_global_zero:

    #     figures = {}
    #     values = {}

    #     for metric in self.metrics:
    #         # Log values.
    #         if hasattr(metric, "get_values"):
    #             m_values = metric.get_values()
    #             if isinstance(m_values, dict):
    #                 m_values = {
    #                     f"{metric.name} ({key})": value
    #                     for key, value in m_values.items()
    #                 }
    #             else:
    #                 m_values = {metric.name: m_values}

    #             values.update(m_values)

    #         # Log figures.
    #         if hasattr(metric, "get_figures"):
    #             m_figures = metric.get_figures()
    #             if isinstance(m_figures, dict):
    #                 m_figures = {
    #                     f"{metric.name} ({key})": value
    #                     for key, value in m_figures.items()
    #                 }
    #             else:
    #                 m_figures = {metric.name: m_figures}
    #             figures.update(m_figures)

    #     for key, value in values.items():
    #         if isinstance(value, np.ndarray):
    #             values[key] = value.item()

    #     log_scalar = writer.add_scalar
    #     for key, value in values.items():
    #         log_scalar(key, value, i_epoch)

    #     log_image = writer.add_figure
    #     for key, value in figures.items():
    #         log_image(key, value, i_epoch)

    def configure_optimizers(self):
        if self.training_schedule is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            return optimizer

        curr_config = self.current_training_config
        curr_name = self.stage_name

        optimizer, scheduler, _ = curr_config.get_optimizer_and_scheduler(
            curr_name, self
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

    def on_fit_end(self):
        self._tensorboard = None
        self.stage += 1

    def on_save_checkpoint(self, checkpoint) -> None:
        """
        Hook used to store 'stage' attribute in checkpoint.
        """
        checkpoint["stage"] = self.stage

    def on_load_checkpoint(self, checkpoint) -> None:
        """
        Hook used load store 'stage' attribute from checkpoint.
        """
        self.stage = checkpoint["stage"]


# class QuantnnLightning(L.LightningModule):
#     """
#     Pytorch Lightning module for quantnn pytorch models.
#     """

#     def __init__(
#         self,
#         qrnn,
#         loss,
#         name=None,
#         optimizer=None,
#         scheduler=None,
#         metrics=None,
#         mask=None,
#         transformation=None,
#         log_dir=None,
#     ):
#         super().__init__()
#         self.validation_step_outputs = []
#         self.qrnn = qrnn
#         self.model = qrnn.model
#         self.loss = loss
#         self._stage = 0
#         self._stage_name = None

#         self.optimizer = optimizer
#         self.current_optimizer = None
#         self.scheduler = scheduler

#         self.metrics = metrics
#         if self.metrics is None:
#             self.metrics = []
#         for metric in self.metrics:
#             metric.model = self.qrnn
#             metric.mask = mask

#         self.transformation = transformation

#         if log_dir is None:
#             log_dir = "lightning_logs"
#         self.log_dir = log_dir
#         self.name = name
#         self._tensorboard = None
