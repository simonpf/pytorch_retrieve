"""
pytorch_retrieve.lightning

"""
import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler
from torch import nn
from torchmetrics import Metric
import lightning as L
from lightning.pytorch import loggers

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
        self.name = name

        self.training_schedule = training_schedule
        if self.training_schedule is None:
            self.stage_names = None
        else:
            self.stage_names = list(self.training_schedule.keys())

        self.stage = 0

        if model_dir is None:
            model_dir = Path(".")
        self.model_dir = model_dir
        self.log_dir = Path(model_dir) / "logs"
        if logger is None:
            logger = loggers.TensorBoardLogger
        self.logger_class = logger

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

        if isinstance(pred, list):
            if not isinstance(target, list):
                raise RuntimeError(
                    "Model predicts a sequence but the reference data is not."
                )
            loss = torch.tensor(0.0)
            for pred_s, target_s in zip(pred, target):
                loss += pred_s.loss(target_s)
            return loss

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

            pred_k = pred[key]
            target_k = target[key]

            if isinstance(pred_k, list):
                if not isinstance(target_k, list):
                    raise RuntimeError(
                        "Model predicts a sequence but the reference data is not."
                    )

                losses[key] = 0.0
                for pred_k_s, target_k_s in zip(pred_k, target_k):
                    mask = torch.isnan(target_k_s)
                    if mask.any():
                        target_k_s = torch.nan_to_num(target_k_s, 0.0)
                        target_k_s = MaskedTensor(target_k_s, mask=mask)

                    loss_k_s = pred_k_s.loss(target_k_s)
                    tot_loss += loss_k_s
                    losses[name] += loss_k_s.item()

            else:
                mask = torch.isnan(target_k)
                if mask.any():
                    target_k = torch.nan_to_num(target_k, 0.0)
                    target_k = MaskedTensor(target_k, mask=mask)

                loss_k = pred[name].loss(target_k)
                tot_loss += loss_k
                losses[name] = loss_k.item()

        log_dict = {}
        for name, loss in losses.items():
            log_dict[f"Training loss ({name})"] = loss
        self.log_dict(log_dict)
        self.log("Training loss", tot_loss)

        losses["loss"] = tot_loss
        return losses

    def on_train_start(self):
        self.metrics = self.current_training_config.get_metrics_dict(
            self.model.output_names
        )

    def validation_step_single_sequence(
        self, pred: List[torch.Tensor], target: Union[List[torch.Tensor], dict]
    ) -> torch.Tensor:
        """
        Validation step for a sequence prediction with a single output.
        """
        if isinstance(target, dict):
            if len(target) > 1:
                raise RuntimeError(
                    "The model prediction is a single, unnamed sequence "
                    " but the target is a dict  with more than 1 entry. "
                    " Therefore, it is not clear which  target element "
                    " the predictions should be associated with."
                )
            target = next(iter(target.values()))

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

        loss = 0.0
        for pred_s, loss_s in zip(pred, target):
            loss += pred_s.loss(target_)

            scalar_pred = pred_s.expected_value()
            for metric in scalar_metrics:
                metric = metric.to(device=scalar_pred.device)
                metric.update(scalar_pred, target_s)

        return loss

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
                "the model provided only a single, unnamed output."
            )
        metrics = next(iter(metrics.values()))

        loss = pred.loss(target)

        scalar_metrics = [
            metric for metric in metrics if isinstance(metric, ScalarMetric)
        ]
        scalar_pred = pred.expected_value()
        for metric in scalar_metrics:
            metric = metric.to(device=scalar_pred.device)
            metric.update(scalar_pred, target)

        other_metrics = [metric for metric in metrics if not isinstance(metric, ScalarMetric)]
        for metric in other_metrics:
            metric = metric.to(device=scalar_pred.device)
            metric.update(pred, target)

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
        for name, pred_k in pred.items():
            key = name.split("::")[-1]
            target_k = target[key]
            losses[name] = 0.0

            # Determine scalar metrics for this outout
            metrics_k = metrics.get(name, [])
            scalar_metrics = [
                metric
                for metric in metrics_k
                if isinstance(metric, ScalarMetric)
            ]
            # Other metrics
            other_metrics = [metric for metric in metrics_k if not isinstance(metric, ScalarMetric)]

            if isinstance(pred_k, list):
                for pred_k_s, target_k_s in zip(pred_k, target_k):
                    mask = torch.isnan(target_k_s)
                    if mask.any():
                        #target_k_s = torch.nan_to_num(target_k_s, 0.0)
                        target_k_s = MaskedTensor(target_k_s, mask=mask)

                    loss_k_s = pred_k_s.loss(target_k_s)
                    tot_loss += loss_k_s
                    losses[name] += loss_k_s.item()

                    if len(scalar_metrics) > 0:
                        pred_k_s = pred_k_s.expected_value()
                        for metric in scalar_metrics:
                            metric = metric.to(device=pred_k_s.device)
                            metric.update(pred_k_s, target_k_s)

                for metric in other_metrics:
                    metric = metric.to(device=pred_k_s.device)
                    metric.update(pred_k, target_k)


            else:
                mask = torch.isnan(target_k)
                if mask.any():
                    #target_k = torch.nan_to_num(target_k, 0.0)
                    target_k = MaskedTensor(target_k, mask=mask)

                loss_k = pred_k.loss(target_k)
                tot_loss += loss_k
                losses[name] += loss_k.item()

                for metric in other_metrics:
                    metric = metric.to(device=pred_k.device)
                    metric.update(pred_k, target_k)

                if len(scalar_metrics) > 0:
                    pred_k = pred_k.expected_value()
                    for metric in scalar_metrics:
                        metric = metric.to(device=pred_k.device)
                        metric.update(pred_k, target_k)

        log_dict = {}
        for name, loss in losses.items():
            log_dict[f"Validation loss ({name})"] = loss
        self.log_dict(log_dict)
        self.log("Validation loss", tot_loss)

    def on_validation_epoch_end(self):
        for output_name, metrics in self.metrics.items():
            for metric in metrics:
                metric = metric.to(self.device)
                metric.log(self, output_name=output_name)
                metric.reset()

        self.log(
            "Learning rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
        )

    def configure_optimizers(self):
        if self.training_schedule is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            return optimizer

        curr_config = self.current_training_config
        curr_name = self.stage_name

        # If set, use 'lr' attribute. This is done to support the automatic
        # LR search provided by Lightning.
        if hasattr(self, "lr"):
            curr_config = copy.copy(curr_config)
            curr_config.optimizer_args["lr"] = self.lr

        optimizer, scheduler = curr_config.get_optimizer_and_scheduler(curr_name, self)

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
