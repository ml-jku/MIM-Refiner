from functools import partial

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_accuracy

from callbacks.base.periodic_callback import PeriodicCallback
from metrics.entropy import multiclass_entropy


class OfflineAccuracyCallback(PeriodicCallback):
    def __init__(self, dataset_key, to_cpu=False, topk=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.to_cpu = to_cpu
        self.topk = topk or [1, 5]
        self.__config_id = None
        self.num_classes = None

    def _before_training(self, model, trainer, **kwargs):
        assert len(model.output_shape) == 1
        self.num_classes = self.data_container.get_dataset("train").getdim("class")

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(key=self.dataset_key, mode="x class")

    def _forward(self, batch, model, trainer):
        (x, cls), _ = batch
        x = x.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            predictions = model.classify(x)
        if self.to_cpu:
            predictions = {name: prediction.cpu() for name, prediction in predictions.items()}
        return predictions, cls.clone()

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        predictions, y = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )

        # log
        y = y.to(model.device, non_blocking=True)
        for prediction_name, y_hat in predictions.items():
            if self.to_cpu:
                y_hat = y_hat.to(model.device, non_blocking=True)
            # accuracy
            for topk in self.topk:
                acc = multiclass_accuracy(
                    preds=y_hat,
                    target=y,
                    top_k=topk,
                    num_classes=self.num_classes,
                    average="micro",
                )
                acc_key = f"accuracy{topk}/{self.dataset_key}/{prediction_name}"
                self.writer.add_scalar(acc_key, acc, logger=self.logger, format_str=".6f")
            # loss
            loss = F.cross_entropy(y_hat, y)
            loss_key = f"loss/{self.dataset_key}/{prediction_name}"
            self.writer.add_scalar(loss_key, loss, logger=self.logger)
            # prediction entropy
            # entropy = multiclass_entropy(y_hat=y_hat, y=y, num_classes=self.num_classes)
            # entropy_key = f"entropy/{self.dataset_key}/{prediction_name}"
            # self.writer.add_scalar(entropy_key, entropy)
            # prediction distribution stats
            y_hat_classes = y_hat.argmax(dim=1)
            _, counts = y_hat_classes.unique(return_counts=True, sorted=False)
            samples_per_class_min = counts.min()
            samples_per_class_median = counts.median()
            samples_per_class_max = counts.max()
            key = f"preds_per_class/%s/{self.dataset_key}/{prediction_name}"
            self.writer.add_scalar(key % "min", samples_per_class_min)
            self.writer.add_scalar(key % "median", samples_per_class_median)
            self.writer.add_scalar(key % "max", samples_per_class_max)