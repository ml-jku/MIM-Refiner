from collections import defaultdict

import numpy as np
import torch
from torchmetrics.functional.classification import multiclass_jaccard_index

from callbacks.base.periodic_callback import PeriodicCallback
from distributed.gather import all_reduce_mean_grad


class OnlineSemsegCallback(PeriodicCallback):
    def __init__(self, verbose=True, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.tracked_mious = defaultdict(list)
        self.num_classes = None

    def _before_training(self, model, trainer, **kwargs):
        self.num_classes = self.data_container.get_dataset("train").getdim("semseg")
        assert 2 < self.num_classes

    def _track_after_accumulation_step(self, update_outputs, trainer, **kwargs):
        target = update_outputs["target"]
        for name, pred in update_outputs["preds"].items():
            acc = multiclass_jaccard_index(
                preds=pred,
                target=target,
                num_classes=self.num_classes,
                average="micro",
                ignore_index=-1,
            )
            self.tracked_mious[name].append(acc)

    def _periodic_callback(self, **_):
        kwargs = dict(logger=self.logger, format_str=".6f") if self.verbose else {}
        for name, tracked_mious in self.tracked_mious.items():
            mean = all_reduce_mean_grad(torch.stack(tracked_mious).mean())
            self.writer.add_scalar(
                key=f"miou/online/{name}/{self.to_short_interval_string()}",
                value=mean,
                **kwargs,
            )
        self.tracked_mious.clear()
