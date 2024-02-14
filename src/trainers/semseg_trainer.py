from torch import nn
import torch.nn.functional as F

from callbacks.online_callbacks.online_semseg_callback import OnlineSemsegCallback
from utils.object_from_kwargs import objects_from_kwargs
from .base.sgd_trainer import SgdTrainer


class SemsegTrainer(SgdTrainer):
    def get_trainer_callbacks(self, model=None):
        return [
            OnlineSemsegCallback(
                verbose=False,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            OnlineSemsegCallback(
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @property
    def output_shape(self):
        assert len(self.input_shape) == 3
        return self.data_container.get_dataset("train").getdim("semseg"), *self.input_shape[1:]

    @property
    def dataset_mode(self):
        return "index x semseg"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def forward(self, batch, reduction="mean"):
            # prepare data
            (idx, x, target), ctx = batch
            x = x.to(self.model.device, non_blocking=True)
            target = target.to(self.model.device, non_blocking=True)

            # forward
            preds = self.model(x)

            # calculate loss
            losses = {
                name: F.cross_entropy(pred, target, reduction=reduction, ignore_index=-1)
                for name, pred in preds.items()
            }
            losses["total"] = sum(losses.values())

            # compose outputs (for callbacks to use)
            outputs = {
                "idx": idx,
                "preds": preds,
                "target": target,
            }
            yield losses, outputs
