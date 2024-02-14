from collections import defaultdict

import torch

from callbacks.base.periodic_callback import PeriodicCallback
from distributed.gather import all_reduce_mean_nograd


class OnlineLossCallback(PeriodicCallback):
    def __init__(self, verbose, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.tracked_losses = defaultdict(list)

    def _track_after_accumulation_step(self, losses, **kwargs):
        for name, loss in losses.items():
            self.tracked_losses[name].append(loss.detach())

    def _periodic_callback(self, **_):
        for name, tracked_loss in self.tracked_losses.items():
            mean_loss = all_reduce_mean_nograd(torch.stack(tracked_loss).mean())
            self.writer.add_scalar(
                key=f"loss/online/{name}/{self.to_short_interval_string()}",
                value=mean_loss,
                logger=self.logger if self.verbose else None,
            )
        self.tracked_losses.clear()
