import torch
from torch import nn

from losses import basic_loss_fn_from_kwargs
from utils.factory import create
from utils.loss_utils import apply_reduction
from kappamodules.functional.patchify import patchify_as_1d
from .basic.mse_loss import MSELoss


class ElementwiseLoss(nn.Module):
    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = create(loss_function, basic_loss_fn_from_kwargs)

    def forward(self, prediction, target, reduction="mean"):
        # unreduced loss
        loss = self.loss_function(prediction, target, reduction="none")
        # apply reduction
        loss = apply_reduction(loss, reduction=reduction)
        return loss
