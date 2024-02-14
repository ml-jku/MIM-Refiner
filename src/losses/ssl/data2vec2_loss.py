import torch
from torch import nn

from losses import basic_loss_fn_from_kwargs
from utils.factory import create
from utils.loss_utils import apply_reduction
from kappamodules.functional.patchify import patchify_as_1d
from kappamodules.losses import MSELoss
import torch.nn.functional as F


class Data2vec2Loss(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = create(loss_fn, basic_loss_fn_from_kwargs) or MSELoss()

    def forward(self, prediction, target, mask, reduction="mean"):
        _, num_patches = mask.shape

        # split into aux/patch tokens
        num_aux_tokens = target.size(1) - num_patches
        if num_aux_tokens > 0:
            prediction_aux = prediction[:, :num_aux_tokens]
            prediction_patches = prediction[:, num_aux_tokens:]
            target_aux = target[:, :num_aux_tokens]
            target_patches = target[:, num_aux_tokens:]
        else:
            prediction_aux = None
            prediction_patches = prediction
            target_aux = None
            target_patches = target

        # reduction would require different selection of masked patches
        # current one flattens (bs, seqlen, dim) to (bs * masked_seqlen, dim)
        assert reduction == "mean"
        # select masked targets
        prediction_patches = prediction_patches[mask]
        target_patches = target_patches[mask]
        # patch loss
        loss_patches = self.loss_fn(prediction_patches, target_patches)
        # aux loss
        if target_aux is not None:
            loss_aux = self.loss_fn(prediction_aux, target_aux)
        else:
            loss_aux = None
        return loss_patches, loss_aux
