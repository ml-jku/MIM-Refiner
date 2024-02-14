import torch
from torch import nn

from losses import basic_loss_fn_from_kwargs
from utils.factory import create
from utils.loss_utils import apply_reduction
from kappamodules.functional.patchify import patchify_as_1d
from kappamodules.losses import MSELoss
import torch.nn.functional as F


class MaeLoss(nn.Module):
    def __init__(self, loss_fn=None, normalize_pixels=False, eps=1e-6):
        super().__init__()
        self.loss_fn = create(loss_fn, basic_loss_fn_from_kwargs) or MSELoss()
        self.normalize_pixels = normalize_pixels
        self.eps = eps

    def forward(self, prediction, target, patch_size, mask, reduction="mean"):
        # prepare target
        patchified_target = patchify_as_1d(x=target, patch_size=patch_size)
        # per-patch normalization (basically a layernorm)
        if self.normalize_pixels:
            patchified_target = F.layer_norm(
                patchified_target,
                normalized_shape=(patchified_target.size(-1),),
                eps=self.eps,
            )

        # unreduced loss
        loss = self.loss_fn(prediction, patchified_target, reduction="none")
        # mean loss per patch
        # [batch_size, num_patches, c*prod(patch_size))] -> [batch_size, num_patches]
        loss = loss.mean(dim=-1)
        # loss on removed patches (mask is 1 if the patch was removed)
        # [batch_size, num_patches] -> [batch_size]
        loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)

        # apply reduction
        loss = apply_reduction(loss, reduction=reduction)
        return loss
