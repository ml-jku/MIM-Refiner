import torch
import torch.nn.functional as F
from torch import nn

from distributed.config import get_rank
from distributed.gather import all_gather_grad


class NnclrAllswapLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected, predicted, reduction="mean"):
        assert torch.is_tensor(projected) and projected.ndim == 3
        assert torch.is_tensor(predicted) and predicted.ndim == 3
        assert projected.grad_fn is None
        total_loss = 0.

        # preprocess
        projected = F.normalize(projected, dim=-1)
        predicted = F.normalize(predicted, dim=-1)

        # gather + create labels
        predicted = all_gather_grad(predicted)
        rank = get_rank()
        batch_size = len(projected)
        labels = torch.arange(batch_size * rank, batch_size * (rank + 1), device=predicted.device)

        # global loss
        global_loss = 0.
        num_global_views = projected.size(1)
        assert predicted.size(1) >= num_global_views
        for i in range(num_global_views):
            for ii in range(num_global_views):
                if i == ii:
                    continue
                loss = self._loss(
                    projected=projected[:, i],
                    predicted=predicted[:, ii],
                    labels=labels,
                    reduction=reduction,
                )
                total_loss = total_loss + loss
                global_loss = global_loss + loss
        num_global_terms = num_global_views * (num_global_views - 1)
        if num_global_terms > 0:
            global_loss = global_loss / num_global_terms

        # local loss
        local_loss = 0.
        num_local_views = predicted.size(1) - num_global_views
        for i in range(num_global_views):
            for ii in range(num_local_views):
                loss = self._loss(
                    projected=projected[:, i],
                    predicted=predicted[:, ii],
                    labels=labels,
                    reduction=reduction,
                )
                total_loss = total_loss + loss
                local_loss = local_loss + loss
        num_local_terms = num_local_views * num_global_views
        if num_local_terms > 0:
            local_loss = local_loss / num_local_terms

        # normalize by loss terms (num_terms == 0 if called with single view for evaluation)
        num_terms = num_global_terms + num_local_terms
        if num_terms > 0:
            total_loss = total_loss / num_terms

        # compose infos
        losses = dict(total=total_loss, global_loss=global_loss)
        infos = {}
        if num_local_terms > 0:
            losses["local_loss"] = local_loss
        return losses, infos

    def _loss(self, projected, predicted, labels, reduction):
        logits = projected @ predicted.T / self.temperature
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        return loss
