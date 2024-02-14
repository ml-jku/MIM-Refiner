import torch
import torch.nn.functional as F
from torch import nn

from distributed.config import get_rank
from distributed.gather import all_gather_grad
from utils.loss_utils import apply_reduction


class NnclrLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, projected_preswap, projected_postswap, predicted, reduction="mean"):
        assert torch.is_tensor(projected_preswap) and projected_preswap.ndim == 3
        assert torch.is_tensor(projected_postswap) and projected_postswap.ndim == 3
        assert torch.is_tensor(predicted) and predicted.ndim == 3
        total_loss = 0.

        # preprocess
        projected_preswap = F.normalize(projected_preswap, dim=-1)
        projected_postswap = F.normalize(projected_postswap, dim=-1)
        predicted = F.normalize(predicted, dim=-1)

        # gather + create labels
        predicted = all_gather_grad(predicted)
        rank = get_rank()
        batch_size = len(projected_preswap)
        labels = torch.arange(batch_size * rank, batch_size * (rank + 1), device=predicted.device)

        # global loss
        global_loss = 0.
        global_alignment = 0.
        num_global_views = projected_preswap.size(1)
        assert predicted.size(1) >= num_global_views
        for i in range(num_global_views):
            for ii in range(num_global_views):
                if i == ii:
                    continue
                loss, alignment = self._loss(
                    projected_preswap=projected_preswap[:, i],
                    projected_postswap=projected_postswap[:, i],
                    predicted=predicted[:, ii],
                    labels=labels,
                    reduction=reduction,
                )
                total_loss = total_loss + loss
                global_loss = global_loss + loss
                global_alignment = global_alignment + alignment
        num_global_terms = num_global_views * (num_global_views - 1)
        if num_global_terms > 0:
            global_loss = global_loss / num_global_terms
            global_alignment = global_alignment / num_global_terms

        # local loss
        local_loss = 0.
        local_alignment = 0.
        num_local_views = predicted.size(1) - num_global_views
        for i in range(num_global_views):
            for ii in range(num_local_views):
                loss, alignment = self._loss(
                    projected_preswap=projected_preswap[:, i],
                    projected_postswap=projected_postswap[:, i],
                    predicted=predicted[:, ii],
                    labels=labels,
                    reduction=reduction,
                )
                total_loss = total_loss + loss
                local_loss = local_loss + loss
                local_alignment = local_alignment + alignment
        num_local_terms = num_local_views * num_global_views
        if num_local_terms > 0:
            local_loss = local_loss / num_local_terms
            local_alignment = local_alignment / num_local_terms

        # normalize by loss terms (num_terms == 0 if called with single view for evaluation)
        num_terms = num_global_terms + num_local_terms
        if num_terms > 0:
            total_loss = total_loss / num_terms

        # compose infos
        losses = dict(total=total_loss, global_loss=global_loss)
        infos = dict(global_alignment=global_alignment)
        if num_local_terms > 0:
            losses["local_loss"] = local_loss
            infos["local_alignment"] = local_alignment
        return losses, infos

    def _loss(self, projected_preswap, projected_postswap, predicted, labels, reduction):
        logits = projected_postswap @ predicted.T / self.temperature
        # don't swap negatives -> replace off-diagonal with preswap
        preswap_logits = projected_preswap @ predicted.T / self.temperature
        mask = F.one_hot(labels, num_classes=preswap_logits.shape[1])
        logits = logits * mask + preswap_logits * (1 - mask)
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        # alignment
        with torch.no_grad():
            alignment = torch.gather(logits.softmax(dim=-1), index=labels.unsqueeze(1), dim=1)
        # apply reduction
        return loss, apply_reduction(alignment, reduction=reduction)
