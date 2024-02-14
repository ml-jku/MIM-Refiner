from torch import nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    @staticmethod
    def forward(pred, target, reduction="mean"):
        return F.mse_loss(pred, target, reduction=reduction)
