import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target)
        p = pred.flatten()
        t = target.flatten()
        intersection = (p * t).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            p.sum() + t.sum() + self.smooth
        )
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss
