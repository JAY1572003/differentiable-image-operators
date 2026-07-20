import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6, pos_weight: float = 6.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Crack pixels are rare (often <5% of the image), so without extra
        # weight the model can "cheat" by shrinking its output toward all-
        # background and still get a low loss. pos_weight makes mistakes on
        # crack pixels count more, so shrinking to nothing is no longer cheap.
        weight = target * self.pos_weight + (1.0 - target)

        # clamp to avoid log(0) since our operators can output exact 0/1
        pred_clamped = pred.clamp(min=1e-6, max=1 - 1e-6)
        bce = F.binary_cross_entropy(pred_clamped, target, weight=weight)

        p = pred.flatten()
        t = target.flatten()
        intersection = (p * t).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            p.sum() + t.sum() + self.smooth
        )
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss