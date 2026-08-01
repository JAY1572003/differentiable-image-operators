import torch


class DiceLoss(torch.nn.Module):
    """Dice loss for binary segmentation. Differentiable."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)


class BCEDiceLoss(torch.nn.Module):
    """
    Combined BCE + Dice loss. Standard combo for binary segmentation.

    pos_weight: crack pixels are ~2-3% of these masks. Without weighting them
    more heavily, BCE can get a deceptively low loss by predicting almost all
    background — this was confirmed experimentally (loss flatlines, IoU -> 0).
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6, pos_weight=6.0):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        pred_clamped = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        weight = target * self.pos_weight + (1.0 - target)
        bce = torch.nn.functional.binary_cross_entropy(pred_clamped, target, weight=weight)
        return self.bce_weight * bce + self.dice_weight * self.dice(pred, target)

def compute_iou(pred_binary, target_binary):
    """IoU between two hard binary masks. For evaluation, not training."""
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_dice_score(pred_binary, target_binary):
    """Dice score between two hard binary masks. For evaluation, not training."""
    intersection = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    if total == 0:
        return 1.0
    return (2.0 * intersection / total).item()