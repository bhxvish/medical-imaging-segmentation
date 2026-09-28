"""
Milestone 5: combined Dice + BCE loss.

Plain BCE treats every pixel independently. In this dataset most pixels
are background (see the class-imbalance note in dataset.py) -- a model can
reach deceptively low BCE while predicting an almost-empty mask, because
getting ~95% of (mostly background) pixels "right" by predicting nothing
still scores well pixel-by-pixel. Dice loss instead directly measures
overlap between predicted and true mask (2*|A∩B| / (|A|+|B|)), which stays
informative even when the foreground region is small relative to the
image. Combining the two gives Dice's region-overlap signal plus BCE's
stable per-pixel gradient, which is the standard recipe for imbalanced
binary segmentation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss. `pred` is expected to already be sigmoid
    probabilities in [0, 1] (not logits); `target` is a 0/1 mask. Both
    shaped [B, C, H, W] (or broadcastable). Averaged over the batch.
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """loss = bce_weight * BCE(pred, target) + dice_weight * DiceLoss(pred, target)"""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        d_loss = dice_loss(pred, target, smooth=self.smooth)
        return self.bce_weight * bce_loss + self.dice_weight * d_loss
