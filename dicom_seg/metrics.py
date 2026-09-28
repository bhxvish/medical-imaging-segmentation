"""
Milestone 6: Dice coefficient and IoU (Jaccard index) for binary
segmentation evaluation -- the "hard" (thresholded) counterparts to the
soft Dice loss used for training.

Dice = 2*|A∩B| / (|A|+|B|)   (equivalent to the F1 score for segmentation)
IoU  = |A∩B| / |A∪B|          (stricter: penalizes false positives/negatives
                                more than Dice does, since the denominator
                                doesn't double-count the intersection)
"""
from __future__ import annotations

import numpy as np
import torch


def _to_binary_numpy(mask, threshold: float = 0.5) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    return (mask >= threshold).astype(np.uint8)


def dice_coefficient(pred, target, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """`pred` may be probabilities (thresholded at `threshold`) or already
    a 0/1 mask. `target` is a 0/1 (or already-binary) mask. Computed over
    all elements passed in (call per-sample for a per-image score).
    """
    pred_bin = _to_binary_numpy(pred, threshold)
    target_bin = _to_binary_numpy(target, threshold=0.5)

    intersection = np.logical_and(pred_bin, target_bin).sum()
    total = pred_bin.sum() + target_bin.sum()
    if total == 0:
        return 1.0  # both empty: perfect agreement, not undefined
    return float((2.0 * intersection + smooth) / (total + smooth))


def iou_score(pred, target, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred_bin = _to_binary_numpy(pred, threshold)
    target_bin = _to_binary_numpy(target, threshold=0.5)

    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    if union == 0:
        return 1.0  # both empty: perfect agreement, not undefined
    return float((intersection + smooth) / (union + smooth))
