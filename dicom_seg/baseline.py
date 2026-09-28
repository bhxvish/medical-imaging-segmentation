"""
Milestone 6: a naive, no-learning baseline to compare the U-Net against.

Otsu's method picks a single global intensity threshold that best
separates an image into two classes, with no training and no notion of
anatomy. Pneumothorax (air trapped outside a collapsed lung) shows up as an
abnormally lucent (dark) region with no lung markings, so "flag the darker
tissue-density pixels" is at least a plausible naive heuristic -- not a
strawman like "predict nothing" -- while still being something a first
image-processing course would produce, with no learned structure at all.
"""
from __future__ import annotations

import cv2
import numpy as np


def naive_threshold_baseline(image: np.ndarray) -> np.ndarray:
    """`image` is a float32 array in [0, 1] (as returned by dicom_to_image).
    Returns a 0/1 uint8 mask flagging pixels darker than the image's Otsu
    threshold, excluding pure-black background (outside the body) so the
    baseline isn't trivially "everything outside the chest".
    """
    img_u8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    otsu_value, _ = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (img_u8 > 0) & (img_u8 <= otsu_value)
    return mask.astype(np.uint8)
