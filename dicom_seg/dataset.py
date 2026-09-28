"""
Milestone 3: segmentation dataset pipeline for the SIIM-ACR Pneumothorax
subset -- pairs each DICOM image with its decoded mask, resized to a
consistent size for batching.

Class imbalance note: even on a positive (pneumothorax-present) image, the
lesion region is typically a small fraction of total pixels -- most pixels
in a chest X-ray are background/other anatomy, not the region of interest.
Plain per-pixel cross-entropy would let a model get deceptively low loss by
predicting "all background". This dataset doesn't rebalance pixels itself
(that's the loss function's job, see Milestone 5's Dice+BCE combination) --
but it does keep a roughly even split of positive/negative *images* so the
model sees enough real lesions to learn from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import dicom_to_image
from .rle import rle_decode_multi

PathLike = str | Path


@dataclass
class ManifestEntry:
    uid: str
    dicom_path: Path
    rle_strings: List[str]
    has_mask: bool


def build_manifest(csv_path: PathLike, dicom_dir: PathLike) -> List[ManifestEntry]:
    """Pair each locally-downloaded DICOM file with its RLE mask string(s)
    from the labels CSV, keyed by SOPInstanceUID (the DICOM filename stem).

    Only images that actually exist in `dicom_dir` are included, since the
    subset downloaded to disk is a fraction of the full labels CSV.
    """
    dicom_dir = Path(dicom_dir)
    df = pd.read_csv(csv_path)
    df["EncodedPixels"] = df["EncodedPixels"].astype(str).str.strip()

    grouped = df.groupby("ImageId")["EncodedPixels"].apply(list)

    entries = []
    for dicom_path in sorted(dicom_dir.glob("*.dcm")):
        uid = dicom_path.stem
        if uid not in grouped.index:
            continue
        rle_strings = grouped.loc[uid]
        has_mask = any(s != "-1" for s in rle_strings)
        entries.append(ManifestEntry(uid=uid, dicom_path=dicom_path,
                                      rle_strings=rle_strings, has_mask=has_mask))
    return entries


def train_val_split(entries: List[ManifestEntry], val_fraction: float = 0.2,
                     seed: int = 42) -> tuple[List[ManifestEntry], List[ManifestEntry]]:
    """Stratified split by has_mask, so both splits keep a similar
    positive/negative ratio.
    """
    rng = np.random.default_rng(seed)
    positives = [e for e in entries if e.has_mask]
    negatives = [e for e in entries if not e.has_mask]

    def split_group(group):
        idx = rng.permutation(len(group))
        n_val = max(1, int(len(group) * val_fraction)) if group else 0
        val_idx = set(idx[:n_val].tolist())
        train = [g for i, g in enumerate(group) if i not in val_idx]
        val = [g for i, g in enumerate(group) if i in val_idx]
        return train, val

    train_pos, val_pos = split_group(positives)
    train_neg, val_neg = split_group(negatives)
    return train_pos + train_neg, val_pos + val_neg


class SIIMPneumothoraxDataset(Dataset):
    """Returns (image_tensor, mask_tensor), both float32 [1, H, W] in
    [0, 1] / {0, 1}, resized to `target_size`.
    """

    def __init__(self, entries: List[ManifestEntry], target_size: int = 256):
        self.entries = entries
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]

        image = dicom_to_image(entry.dicom_path, output_range="unit")  # (H, W) float32
        mask = rle_decode_multi(entry.rle_strings, height=image.shape[0], width=image.shape[1])

        size = (self.target_size, self.target_size)
        image_resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # Nearest-neighbor for the mask: any smoothing interpolation would
        # produce fractional (non-binary) mask values at edges.
        mask_resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)
        return image_tensor, mask_tensor
