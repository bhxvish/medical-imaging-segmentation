"""
Milestone 3 deliverable: visually verify image/mask alignment before
trusting the dataset pipeline, plus basic dataset stats.

Run from project root: .venv/Scripts/python.exe scripts/milestone3_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from dicom_seg.dataset import SIIMPneumothoraxDataset, build_manifest, train_val_split
from dicom_seg.preprocessing import dicom_to_image
from dicom_seg.rle import rle_decode_multi

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "siim_subset" / "stage_2_train.csv"
DICOM_DIR = ROOT / "data" / "siim_subset" / "dicom"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def main():
    entries = build_manifest(CSV_PATH, DICOM_DIR)
    n_pos = sum(1 for e in entries if e.has_mask)
    n_neg = sum(1 for e in entries if not e.has_mask)
    print(f"Manifest: {len(entries)} images ({n_pos} positive, {n_neg} negative)")

    train, val = train_val_split(entries, val_fraction=0.2, seed=42)
    print(f"Train: {len(train)} ({sum(e.has_mask for e in train)} positive)")
    print(f"Val:   {len(val)} ({sum(e.has_mask for e in val)} positive)")

    # Pick a few positive examples (most informative for checking alignment)
    # and a couple of negatives, for the overlay grid.
    positives = [e for e in entries if e.has_mask][:4]
    negatives = [e for e in entries if not e.has_mask][:2]
    examples = positives + negatives

    fig, axes = plt.subplots(2, len(examples), figsize=(3 * len(examples), 6))
    for col, entry in enumerate(examples):
        image = dicom_to_image(entry.dicom_path, output_range="unit")
        mask = rle_decode_multi(entry.rle_strings, height=image.shape[0], width=image.shape[1])

        axes[0, col].imshow(image, cmap="gray")
        axes[0, col].set_title(f"{'pos' if entry.has_mask else 'neg'}: {entry.uid[:12]}...",
                                fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(image, cmap="gray")
        overlay = np.ma.masked_where(mask == 0, mask)
        axes[1, col].imshow(overlay, cmap="autumn", alpha=0.5)
        axes[1, col].set_title("+ mask overlay", fontsize=8)
        axes[1, col].axis("off")

    fig.tight_layout()
    out_path = OUT_DIR / "milestone3_mask_overlay.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # Sanity-check the PyTorch Dataset wrapper end to end.
    ds = SIIMPneumothoraxDataset(entries, target_size=256)
    image_t, mask_t = ds[0]
    print(f"Dataset[0]: image {tuple(image_t.shape)} {image_t.dtype}, "
          f"mask {tuple(mask_t.shape)} {mask_t.dtype}, "
          f"mask unique values: {mask_t.unique().tolist()}")


if __name__ == "__main__":
    main()
