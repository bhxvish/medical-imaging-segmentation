"""
Milestone 6 (baseline half): evaluate the naive Otsu-thresholding baseline
against ground truth on the SIIM-ACR subset, with no trained model
involved. The U-Net's Dice/IoU (once trained on Colab) goes in the same
table in BENCHMARKS.md for direct comparison.

Run from project root: .venv/Scripts/python.exe scripts/evaluate_baseline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from dicom_seg.baseline import naive_threshold_baseline
from dicom_seg.dataset import build_manifest, train_val_split
from dicom_seg.metrics import dice_coefficient, iou_score
from dicom_seg.preprocessing import dicom_to_image
from dicom_seg.rle import rle_decode_multi

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "siim_subset" / "stage_2_train.csv"
DICOM_DIR = ROOT / "data" / "siim_subset" / "dicom"


def main():
    entries = build_manifest(CSV_PATH, DICOM_DIR)
    _, val_entries = train_val_split(entries, val_fraction=0.2, seed=42)
    print(f"Evaluating naive baseline on {len(val_entries)} validation images "
          f"({sum(e.has_mask for e in val_entries)} positive)")

    dice_scores, iou_scores = [], []
    dice_pos, iou_pos = [], []

    for entry in val_entries:
        image = dicom_to_image(entry.dicom_path, output_range="unit")
        target = rle_decode_multi(entry.rle_strings, height=image.shape[0], width=image.shape[1])
        pred = naive_threshold_baseline(image)

        d = dice_coefficient(pred, target)
        i = iou_score(pred, target)
        dice_scores.append(d)
        iou_scores.append(i)
        if entry.has_mask:
            dice_pos.append(d)
            iou_pos.append(i)

    print(f"\nNaive Otsu-threshold baseline (all {len(val_entries)} val images):")
    print(f"  Mean Dice: {np.mean(dice_scores):.4f}")
    print(f"  Mean IoU:  {np.mean(iou_scores):.4f}")
    print(f"\nPositive-only images ({len(dice_pos)}), the ones that actually matter clinically:")
    print(f"  Mean Dice: {np.mean(dice_pos):.4f}")
    print(f"  Mean IoU:  {np.mean(iou_pos):.4f}")


if __name__ == "__main__":
    main()
