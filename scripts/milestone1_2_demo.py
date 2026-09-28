"""
Generates the Milestone 1 & 2 deliverables:
  - printed metadata inspection for a CT and an MR sample file
  - rendered example images proving correct orientation/scaling
  - a lung-window vs bone-window side-by-side on the same CT slice

Uses pydicom's bundled sample DICOM files (no external dataset needed).
Run from the project root: .venv/Scripts/python.exe scripts/milestone1_2_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from pydicom.data import get_testdata_file

from dicom_seg.preprocessing import (
    apply_window,
    describe_metadata,
    dicom_to_image,
    get_modality_corrected,
    load_dicom,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def print_metadata(name: str, ds) -> None:
    print(f"\n=== {name} metadata ===")
    for field, value in describe_metadata(ds).items():
        print(f"  {field:28s} = {value}")


def main():
    ct_ds = load_dicom(get_testdata_file("CT_small.dcm"))
    mr_ds = load_dicom(get_testdata_file("MR_small.dcm"))

    print_metadata("CT_small.dcm", ct_ds)
    print_metadata("MR_small.dcm", mr_ds)

    # --- Milestone 1: correctly oriented/scaled/normalized renders ---
    ct_img = dicom_to_image(ct_ds)
    mr_img = dicom_to_image(mr_ds)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(ct_img, cmap="gray")
    axes[0].set_title("CT_small.dcm (preprocessed)")
    axes[0].axis("off")
    axes[1].imshow(mr_img, cmap="gray")
    axes[1].set_title("MR_small.dcm (preprocessed)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "milestone1_examples.png", dpi=150)
    print(f"\nSaved {OUT_DIR / 'milestone1_examples.png'}")

    # --- Milestone 2: windowing, lung vs bone on the same CT slice ---
    hu = get_modality_corrected(ct_ds)
    lung = apply_window(hu, center=-600, width=1500)
    bone = apply_window(hu, center=400, width=1800)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(lung, cmap="gray")
    axes[0].set_title("Lung window (C=-600, W=1500)")
    axes[0].axis("off")
    axes[1].imshow(bone, cmap="gray")
    axes[1].set_title("Bone window (C=400, W=1800)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "milestone2_windowing.png", dpi=150)
    print(f"Saved {OUT_DIR / 'milestone2_windowing.png'}")


if __name__ == "__main__":
    main()
