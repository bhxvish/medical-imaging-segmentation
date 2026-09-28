"""
Milestone 1 & 2: DICOM loading, metadata inspection, intensity correction,
and CT windowing.

Pipeline for turning a raw DICOM file into a model-ready image:

    raw stored pixel values (arbitrary integers, NOT display-ready)
        -> Modality LUT (RescaleSlope/RescaleIntercept, or a LUT Sequence)
           gives real-world intensity units (e.g. Hounsfield Units for CT)
        -> VOI LUT / windowing (WindowCenter/WindowWidth, or a VOI LUT Sequence)
           maps the clinically relevant range onto full display contrast
        -> PhotometricInterpretation correction (MONOCHROME1 is inverted:
           the minimum stored value is meant to display as white)
        -> normalize to [0, 1] or [0, 255]

Getting the last two steps wrong is the classic bug: images render as
photographic negatives (MONOCHROME1 mistake) or with washed-out/blown-out
contrast (skipping rescale/windowing).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pydicom
from pydicom import Dataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut

PathLike = Union[str, Path]

# Fields worth inspecting on every DICOM before touching pixel data, with a
# one-line explanation of what each one means and why it matters.
METADATA_FIELDS = {
    "PatientAge": "Patient age at time of study (e.g. '034Y'). Demographic context only.",
    "Modality": "Acquisition device type: CT, CR/DX (X-ray), MR, US, etc. Determines "
                "what the stored pixel values physically represent.",
    "PhotometricInterpretation": "How stored pixel values map to displayed brightness. "
                "MONOCHROME2: low value = dark (normal). MONOCHROME1: low value = "
                "bright (inverted) -- get this wrong and images look like negatives.",
    "Rows": "Image height in pixels.",
    "Columns": "Image width in pixels.",
    "BitsStored": "Number of bits actually used per pixel sample (e.g. 12 bits packed "
                "into a 16-bit word). Values above 2**BitsStored-1 shouldn't occur.",
    "BitsAllocated": "Number of bits allocated per pixel sample in memory (usually 16).",
    "PixelRepresentation": "0 = unsigned integer pixel values, 1 = signed (two's "
                "complement). CT images are commonly signed to represent negative HU.",
    "RescaleSlope": "Modality LUT slope: real_value = stored_value * slope + intercept. "
                "Converts raw stored integers into physical units (e.g. Hounsfield Units).",
    "RescaleIntercept": "Modality LUT intercept (see RescaleSlope). For CT this is "
                "commonly -1024, since stored values are usually unsigned/offset "
                "while HU air is -1000.",
    "WindowCenter": "Center of the default display window (VOI LUT), in the same units "
                "as the rescaled pixel values.",
    "WindowWidth": "Width of the default display window. Narrower = higher contrast "
                "over a smaller intensity range.",
}


def load_dicom(path: PathLike) -> Dataset:
    """Read a DICOM file from disk."""
    return pydicom.dcmread(str(path))


def describe_metadata(ds: Dataset) -> dict:
    """Return the key metadata fields present on `ds`, with their values.

    Fields not present on this particular file are omitted (DICOM is a
    standard with many optional fields -- not every modality populates
    every tag).
    """
    return {
        field: getattr(ds, field)
        for field in METADATA_FIELDS
        if field in ds
    }


def get_modality_corrected(ds: Dataset) -> np.ndarray:
    """Raw stored pixel values -> real-world intensity units.

    Applies the Modality LUT: either a linear RescaleSlope/RescaleIntercept
    transform, or a Modality LUT Sequence, whichever the file provides. For
    CT this yields Hounsfield Units. For X-ray there usually is no rescale
    (slope=1, intercept=0), so this is close to a no-op but is still the
    correct step to apply before windowing.
    """
    raw = ds.pixel_array.astype(np.float64)
    return apply_modality_lut(raw, ds).astype(np.float64)


def apply_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map the intensity window [center - width/2, center + width/2] to
    [0, 255], clipping everything outside it.

    This is the core windowing operation: CT stores a huge dynamic range
    (air ~ -1000 HU to dense bone ~ +1000-3000 HU) that no display can show
    with good contrast all at once. Picking a narrow window around the
    tissue of interest (e.g. center=-600, width=1500 for lung; center=400,
    width=1800 for bone) throws away irrelevant range in exchange for much
    higher contrast in the range that matters.
    """
    width = max(width, 1e-6)
    lower = center - width / 2.0
    upper = center + width / 2.0
    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / (upper - lower) * 255.0
    return windowed.astype(np.uint8)


def dicom_to_image(
    path_or_ds: Union[PathLike, Dataset],
    apply_voi: bool = True,
    output_range: str = "unit",
) -> np.ndarray:
    """Full Milestone-1 pipeline: DICOM -> correctly oriented, correctly
    scaled, normalized NumPy image.

    Steps:
      1. Read pixel data and apply the Modality LUT (rescale to real units).
      2. Apply the VOI LUT / default window, if present and requested
         (falls back to the raw modality-corrected values otherwise).
      3. Min-max normalize to [0, 1].
      4. Invert if PhotometricInterpretation is MONOCHROME1, so that the
         output always follows the "low value = dark" convention regardless
         of how the source file stored it.

    Parameters
    ----------
    path_or_ds: path to a .dcm file, or an already-loaded pydicom Dataset.
    apply_voi: whether to apply the file's VOI LUT/default window before
        normalizing. Set False to get a plain min-max normalization of the
        modality-corrected values instead (useful when you'll window
        yourself, e.g. for the lung/bone comparison in Milestone 2).
    output_range: "unit" for float32 in [0, 1], or "uint8" for [0, 255].
    """
    ds = path_or_ds if isinstance(path_or_ds, Dataset) else load_dicom(path_or_ds)

    modality_arr = get_modality_corrected(ds)
    arr = apply_voi_lut(modality_arr, ds) if apply_voi else modality_arr
    arr = arr.astype(np.float64)

    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        norm = (arr - lo) / (hi - lo)
    else:
        norm = np.zeros_like(arr)

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        norm = 1.0 - norm

    if output_range == "uint8":
        return (norm * 255.0).astype(np.uint8)
    return norm.astype(np.float32)
