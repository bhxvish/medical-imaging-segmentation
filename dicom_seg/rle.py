"""
RLE (run-length encoding) mask decode/encode for the SIIM-ACR Pneumothorax
dataset.

Faithful re-implementation of Kaggle's official `mask_functions.py`
(`rle2mask`/`mask2rle`), which encodes runs while iterating pixels
column-by-column (outer loop over width, inner loop over height) and treats
each "start" value as a GAP length from the end of the previous run, not an
absolute pixel index (i.e. this is a delta-encoded RLE, not the
absolute-offset RLE used by e.g. COCO).

Because of that column-major traversal, decoding naively reshapes to
(width, height) -- the transpose to (height, width), matching a normal
image array's (rows, cols) axes, is applied here so callers get masks
already aligned with `dicom_to_image` output. Forgetting this transpose is
the single most common bug reported against this exact dataset.
"""
from __future__ import annotations

import numpy as np

EMPTY_MASK_TOKEN = "-1"


def rle_decode(rle_string: str, height: int, width: int) -> np.ndarray:
    """Decode one SIIM-style RLE string into a (height, width) uint8 mask
    of 0/1 values. Returns an all-zero mask for the "no pneumothorax"
    sentinel ("-1") or an empty string.
    """
    rle_string = rle_string.strip()
    if rle_string == "" or rle_string == EMPTY_MASK_TOKEN:
        return np.zeros((height, width), dtype=np.uint8)

    values = np.asarray([int(x) for x in rle_string.split()], dtype=np.int64)
    starts = values[0::2]
    lengths = values[1::2]

    flat = np.zeros(width * height, dtype=np.uint8)
    position = 0
    for start, length in zip(starts, lengths):
        position += int(start)
        flat[position:position + int(length)] = 1
        position += int(length)

    mask = flat.reshape(width, height)
    return mask.T


def rle_decode_multi(rle_strings: list[str], height: int, width: int) -> np.ndarray:
    """Decode and OR-combine multiple RLE strings for the same image (some
    images have several disconnected pneumothorax regions, one CSV row
    each).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for rle_string in rle_strings:
        mask |= rle_decode(rle_string, height, width)
    return mask


def rle_encode(mask: np.ndarray) -> str:
    """Encode a (height, width) 0/1 (or 0/255) mask back into a SIIM-style
    delta RLE string. Inverse of `rle_decode` for a single connected/simple
    mask; mainly used for round-trip testing.
    """
    width, height = mask.shape[1], mask.shape[0]
    binary = (mask.T.flatten() > 0).astype(np.uint8)  # column-major flatten via (width,height)

    if not binary.any():
        return EMPTY_MASK_TOKEN

    pairs = []
    last_end = 0  # position just after the previous run
    in_run = False
    run_start = 0
    for i, v in enumerate(binary):
        if v == 1 and not in_run:
            run_start = i
            in_run = True
        elif v == 0 and in_run:
            pairs.append((run_start - last_end, i - run_start))
            last_end = i
            in_run = False
    if in_run:
        pairs.append((run_start - last_end, len(binary) - run_start))

    return " ".join(f"{gap} {length}" for gap, length in pairs)
