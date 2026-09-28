from pathlib import Path

import numpy as np
import pytest

from dicom_seg.dataset import (
    ManifestEntry,
    SIIMPneumothoraxDataset,
    build_manifest,
    train_val_split,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "siim_subset"
CSV_PATH = DATA_DIR / "stage_2_train.csv"
DICOM_DIR = DATA_DIR / "dicom"

requires_subset = pytest.mark.skipif(
    not (CSV_PATH.exists() and DICOM_DIR.exists() and any(DICOM_DIR.glob("*.dcm"))),
    reason="SIIM-ACR subset not downloaded (data/siim_subset/dicom is empty)",
)


def make_entries(n_pos, n_neg):
    entries = []
    for i in range(n_pos):
        entries.append(ManifestEntry(uid=f"pos{i}", dicom_path=Path("dummy"),
                                      rle_strings=["1 1"], has_mask=True))
    for i in range(n_neg):
        entries.append(ManifestEntry(uid=f"neg{i}", dicom_path=Path("dummy"),
                                      rle_strings=["-1"], has_mask=False))
    return entries


class TestTrainValSplit:
    def test_split_is_stratified_and_covers_all_entries(self):
        entries = make_entries(n_pos=20, n_neg=20)
        train, val = train_val_split(entries, val_fraction=0.2, seed=0)

        assert len(train) + len(val) == len(entries)
        assert set(e.uid for e in train).isdisjoint(set(e.uid for e in val))

        val_pos = sum(1 for e in val if e.has_mask)
        val_neg = sum(1 for e in val if not e.has_mask)
        # both classes represented in the held-out split
        assert val_pos > 0
        assert val_neg > 0

    def test_split_is_reproducible_with_same_seed(self):
        entries = make_entries(n_pos=10, n_neg=10)
        train1, val1 = train_val_split(entries, seed=7)
        train2, val2 = train_val_split(entries, seed=7)
        assert [e.uid for e in val1] == [e.uid for e in val2]


@requires_subset
class TestBuildManifest:
    def test_manifest_only_includes_downloaded_files(self):
        entries = build_manifest(CSV_PATH, DICOM_DIR)
        downloaded_uids = {p.stem for p in DICOM_DIR.glob("*.dcm")}
        assert len(entries) > 0
        assert {e.uid for e in entries} <= downloaded_uids

    def test_manifest_has_mask_matches_rle_content(self):
        entries = build_manifest(CSV_PATH, DICOM_DIR)
        for e in entries[:20]:
            expected = any(s.strip() != "-1" for s in e.rle_strings)
            assert e.has_mask == expected


@requires_subset
class TestSIIMPneumothoraxDataset:
    def test_getitem_shapes_and_dtypes(self):
        entries = build_manifest(CSV_PATH, DICOM_DIR)
        ds = SIIMPneumothoraxDataset(entries[:5], target_size=128)
        image, mask = ds[0]

        assert image.shape == (1, 128, 128)
        assert mask.shape == (1, 128, 128)
        assert image.dtype.is_floating_point
        assert mask.dtype.is_floating_point
        assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
        unique_vals = set(mask.unique().tolist())
        assert unique_vals <= {0.0, 1.0}

    def test_len_matches_entries(self):
        entries = build_manifest(CSV_PATH, DICOM_DIR)
        ds = SIIMPneumothoraxDataset(entries)
        assert len(ds) == len(entries)
