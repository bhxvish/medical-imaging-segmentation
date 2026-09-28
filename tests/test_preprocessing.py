import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from dicom_seg.preprocessing import (
    apply_window,
    describe_metadata,
    dicom_to_image,
    get_modality_corrected,
    load_dicom,
)


def make_synthetic_dicom(photometric: str, rows=4, cols=4, pixel_values=None) -> Dataset:
    """Build a minimal-but-valid in-memory DICOM dataset for controlled tests."""
    if pixel_values is None:
        pixel_values = np.linspace(0, 255, rows * cols, dtype=np.uint8).reshape(rows, cols)

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta

    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_values.astype(np.uint8).tobytes()
    return ds


class TestMetadata:
    def test_describe_metadata_extracts_known_fields(self):
        ds = load_dicom(get_testdata_file("CT_small.dcm"))
        meta = describe_metadata(ds)
        assert meta["Modality"] == "CT"
        assert meta["PhotometricInterpretation"] == "MONOCHROME2"
        assert meta["RescaleIntercept"] == -1024
        assert meta["Rows"] == ds.Rows

    def test_describe_metadata_omits_absent_fields(self):
        ds = load_dicom(get_testdata_file("MR_small.dcm"))
        meta = describe_metadata(ds)
        # MR_small.dcm has no RescaleSlope/Intercept -- must not be present.
        assert "RescaleSlope" not in meta or ds.get("RescaleSlope") is not None


class TestModalityCorrection:
    def test_ct_rescale_to_hounsfield_units(self):
        ds = load_dicom(get_testdata_file("CT_small.dcm"))
        raw = ds.pixel_array.astype(np.float64)
        hu = get_modality_corrected(ds)
        # RescaleSlope=1, RescaleIntercept=-1024 for this file.
        np.testing.assert_allclose(hu, raw * 1 + (-1024))

    def test_no_rescale_is_near_identity(self):
        ds = load_dicom(get_testdata_file("MR_small.dcm"))
        raw = ds.pixel_array.astype(np.float64)
        corrected = get_modality_corrected(ds)
        np.testing.assert_allclose(corrected, raw)


class TestWindowing:
    def test_apply_window_hand_computed(self):
        image = np.array([[-1000.0, -600.0, -200.0, 200.0]])
        # center=-600, width=1500 -> lower=-1350, upper=150
        windowed = apply_window(image, center=-600, width=1500)
        lower, upper = -1350.0, 150.0
        expected = np.clip(image, lower, upper)
        expected = ((expected - lower) / (upper - lower) * 255.0).astype(np.uint8)
        np.testing.assert_array_equal(windowed, expected)

    def test_apply_window_clips_outside_range(self):
        image = np.array([[-2000.0, 5000.0]])
        windowed = apply_window(image, center=0, width=400)
        assert windowed[0, 0] == 0
        assert windowed[0, 1] == 255

    def test_lung_vs_bone_window_differ(self):
        ds = load_dicom(get_testdata_file("CT_small.dcm"))
        hu = get_modality_corrected(ds)
        lung = apply_window(hu, center=-600, width=1500)
        bone = apply_window(hu, center=400, width=1800)
        assert not np.array_equal(lung, bone)


class TestPhotometricInterpretation:
    def test_monochrome1_is_inverted_relative_to_monochrome2(self):
        values = np.array([[0, 85], [170, 255]], dtype=np.uint8)
        ds_m2 = make_synthetic_dicom("MONOCHROME2", rows=2, cols=2, pixel_values=values)
        ds_m1 = make_synthetic_dicom("MONOCHROME1", rows=2, cols=2, pixel_values=values)

        img_m2 = dicom_to_image(ds_m2, apply_voi=False, output_range="unit")
        img_m1 = dicom_to_image(ds_m1, apply_voi=False, output_range="unit")

        # Same stored values, opposite photometric interpretation ->
        # MONOCHROME1 output should be the complement of MONOCHROME2's.
        np.testing.assert_allclose(img_m1, 1.0 - img_m2, atol=1e-6)

    def test_monochrome1_min_stored_value_is_brightest(self):
        values = np.array([[0, 128], [200, 255]], dtype=np.uint8)
        ds = make_synthetic_dicom("MONOCHROME1", rows=2, cols=2, pixel_values=values)
        img = dicom_to_image(ds, apply_voi=False, output_range="unit")
        # stored 0 (the minimum) must map to the brightest output pixel (1.0)
        assert img[0, 0] == pytest.approx(1.0)
        # stored 255 (the maximum) must map to the darkest output pixel (0.0)
        assert img[1, 1] == pytest.approx(0.0)


class TestDicomToImage:
    def test_output_shape_matches_rows_columns(self):
        ds = load_dicom(get_testdata_file("CT_small.dcm"))
        img = dicom_to_image(ds)
        assert img.shape == (ds.Rows, ds.Columns)

    def test_output_range_is_normalized(self):
        ds = load_dicom(get_testdata_file("MR_small.dcm"))
        img = dicom_to_image(ds, output_range="unit")
        assert img.dtype == np.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_uint8_output_range(self):
        ds = load_dicom(get_testdata_file("CT_small.dcm"))
        img = dicom_to_image(ds, output_range="uint8")
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255
