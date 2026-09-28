import numpy as np

from dicom_seg.rle import rle_decode, rle_decode_multi, rle_encode


class TestRleDecode:
    def test_hand_constructed_example(self):
        # Hand-traced against the official SIIM rle2mask semantics (delta-encoded
        # starts, column-major width/height traversal) for the mask (height=2,
        # width=3):
        #   [[0, 1, 1],
        #    [0, 0, 1]]
        # which column-major-flattens (for x in width: for y in height) to
        # [0, 0, 1, 0, 1, 1] -> runs: gap=2,len=1 then gap=1,len=2 -> "2 1 1 2".
        decoded = rle_decode("2 1 1 2", height=2, width=3)
        expected = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(decoded, expected)

    def test_no_mask_sentinel(self):
        decoded = rle_decode("-1", height=10, width=10)
        assert decoded.shape == (10, 10)
        assert decoded.sum() == 0

    def test_empty_string(self):
        decoded = rle_decode("", height=5, width=5)
        assert decoded.sum() == 0

    def test_single_run_middle(self):
        # width=4, height=1 -> column-major traversal == plain 1D here.
        # gap=2, length=2 -> pixels 2,3 set.
        decoded = rle_decode("2 2", height=1, width=4)
        np.testing.assert_array_equal(decoded, np.array([[0, 0, 1, 1]], dtype=np.uint8))

    def test_output_dtype_and_binary(self):
        decoded = rle_decode("2 1 1 2", height=2, width=3)
        assert decoded.dtype == np.uint8
        assert set(np.unique(decoded).tolist()) <= {0, 1}


class TestRleDecodeMulti:
    def test_combines_disjoint_regions(self):
        # Two separate single-run RLEs on a 1x6 image, non-overlapping. Each
        # RLE string is decoded independently from position 0.
        rle_a = "0 1"   # pixel 0
        rle_b = "5 1"   # gap 5 from position 0 -> pixel 5
        combined = rle_decode_multi([rle_a, rle_b], height=1, width=6)
        expected = np.array([[1, 0, 0, 0, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(combined, expected)

    def test_empty_list_gives_empty_mask(self):
        combined = rle_decode_multi([], height=4, width=4)
        assert combined.sum() == 0


class TestRleRoundTrip:
    def test_encode_decode_round_trip(self):
        # A mask that doesn't touch the very last flattened pixel, so it's
        # unaffected by the official format's "un-flushed trailing run" quirk.
        mask = np.array(
            [[0, 1, 1, 0],
             [0, 0, 1, 0],
             [1, 1, 0, 0]],
            dtype=np.uint8,
        )
        rle_string = rle_encode(mask)
        decoded = rle_decode(rle_string, height=mask.shape[0], width=mask.shape[1])
        np.testing.assert_array_equal(decoded, mask)

    def test_all_zero_mask_round_trip(self):
        mask = np.zeros((3, 3), dtype=np.uint8)
        rle_string = rle_encode(mask)
        assert rle_string == "-1"
        decoded = rle_decode(rle_string, height=3, width=3)
        np.testing.assert_array_equal(decoded, mask)
