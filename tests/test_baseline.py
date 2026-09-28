import numpy as np

from dicom_seg.baseline import naive_threshold_baseline


class TestNaiveThresholdBaseline:
    def test_output_is_binary(self):
        rng = np.random.default_rng(0)
        image = rng.random((32, 32)).astype(np.float32)
        mask = naive_threshold_baseline(image)
        assert set(np.unique(mask).tolist()) <= {0, 1}

    def test_output_shape_matches_input(self):
        image = np.random.default_rng(1).random((40, 50)).astype(np.float32)
        mask = naive_threshold_baseline(image)
        assert mask.shape == image.shape

    def test_pure_black_background_excluded(self):
        image = np.zeros((10, 10), dtype=np.float32)
        mask = naive_threshold_baseline(image)
        assert mask.sum() == 0

    def test_flags_darker_region_in_bimodal_image(self):
        image = np.full((20, 20), 0.9, dtype=np.float32)
        image[5:10, 5:10] = 0.2  # a clearly darker patch amid bright tissue
        mask = naive_threshold_baseline(image)
        # the dark patch should be (mostly) flagged
        assert mask[5:10, 5:10].mean() > 0.8
        # the bright surrounding area should mostly not be flagged
        assert mask[15:20, 15:20].mean() < 0.2
