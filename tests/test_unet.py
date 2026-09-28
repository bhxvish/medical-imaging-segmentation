import pytest
import torch

from dicom_seg.unet import UNet


class TestUNetShapes:
    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_output_shape_matches_input(self, size):
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, size, size)
        y = model(x)
        assert y.shape == (2, 1, size, size)

    def test_output_is_in_unit_range(self):
        # sigmoid output -> must lie in (0, 1)
        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            y = model(x)
        assert float(y.min()) >= 0.0
        assert float(y.max()) <= 1.0

    def test_multi_channel_input(self):
        model = UNet(in_channels=3, out_channels=1)
        x = torch.randn(1, 3, 128, 128)
        y = model(x)
        assert y.shape == (1, 1, 128, 128)

    def test_smaller_feature_config_for_fast_tests(self):
        model = UNet(in_channels=1, out_channels=1, features=[8, 16])
        x = torch.randn(1, 1, 64, 64)
        y = model(x)
        assert y.shape == (1, 1, 64, 64)

    def test_gradients_flow_end_to_end(self):
        # Cheap check that skip connections/concat didn't break autograd.
        model = UNet(in_channels=1, out_channels=1, features=[8, 16])
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        y = model(x)
        y.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
