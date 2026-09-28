import numpy as np
import torch

from dicom_seg.gradcam import SegGradCAM
from dicom_seg.unet import UNet


def make_model():
    # Small feature config: this test only checks mechanics (hooks, shapes,
    # gradient flow), not segmentation quality, so a tiny/untrained model
    # is enough and keeps the test fast.
    return UNet(in_channels=1, out_channels=1, features=[8, 16])


class TestSegGradCAM:
    def test_cam_output_shape_matches_input(self):
        model = make_model()
        cam_tool = SegGradCAM(model, target_layer=model.bottleneck)
        x = torch.randn(1, 1, 64, 64)
        cam, probs = cam_tool(x)
        assert cam.shape == (64, 64)
        assert probs.shape == (1, 1, 64, 64)

    def test_cam_values_in_unit_range(self):
        model = make_model()
        cam_tool = SegGradCAM(model, target_layer=model.bottleneck)
        x = torch.randn(1, 1, 64, 64)
        cam, _ = cam_tool(x)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0 + 1e-6

    def test_works_with_region_of_interest(self):
        model = make_model()
        cam_tool = SegGradCAM(model, target_layer=model.bottleneck)
        x = torch.randn(1, 1, 64, 64)
        region = torch.zeros(1, 1, 64, 64)
        region[0, 0, 20:40, 20:40] = 1.0
        cam, _ = cam_tool(x, region=region)
        assert cam.shape == (64, 64)
        assert np.isfinite(cam).all()

    def test_works_with_encoder_layer(self):
        model = make_model()
        cam_tool = SegGradCAM(model, target_layer=model.encoders[-1])
        x = torch.randn(1, 1, 64, 64)
        cam, _ = cam_tool(x)
        assert cam.shape == (64, 64)

    def test_does_not_mutate_model_weights(self):
        model = make_model()
        cam_tool = SegGradCAM(model, target_layer=model.bottleneck)
        before = [p.clone() for p in model.parameters()]
        x = torch.randn(1, 1, 64, 64)
        cam_tool(x)
        after = list(model.parameters())
        for b, a in zip(before, after):
            assert torch.equal(b, a)
