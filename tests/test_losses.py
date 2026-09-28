import torch

from dicom_seg.losses import DiceBCELoss, dice_loss


class TestDiceLoss:
    def test_identical_masks_give_near_zero_loss(self):
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 1:3, 1:3] = 1.0
        pred = target.clone()
        loss = dice_loss(pred, target)
        assert loss.item() < 1e-4

    def test_disjoint_masks_give_loss_near_one(self):
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 0, 0] = 1.0
        pred = torch.zeros(1, 1, 4, 4)
        pred[0, 0, 3, 3] = 1.0
        loss = dice_loss(pred, target)
        assert loss.item() > 0.99

    def test_loss_decreases_as_overlap_increases(self):
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 1:3, 1:3] = 1.0

        pred_no_overlap = torch.zeros(1, 1, 4, 4)
        pred_no_overlap[0, 0, 0, 0] = 1.0

        pred_partial = torch.zeros(1, 1, 4, 4)
        pred_partial[0, 0, 1, 1] = 1.0  # one of the four target pixels

        loss_none = dice_loss(pred_no_overlap, target).item()
        loss_partial = dice_loss(pred_partial, target).item()
        assert loss_partial < loss_none


class TestDiceBCELoss:
    def test_forward_returns_scalar_and_is_finite(self):
        criterion = DiceBCELoss()
        pred = torch.rand(2, 1, 8, 8)
        target = (torch.rand(2, 1, 8, 8) > 0.5).float()
        loss = criterion(pred, target)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradients_flow(self):
        criterion = DiceBCELoss()
        pred = torch.rand(2, 1, 8, 8, requires_grad=True)
        target = (torch.rand(2, 1, 8, 8) > 0.5).float()
        loss = criterion(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)
