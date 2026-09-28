import numpy as np
import pytest
import torch

from dicom_seg.metrics import dice_coefficient, iou_score


class TestDiceCoefficient:
    def test_identical_masks_give_dice_one(self):
        mask = np.array([[0, 1, 1], [0, 0, 1]])
        assert dice_coefficient(mask, mask) == pytest.approx(1.0)

    def test_disjoint_masks_give_dice_zero(self):
        a = np.array([[1, 0], [0, 0]])
        b = np.array([[0, 0], [0, 1]])
        assert dice_coefficient(a, b) == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap_hand_computed(self):
        # A = {(0,0),(0,1)}, B = {(0,1),(1,1)} -> intersection=1, |A|=2,|B|=2
        # Dice = 2*1 / (2+2) = 0.5
        a = np.array([[1, 1], [0, 0]])
        b = np.array([[0, 1], [0, 1]])
        assert dice_coefficient(a, b) == pytest.approx(0.5, abs=1e-4)

    def test_both_empty_masks_give_dice_one(self):
        a = np.zeros((3, 3))
        b = np.zeros((3, 3))
        assert dice_coefficient(a, b) == pytest.approx(1.0)

    def test_accepts_torch_tensor_and_probabilities(self):
        target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        pred_probs = torch.tensor([[0.9, 0.8], [0.1, 0.2]])  # thresholds to same as target
        assert dice_coefficient(pred_probs, target) == pytest.approx(1.0)


class TestIoUScore:
    def test_identical_masks_give_iou_one(self):
        mask = np.array([[1, 0], [1, 1]])
        assert iou_score(mask, mask) == pytest.approx(1.0)

    def test_disjoint_masks_give_iou_zero(self):
        a = np.array([[1, 0], [0, 0]])
        b = np.array([[0, 0], [0, 1]])
        assert iou_score(a, b) == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap_hand_computed(self):
        # A = {(0,0),(0,1)}, B = {(0,1),(1,1)} -> intersection=1, union=3
        # IoU = 1/3
        a = np.array([[1, 1], [0, 0]])
        b = np.array([[0, 1], [0, 1]])
        assert iou_score(a, b) == pytest.approx(1 / 3, abs=1e-4)

    def test_iou_is_stricter_than_dice_on_same_input(self):
        a = np.array([[1, 1], [0, 0]])
        b = np.array([[0, 1], [0, 1]])
        assert iou_score(a, b) < dice_coefficient(a, b)
