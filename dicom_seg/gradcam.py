"""
Milestone 7: Seg-Grad-CAM -- Grad-CAM adapted for segmentation.

Grad-CAM was originally defined for classification: backpropagate a single
predicted class *score* to a convolutional layer's activations, weight each
channel by its averaged gradient (how much that channel's activation
influenced the score), sum, and ReLU (only positive influence matters).

Segmentation has no single class score. The adaptation used here (a
straightforward reading of what's often called Seg-Grad-CAM) replaces the
classification score with the sum of the predicted mask's LOGITS over a
region of interest -- by default the whole image, or a caller-supplied
binary region to explain "why did the model activate *here*". Logits are
used rather than the post-sigmoid probabilities because sigmoid saturates:
gradients through a probability near 0 or 1 vanish, which would make the
attribution unreliable exactly where the model is most confident.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class SegGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """`target_layer` is typically the last conv layer of the encoder
        or the bottleneck -- deep enough to encode high-level structure,
        shallow enough (spatially) to still localize.
        """
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, region: torch.Tensor | None = None):
        """Returns (cam, probs): `cam` is a [H, W] numpy heatmap in [0, 1]
        for input_tensor[0] (batch size 1 expected), upsampled to the
        input's spatial size. `probs` is the model's normal sigmoid output.
        """
        self.model.zero_grad(set_to_none=True)
        logits, probs = self.model.forward_with_logits(input_tensor)

        score = (logits * region).sum() if region is not None else logits.sum()
        score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError("target_layer hooks never fired -- is it part of the model's forward pass?")

        # Global-average-pool the gradients over space -> one importance
        # weight per channel, exactly as in the original Grad-CAM.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only positive influence on the score matters

        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0]
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), probs.detach()
