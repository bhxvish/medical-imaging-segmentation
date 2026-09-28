"""
Milestone 4: U-Net for binary segmentation, built from scratch (Ronneberger
et al., 2015).

Encoder: repeated (Conv-BN-ReLU) x2 blocks, each followed by MaxPool2d to
halve spatial resolution while doubling channel depth.
Bottleneck: the deepest, lowest-resolution block.
Decoder: ConvTranspose2d doubles resolution back up at each stage. Before
each decoder conv block, the corresponding encoder feature map (same
resolution) is concatenated on the channel axis -- the skip connection.
Skip connections are what let the decoder recover fine spatial detail that
plain downsampling destroys; without them the output is a blurry,
low-resolution guess upsampled back to full size, not a precise mask.
Output: 1x1 conv to a single channel + sigmoid, for binary segmentation.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU, twice."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = (64, 128, 256, 512),
    ):
        super().__init__()
        features = list(features)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for f in features:
            self.encoders.append(ConvBlock(prev_channels, f))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev_channels = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(prev_channels, f, kernel_size=2, stride=2)
            )
            # input channels doubled: upsampled features concatenated with
            # the skip connection from the corresponding encoder stage.
            self.decoders.append(ConvBlock(f * 2, f))
            prev_channels = f

        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, reversed(skip_connections)
        ):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                # Only triggered by odd input sizes that don't divide evenly
                # by 2 at every downsampling stage; crop the skip connection
                # to match rather than silently misaligning features.
                skip = _center_crop(skip, x.shape[-2:])
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.output_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self._logits(x))

    def forward_with_logits(self, x: torch.Tensor):
        """Same forward pass, but also returns the pre-sigmoid logits --
        needed by Seg-Grad-CAM, which backpropagates from raw logits rather
        than the saturating sigmoid output.
        """
        logits = self._logits(x)
        return logits, self.output_activation(logits)


def _center_crop(tensor: torch.Tensor, target_hw) -> torch.Tensor:
    _, _, h, w = tensor.shape
    target_h, target_w = target_hw
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return tensor[:, :, top:top + target_h, left:left + target_w]
