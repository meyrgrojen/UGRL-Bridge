"""Dual-domain complex encoder."""

from __future__ import annotations

import torch
from torch import nn

from ugrl_mri.models.blocks import ConvBlock, ResidualBlock
from ugrl_mri.utils.complex import complex_abs


class DualDomainEncoder(nn.Module):
    """Fuse image-domain complex features and frequency-domain hints."""

    def __init__(self, base_channels: int = 48, depth: int = 4):
        super().__init__()
        self.image_stem = ConvBlock(4, base_channels)
        self.freq_stem = ConvBlock(2, base_channels)
        self.gate = nn.Sequential(nn.Conv2d(base_channels, base_channels, 1), nn.Sigmoid())
        self.blocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(depth)])

    def forward(self, x_zf: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        k_mag = torch.log1p(complex_abs(kspace)).unsqueeze(1)
        image_in = torch.cat([x_zf, mask, k_mag], dim=1)
        freq_in = torch.cat([k_mag, mask], dim=1)
        f_img = self.image_stem(image_in)
        f_freq = self.freq_stem(freq_in)
        feat = f_img + self.gate(f_freq) * f_freq
        return self.blocks(feat)
