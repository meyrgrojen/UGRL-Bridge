"""Small neural network blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = ConvBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.size(-1)))
        return self.proj(emb)
