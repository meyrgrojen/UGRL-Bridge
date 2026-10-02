"""Local token interaction biased by estimated reconstruction risk."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ReliabilityBiasedTokenAttention(nn.Module):
    def __init__(self, channels: int, window_size: int = 8, heads: int = 4, gamma: float = 1.0):
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.window_size = window_size
        self.heads = heads
        self.gamma = gamma
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)

    def _windows(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        size = self.window_size
        return (
            x.reshape(batch, channels, height // size, size, width // size, size)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(-1, size * size, channels)
        )

    def forward(self, tokens: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = tokens.shape
        size = self.window_size
        pad_h, pad_w = (-height) % size, (-width) % size
        padded = F.pad(tokens, (0, pad_w, 0, pad_h))
        risk = F.interpolate(risk, size=(height, width), mode="bilinear", align_corners=False)
        risk = F.pad(risk, (0, pad_w, 0, pad_h))
        padded_h, padded_w = padded.shape[-2:]

        query, key, value = self.qkv(padded).chunk(3, dim=1)
        head_dim = channels // self.heads

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return self._windows(x).view(-1, size * size, self.heads, head_dim).transpose(1, 2)

        q, k, v = split_heads(query), split_heads(key), split_heads(value)
        context_risk = self._windows(risk).squeeze(-1)
        logits = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        logits = logits - self.gamma * context_risk[:, None, None, :]
        attention = logits.softmax(dim=-1)
        result = (attention @ v).transpose(1, 2).reshape(-1, size * size, channels)
        result = (
            result.view(batch, padded_h // size, padded_w // size, size, size, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, channels, padded_h, padded_w)
        )
        return tokens + self.projection(result[:, :, :height, :width])
