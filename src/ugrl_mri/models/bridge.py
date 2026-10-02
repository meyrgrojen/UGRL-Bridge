"""Reliability-controlled diffusion bridge."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ugrl_mri.models.blocks import ConvBlock, ResidualBlock, SinusoidalTimeEmbedding
from ugrl_mri.utils.complex import hard_data_consistency


class BridgeNet(nn.Module):
    def __init__(self, base_channels: int = 48, time_dim: int = 128):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.in_proj = ConvBlock(5, base_channels)
        self.time_proj = nn.Linear(time_dim, base_channels)
        self.body = nn.Sequential(ResidualBlock(base_channels), ResidualBlock(base_channels), ResidualBlock(base_channels))
        self.out = nn.Conv2d(base_channels, 2, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(torch.cat([x_t, mask, reliability, mask * reliability], dim=1))
        t_emb = self.time_proj(self.time(t)).view(t.size(0), -1, 1, 1)
        h = h + t_emb
        return self.out(self.body(h))


class ReliabilityControlledBridge(nn.Module):
    def __init__(self, base_channels: int = 48, steps_infer: int = 4, uncertainty_noise_scale: float = 0.5):
        super().__init__()
        self.net = BridgeNet(base_channels=base_channels)
        self.steps_infer = steps_infer
        self.uncertainty_noise_scale = uncertainty_noise_scale

    def make_training_state(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x0.size(0)
        t = torch.rand(b, device=x0.device)
        t_img = t.view(b, 1, 1, 1)
        # U is local risk: larger values slow deterministic transport and add noise.
        lam = t_img * (1.0 - 0.35 * reliability)
        sigma = self.uncertainty_noise_scale * reliability * torch.sin(torch.pi * t_img)
        noise = torch.randn_like(x0)
        x_t = (1.0 - lam) * x0 + lam * x1 + sigma * noise
        target_velocity = (1.0 - 0.35 * reliability) * (x1 - x0)
        target_velocity = target_velocity + (
            self.uncertainty_noise_scale * reliability * torch.pi * torch.cos(torch.pi * t_img) * noise
        )
        return x_t, t, target_velocity

    def training_forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_t, t, target_velocity = self.make_training_state(x0, x1, reliability)
        pred_velocity = self.net(x_t, t, mask, reliability)
        return {"x_t": x_t, "t": t, "pred_velocity": pred_velocity, "target_velocity": target_velocity}

    def sample(self, x0: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor, reliability: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = steps or self.steps_infer
        x = x0
        for idx in range(steps):
            t = torch.full((x.size(0),), idx / steps, device=x.device)
            velocity = self.net(x, t, mask, reliability)
            x = x + velocity / steps
            x = hard_data_consistency(x, kspace, mask)
        return x

    def forward(
        self,
        x0: torch.Tensor,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        reliability: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if target is not None and self.training:
            bridge = self.training_forward(x0, target, mask, reliability)
            x_hat = self.sample(x0, kspace, mask, reliability)
            bridge["x_hat"] = x_hat
            return bridge
        x_hat = self.sample(x0, kspace, mask, reliability)
        return {"x_hat": x_hat}
