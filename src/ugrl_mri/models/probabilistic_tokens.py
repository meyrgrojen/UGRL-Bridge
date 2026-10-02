"""Probabilistic local token representation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ProbabilisticTokenHead(nn.Module):
    def __init__(self, in_channels: int, token_dim: int, min_sigma: float = 1.0e-4):
        super().__init__()
        self.min_sigma = min_sigma
        self.mu = nn.Conv2d(in_channels, token_dim, 1)
        self.log_sigma = nn.Conv2d(in_channels, token_dim, 1)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        mu = self.mu(feat)
        sigma = F.softplus(self.log_sigma(feat)) + self.min_sigma
        if self.training:
            z = mu + sigma * torch.randn_like(sigma)
        else:
            z = mu
        u_var = torch.log(sigma.pow(2) + 1.0e-8).mean(dim=1, keepdim=True)
        return {"z": z, "mu": mu, "sigma": sigma, "u_var": u_var}


class RepresentationDensity(nn.Module):
    """Cosine-kernel density over local token means."""

    def __init__(self, token_dim: int, memory_size: int = 4096, temperature: float = 10.0):
        super().__init__()
        self.temperature = temperature
        self.register_buffer("memory", torch.zeros(memory_size, token_dim))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        # Density is evaluated on local 8 x 8 token regions to keep the
        # 4096-entry similarity bank tractable at 320 x 320 resolution.
        mu = F.avg_pool2d(mu, kernel_size=8, stride=8)
        b, d, h, w = mu.shape
        tokens = F.normalize(mu.flatten(2).transpose(1, 2), dim=-1)
        memory = F.normalize(self.memory, dim=-1)
        sim = torch.matmul(tokens, memory.t()) * self.temperature
        log_density = torch.logsumexp(sim, dim=-1) - torch.log(
            torch.tensor(memory.size(0), device=mu.device, dtype=sim.dtype)
        )
        u_den = -log_density.view(b, 1, h, w)
        return u_den

    @torch.no_grad()
    def update_memory(self, mu: torch.Tensor, momentum: float = 0.99) -> None:
        mu = F.avg_pool2d(mu, kernel_size=8, stride=8)
        tokens = F.normalize(mu.flatten(2).transpose(1, 2).reshape(-1, mu.size(1)), dim=-1)
        if tokens.numel() == 0:
            return
        count = self.memory.size(0)
        if not bool(self.initialized):
            token_indices = torch.randint(tokens.size(0), (count,), device=tokens.device)
            self.memory.copy_(tokens[token_indices])
            self.initialized.fill_(True)
            return
        num = min(tokens.size(0), count)
        token_indices = torch.randperm(tokens.size(0), device=tokens.device)[:num]
        prototype_indices = torch.randperm(count, device=tokens.device)[:num]
        updated = momentum * self.memory[prototype_indices] + (1.0 - momentum) * tokens[token_indices]
        self.memory[prototype_indices] = F.normalize(updated, dim=-1)
