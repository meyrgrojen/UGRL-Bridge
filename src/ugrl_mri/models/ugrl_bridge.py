"""UGRL-Bridge single-coil reconstruction network."""

from __future__ import annotations

import torch
from torch import nn

from ugrl_mri.models.bridge import ReliabilityControlledBridge
from ugrl_mri.models.encoder import DualDomainEncoder
from ugrl_mri.models.probabilistic_tokens import ProbabilisticTokenHead, RepresentationDensity
from ugrl_mri.models.reliability import ReliabilityEstimator
from ugrl_mri.models.token_attention import ReliabilityBiasedTokenAttention
from ugrl_mri.utils.complex import hard_data_consistency, ifft2c


class UGRLBridge(nn.Module):
    def __init__(
        self,
        base_channels: int = 48,
        token_dim: int = 192,
        encoder_depth: int = 4,
        window_size: int = 8,
        min_sigma: float = 1.0e-4,
        density_memory_size: int = 4096,
        density_temperature: float = 10.0,
        bridge_steps_infer: int = 4,
        bridge_noise_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = DualDomainEncoder(base_channels=base_channels, depth=encoder_depth)
        self.tokens = ProbabilisticTokenHead(base_channels, token_dim, min_sigma=min_sigma)
        self.density = RepresentationDensity(token_dim, density_memory_size, density_temperature)
        self.token_interaction = ReliabilityBiasedTokenAttention(token_dim, window_size=window_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(token_dim, base_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, 2, 3, padding=1),
        )
        self.reliability = ReliabilityEstimator(hidden_channels=max(16, base_channels // 2))
        self.bridge = ReliabilityControlledBridge(
            base_channels=base_channels,
            steps_infer=bridge_steps_infer,
            uncertainty_noise_scale=bridge_noise_scale,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "UGRLBridge":
        model_cfg = cfg.get("model", {})
        prob_cfg = model_cfg.get("probabilistic_tokens", {})
        bridge_cfg = model_cfg.get("bridge", {})
        return cls(
            base_channels=model_cfg.get("base_channels", 48),
            token_dim=model_cfg.get("token_dim", 192),
            encoder_depth=model_cfg.get("encoder_depth", 4),
            window_size=model_cfg.get("window_size", 8),
            min_sigma=prob_cfg.get("min_sigma", 1.0e-4),
            density_memory_size=prob_cfg.get("density_memory_size", 4096),
            density_temperature=prob_cfg.get("density_temperature", 10.0),
            bridge_steps_infer=bridge_cfg.get("steps_infer", 4),
            bridge_noise_scale=bridge_cfg.get("uncertainty_noise_scale", 0.5),
        )

    @torch.no_grad()
    def update_memory(self, token_means: torch.Tensor, momentum: float = 0.99) -> None:
        self.density.update_memory(token_means.detach(), momentum)

    def forward(self, batch: dict[str, torch.Tensor], phase: str = "joint") -> dict[str, torch.Tensor]:
        if phase not in {"warmup", "calibration", "bridge", "joint"}:
            raise ValueError(f"Unknown training phase: {phase}")
        kspace, mask = batch["kspace"], batch["mask"]
        x_zf = ifft2c(kspace)
        feat = self.encoder(x_zf, kspace, mask)
        token = self.tokens(feat)
        u_den = self.density(token["mu"])

        first_candidate = x_zf + self.decoder(token["z"])
        x_init = hard_data_consistency(first_candidate, kspace, mask)
        if phase == "warmup":
            return {
                "x_hat": x_init,
                "x_init": x_init,
                "x_zf": x_zf,
                "U": torch.zeros_like(mask),
                "mu": token["mu"],
                "sigma": token["sigma"],
                "u_var": token["u_var"],
                "u_den": u_den,
            }

        first_risk = self.reliability(first_candidate, kspace, mask, token["u_var"], u_den)
        attended = self.token_interaction(token["z"], first_risk["U"])
        candidate = x_zf + self.decoder(attended)
        x_init = hard_data_consistency(candidate, kspace, mask)
        risk = self.reliability(candidate, kspace, mask, token["u_var"], u_den)
        result = {
            "x_hat": x_init,
            "x_init": x_init,
            "x_zf": x_zf,
            "mu": token["mu"],
            "sigma": token["sigma"],
            **risk,
        }
        if phase in {"bridge", "joint"}:
            bridge = self.bridge(x_zf, kspace, mask, risk["U"], target=batch.get("target"))
            result.update(bridge)
        return result
