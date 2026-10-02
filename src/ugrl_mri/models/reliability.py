"""Reliability estimation from representation and k-space evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ugrl_mri.models.blocks import ConvBlock
from ugrl_mri.utils.complex import complex_abs, fft2c, ifft2c


class ReliabilityEstimator(nn.Module):
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.missing_risk = nn.Sequential(
            ConvBlock(4, hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Softplus(),
        )
        self.fusion = nn.Sequential(
            ConvBlock(5, hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def physical_terms(self, pred: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_k = fft2c(pred)
        obs_res = mask * (pred_k - kspace)
        u_obs = complex_abs(ifft2c(obs_res)).unsqueeze(1)
        missing_image = ifft2c((1.0 - mask) * pred_k)
        missing_hint = complex_abs(missing_image).unsqueeze(1)
        return u_obs, missing_hint

    def forward(
        self,
        pred: torch.Tensor,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        u_var: torch.Tensor,
        u_den: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        u_obs, missing_hint = self.physical_terms(pred, kspace, mask)
        size = pred.shape[-2:]
        u_var = F.interpolate(u_var, size=size, mode="bilinear", align_corners=False)
        u_den = F.interpolate(u_den, size=size, mode="bilinear", align_corners=False)
        mask_img = mask
        u_miss = self.missing_risk(torch.cat([missing_hint, u_var, u_den, mask_img], dim=1))
        terms = torch.cat([u_var, u_den, u_obs, u_miss, mask_img], dim=1)
        reliability = self.fusion(terms)
        return {
            "U": reliability,
            "u_var": u_var,
            "u_den": u_den,
            "u_obs": u_obs,
            "u_miss": u_miss,
        }
