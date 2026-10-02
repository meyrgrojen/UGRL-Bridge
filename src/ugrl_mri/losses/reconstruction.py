"""Reconstruction, bridge, and supervised reliability losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ugrl_mri.utils.complex import complex_abs, fft2c


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window: int = 7) -> torch.Tensor:
    c1, c2 = 0.01**2, 0.03**2
    pad = window // 2
    mu_x = F.avg_pool2d(x, window, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=pad) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return 1.0 - score.mean()


def high_frequency_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device, dtype=x.dtype)
    kernel = kernel.view(1, 1, 3, 3)
    return F.l1_loss(F.conv2d(x, kernel, padding=1), F.conv2d(y, kernel, padding=1))


class UGRLLoss(nn.Module):
    def __init__(self, weights: dict | None = None):
        super().__init__()
        self.weights = weights or {}

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        phase: str = "joint",
    ) -> dict[str, torch.Tensor]:
        target, pred = batch["target"], output["x_hat"]
        pred_mag = complex_abs(pred).unsqueeze(1)
        target_mag = complex_abs(target).unsqueeze(1)
        losses = {
            "rec_l1": F.l1_loss(pred, target),
            "rec_ssim": ssim_loss(pred_mag, target_mag),
            "hfen": high_frequency_loss(pred_mag, target_mag),
            "data_consistency": F.l1_loss(batch["mask"] * fft2c(pred), batch["kspace"]),
        }
        if phase in {"calibration", "joint"}:
            mu, sigma = output["mu"], output["sigma"]
            variance = sigma.square().clamp_min(1.0e-8)
            losses["probability_regularization"] = 0.5 * (
                mu.square() + variance - variance.log() - 1.0
            ).mean()
            losses["calibration"] = self._calibration_loss(output["U"], pred, target)
            losses["calibration_rank"] = self._ranking_loss(output["U"], pred, target)
        if phase in {"bridge", "joint"} and "pred_velocity" in output:
            losses["bridge"] = F.mse_loss(output["pred_velocity"], output["target_velocity"])

        total = pred.new_tensor(0.0)
        for name, value in losses.items():
            total = total + float(self.weights.get(name, 1.0)) * value
        losses["total"] = total
        return losses

    @staticmethod
    def _error_map(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixel_error = complex_abs(pred.detach() - target).unsqueeze(1)
        return F.avg_pool2d(pixel_error, 7, stride=1, padding=3)

    @classmethod
    def _calibration_loss(cls, risk: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = cls._error_map(pred, target)
        error = error / error.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        risk = risk / risk.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        return F.l1_loss(risk, error)

    @classmethod
    def _ranking_loss(cls, risk: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = F.adaptive_avg_pool2d(cls._error_map(pred, target), (16, 16)).flatten(1)
        risk = F.adaptive_avg_pool2d(risk, (16, 16)).flatten(1)
        error_difference = error[:, 1:] - error[:, :-1]
        risk_difference = risk[:, 1:] - risk[:, :-1]
        valid = error_difference.abs() > error_difference.std(dim=1, keepdim=True).clamp_min(1.0e-6)
        penalties = F.relu(0.05 - error_difference.sign() * risk_difference)
        return (penalties * valid).sum() / valid.sum().clamp_min(1)
