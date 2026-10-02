"""Per-slice reconstruction and reference-based risk metrics."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity
import torch

from ugrl_mri.utils.complex import fft2c


def _patch_means(image: np.ndarray, patch: int = 16) -> np.ndarray:
    height, width = image.shape
    image = image[: height // patch * patch, : width // patch * patch]
    return image.reshape(height // patch, patch, width // patch, patch).mean(axis=(1, 3)).ravel()


def _gradient(image: np.ndarray) -> np.ndarray:
    vertical, horizontal = np.gradient(image)
    return np.hypot(vertical, horizontal)


def _single_slice(pred: np.ndarray, target: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    pred_mag, target_mag = np.abs(pred), np.abs(target)
    error = np.abs(pred - target)
    mse = float(np.mean((pred_mag - target_mag) ** 2))
    peak = float(np.max(target_mag))
    psnr = 20 * np.log10(max(peak, 1.0e-8)) - 10 * np.log10(max(mse, 1.0e-12))
    data_range = max(float(np.max(target_mag) - np.min(target_mag)), 1.0e-8)
    ssim = structural_similarity(pred_mag, target_mag, data_range=data_range)
    nmse = float(np.sum(error**2) / max(np.sum(np.abs(target) ** 2), 1.0e-12))
    local_error = uniform_filter(error, size=7, mode="constant")
    rho = spearmanr(risk.ravel(), local_error.ravel()).statistic

    patch_risk, patch_error = _patch_means(risk), _patch_means(local_error)
    confident = patch_risk <= np.quantile(patch_risk, 0.20)
    severe = patch_error >= np.quantile(patch_error, 0.90)
    hcfr = 100.0 * float(np.mean(severe[confident]))

    target_gradient = _gradient(target_mag)
    pred_gradient = _gradient(pred_mag)
    fine_region = target_gradient >= np.quantile(target_gradient, 0.90)
    fine_error = float(
        np.sum(error[fine_region] ** 2)
        / max(np.sum(np.abs(target[fine_region]) ** 2), 1.0e-12)
    )
    false_detail = 100.0 * float(np.mean(
        (pred_gradient >= np.quantile(pred_gradient, 0.95))
        & (target_gradient <= np.quantile(target_gradient, 0.50))
    ))
    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "nmse": nmse,
        "u_e_rho": float(rho) if np.isfinite(rho) else 0.0,
        "hcfr_percent": hcfr,
        "fine_structure_error": fine_error,
        "fdr_percent": false_detail,
    }


def compute_metrics(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    pred_tensor, target_tensor = output["x_hat"].detach(), batch["target"].detach()
    risk_tensor = output["U"].detach()
    pred = torch.complex(pred_tensor[:, 0], pred_tensor[:, 1]).cpu().numpy()
    target = torch.complex(target_tensor[:, 0], target_tensor[:, 1]).cpu().numpy()
    risk = risk_tensor[:, 0].cpu().numpy()
    per_slice = [_single_slice(p, t, u) for p, t, u in zip(pred, target, risk)]
    result = {key: float(np.mean([item[key] for item in per_slice])) for key in per_slice[0]}
    observed = batch["mask"] * (fft2c(pred_tensor) - batch["kspace"])
    result["observed_kspace_residual"] = float(observed.abs().mean().cpu())
    return result
