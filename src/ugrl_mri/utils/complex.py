"""Complex-valued MRI tensor utilities.

The code uses two-channel real tensors for complex data:
    [B, 2, H, W] = [real, imag].
"""

from __future__ import annotations

import torch


def chan_to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.size(1) != 2:
        raise ValueError(f"Expected [B, 2, H, W], got {tuple(x.shape)}")
    return torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())


def complex_to_chan(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x):
        raise ValueError("Expected a complex tensor.")
    return torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()


def fft2c(x: torch.Tensor) -> torch.Tensor:
    z = chan_to_complex(x)
    z = torch.fft.ifftshift(z, dim=(-2, -1))
    z = torch.fft.fft2(z, norm="ortho")
    z = torch.fft.fftshift(z, dim=(-2, -1))
    return complex_to_chan(z)


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    z = chan_to_complex(k)
    z = torch.fft.ifftshift(z, dim=(-2, -1))
    z = torch.fft.ifft2(z, norm="ortho")
    z = torch.fft.fftshift(z, dim=(-2, -1))
    return complex_to_chan(z)


def complex_abs(x: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return torch.sqrt(x[:, 0].pow(2) + x[:, 1].pow(2) + eps)


def complex_mul_mask(k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return k * mask


def hard_data_consistency(pred: torch.Tensor, kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace sampled k-space locations by measured data."""
    pred_k = fft2c(pred)
    dc_k = mask * kspace + (1.0 - mask) * pred_k
    return ifft2c(dc_k)


def center_crop_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _, _, h, w = x.shape
    _, _, rh, rw = ref.shape
    top = max((h - rh) // 2, 0)
    left = max((w - rw) // 2, 0)
    return x[:, :, top : top + rh, left : left + rw]
