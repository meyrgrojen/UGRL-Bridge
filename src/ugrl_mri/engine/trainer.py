"""Training and evaluation loops for the four-stage protocol."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ugrl_mri.losses import UGRLLoss
from ugrl_mri.metrics import compute_metrics


def move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: UGRLLoss,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    phase: str = "joint",
    grad_clip: float = 1.0,
    memory_momentum: float = 0.99,
) -> dict[str, float]:
    model.train()
    meter = defaultdict(float)
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(batch, phase=phase)
            losses = loss_fn(output, batch, phase=phase)
        if scaler is None:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        else:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        model.update_memory(output["mu"], memory_momentum)
        for k, v in losses.items():
            meter[k] += float(v.detach().cpu())
    return {k: v / max(len(loader), 1) for k, v in meter.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: UGRLLoss,
    device: torch.device,
    phase: str = "joint",
) -> dict[str, float]:
    model.eval()
    meter = defaultdict(float)
    sample_count = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_to_device(batch, device)
        batch_size = batch["target"].size(0)
        sample_count += batch_size
        output = model(batch, phase=phase)
        losses = loss_fn(output, batch, phase=phase)
        metrics = compute_metrics(output, batch)
        for k, v in losses.items():
            meter[f"loss/{k}"] += batch_size * float(v.detach().cpu())
        for k, v in metrics.items():
            meter[f"metric/{k}"] += batch_size * float(v)
    return {k: v / max(sample_count, 1) for k, v in meter.items()}
