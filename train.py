"""Train UGRL-Bridge with reconstruction, calibration, bridge, and joint stages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ugrl_mri.data import MRIManifestDataset, RandomMaskFunc
from ugrl_mri.engine.trainer import evaluate, train_one_epoch
from ugrl_mri.losses import UGRLLoss
from ugrl_mri.models import UGRLBridge
from ugrl_mri.utils.config import load_config


def build_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    data = cfg["data"]
    mask_cfg = data["mask"]
    acceleration = mask_cfg["acceleration"]
    mask_func = RandomMaskFunc(
        acceleration=acceleration,
        center_fraction=mask_cfg["center_fractions"][acceleration],
        seed=cfg["project"]["seed"],
        phase_encode_axis=mask_cfg.get("phase_encode_axis", "width"),
    )
    dataset = MRIManifestDataset(
        root=data["root"],
        manifest=data["manifest"],
        dataset_name=data["dataset_name"],
        split=split,
        mask_func=mask_func,
        image_size=data["image_size"],
    )
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )


def set_trainable(model: UGRLBridge, phase: str) -> None:
    enabled = {
        "warmup": {"encoder", "tokens", "decoder"},
        "calibration": {"encoder", "tokens", "decoder", "token_interaction", "reliability"},
        "bridge": {"bridge"},
        "joint": {"encoder", "tokens", "decoder", "token_interaction", "reliability", "bridge"},
    }[phase]
    for name, module in model.named_children():
        for parameter in module.parameters():
            parameter.requires_grad_(name in enabled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ugrl_mri_singlecoil.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--acceleration", type=int, choices=(4, 8, 12), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["dataset_name"] = args.dataset
    cfg["data"]["image_size"] = 320 if args.dataset in {
        "fastmri_knee", "fastmri_brain_axflair"
    } else 256
    cfg["data"]["root"] = str(args.data_root)
    cfg["data"]["manifest"] = str(args.manifest)
    cfg["data"]["mask"]["acceleration"] = args.acceleration
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["project"]["output_dir"]) / args.dataset / f"R{args.acceleration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    model = UGRLBridge.from_config(cfg).to(device)
    loss_fn = UGRLLoss(cfg["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["optimizer"]["lr"],
        weight_decay=cfg["train"]["optimizer"]["weight_decay"],
    )
    stages = cfg["train"]["epochs"]
    total_epochs = sum(stages.values())
    warmup_epochs = cfg["train"]["scheduler"]["warmup_epochs"]

    def lr_factor(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.cuda.amp.GradScaler() if cfg["train"]["precision"] == "amp" and device.type == "cuda" else None
    best_nmse = float("inf")
    epoch = 0
    with (out_dir / "training_log.jsonl").open("w", encoding="utf-8") as log_file:
        for phase, phase_epochs in stages.items():
            set_trainable(model, phase)
            for _ in range(phase_epochs):
                epoch += 1
                train_log = train_one_epoch(
                    model, train_loader, optimizer, loss_fn, device, scaler,
                    phase=phase, grad_clip=cfg["train"]["grad_clip"],
                    memory_momentum=cfg["model"]["probabilistic_tokens"]["memory_momentum"],
                )
                val_log = evaluate(model, val_loader, loss_fn, device, phase=phase)
                scheduler.step()
                record = {"epoch": epoch, "phase": phase, "train": train_log, "val": val_log}
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record))
                if phase == "joint" and val_log["metric/nmse"] < best_nmse:
                    best_nmse = val_log["metric/nmse"]
                    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, out_dir / "best.pt")


if __name__ == "__main__":
    main()
