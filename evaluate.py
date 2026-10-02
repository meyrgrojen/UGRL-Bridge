"""Evaluate a checkpoint on a fixed manifest and mask bank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from train import build_loader
from ugrl_mri.engine.trainer import move_to_device
from ugrl_mri.metrics import compute_metrics
from ugrl_mri.models import UGRLBridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    cfg["data"]["root"] = str(args.data_root)
    cfg["data"]["manifest"] = str(args.manifest)
    dataset = build_loader(cfg, args.split, shuffle=False).dataset
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["train"].get("num_workers", 4))
    model = UGRLBridge.from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    records = []
    with torch.no_grad():
        for batch in loader:
            sample_id = batch["sample_id"][0]
            batch = move_to_device(batch, device)
            output = model(batch, phase="joint")
            measurements = compute_metrics(output, batch)
            records.append({
                "dataset": cfg["data"]["dataset_name"],
                "acceleration": cfg["data"]["mask"]["acceleration"],
                "split": args.split,
                "sample_id": sample_id,
                **measurements,
            })

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_slice_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metric_keys = [key for key in records[0] if key not in {"dataset", "acceleration", "split", "sample_id"}]
    summary = {key: float(np.mean([row[key] for row in records])) for key in metric_keys}
    summary["samples"] = len(records)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
