"""Reconstruct one measured complex k-space array with a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.io import loadmat
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ugrl_mri.models import UGRLBridge


def read_array(path: Path, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return np.asarray(data[key])
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".mat":
        try:
            return np.asarray(loadmat(path)[key])
        except NotImplementedError:
            with h5py.File(path) as data:
                return np.asarray(data[key])
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path) as data:
            return np.asarray(data[key])
    raise ValueError(f"Unsupported input format: {path}")


def complex_array(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        return array.astype(np.complex64)
    if array.ndim == 3 and array.shape[-1] == 2:
        return (array[..., 0] + 1j * array[..., 1]).astype(np.complex64)
    raise ValueError("Measured k-space must be a complex 2-D array or [H, W, 2]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mask", type=Path, help="Separate mask file; defaults to --input")
    parser.add_argument("--kspace-key", default="kspace")
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = UGRLBridge.from_config(checkpoint["cfg"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    measured = complex_array(read_array(args.input, args.kspace_key))
    mask = np.asarray(read_array(args.mask or args.input, args.mask_key), dtype=np.float32).squeeze()
    if mask.ndim == 1:
        if mask.size == measured.shape[1]:
            mask = np.broadcast_to(mask[None, :], measured.shape)
        elif mask.size == measured.shape[0]:
            mask = np.broadcast_to(mask[:, None], measured.shape)
    if mask.shape != measured.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match k-space {measured.shape}")
    kspace = torch.from_numpy(np.stack((measured.real, measured.imag))[None].astype(np.float32)).to(device)
    mask_tensor = torch.from_numpy(mask[None, None].copy()).to(device)
    with torch.no_grad():
        output = model({"kspace": kspace, "mask": mask_tensor}, phase="joint")
    reconstruction = output["x_hat"][0].cpu().numpy()
    reconstruction = reconstruction[0] + 1j * reconstruction[1]
    risk = output["U"][0, 0].cpu().numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        reconstruction=reconstruction.astype(np.complex64),
        magnitude=np.abs(reconstruction).astype(np.float32),
        risk=risk.astype(np.float32),
    )
    print(f"Saved reconstruction to {args.out}")


if __name__ == "__main__":
    main()
