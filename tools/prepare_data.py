"""Prepare curated complex MRI slices without using unobserved test data.

Input CSV columns: dataset,split,sample_id,subject_id,sequence_id,volume_id,path.
Each path points to a complex 2-D image (or a 3-D stack with slice_index).
No multi-coil combination is performed here.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
import sys

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ugrl_mri.data import RandomMaskFunc, read_manifest


def read_complex(path: Path, key: str, slice_index: str, source_domain: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as contents:
            image = np.asarray(contents[key])
    elif path.suffix.lower() == ".mat":
        try:
            image = loadmat(path)[key]
        except NotImplementedError:
            with h5py.File(path) as handle:
                image = np.asarray(handle[key])
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path) as handle:
            image = np.asarray(handle[key])
    else:
        raise ValueError(f"Unsupported source format: {path}")

    if slice_index:
        image = image[int(slice_index)]
    if image.ndim == 3 and image.shape[-1] == 2 and not np.iscomplexobj(image):
        image = image[..., 0] + 1j * image[..., 1]
    if image.ndim != 2 or not np.iscomplexobj(image):
        raise ValueError(f"Expected a curated 2-D complex single-image slice: {path}, {image.shape}")
    image = image.astype(np.complex64)
    if source_domain == "kspace":
        image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(image), norm="ortho"))
    return image


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape
    if min(height, width) < size:
        raise ValueError(f"Image {image.shape} is smaller than {size} x {size}")
    top, left = (height - size) // 2, (width - size) // 2
    return image[top : top + size, left : left + size]


def zero_filled_magnitude(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image), norm="ortho"))
    zf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace * mask), norm="ortho"))
    return np.abs(zf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-size", type=int, choices=(256, 320), required=True)
    parser.add_argument("--source-key", default="GT")
    parser.add_argument("--source-domain", choices=("image", "kspace"), default="image")
    parser.add_argument("--acceleration", type=int, choices=(4, 8, 12), required=True)
    parser.add_argument("--center-fraction", type=float)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = [row for row in read_manifest(args.input_manifest) if row["dataset"] == args.dataset]
    if not rows:
        raise ValueError(f"No rows for dataset {args.dataset}")
    center_fraction = args.center_fraction or {4: 0.08, 8: 0.04, 12: 0.02}[args.acceleration]
    mask_func = RandomMaskFunc(args.acceleration, center_fraction, args.seed)
    by_volume: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_volume[(row["dataset"], row["split"], row["subject_id"], row["volume_id"])].append(row)

    output_rows = []
    for volume_rows in by_volume.values():
        images: list[np.ndarray] = []
        observed_magnitudes: list[np.ndarray] = []
        for row in volume_rows:
            image = read_complex(
                args.input_root / row["path"], args.source_key,
                row.get("slice_index", ""), args.source_domain,
            )
            image = center_crop(image, args.image_size)
            mask = mask_func(image.shape, f"{row['dataset']}:{row['sample_id']}")[0].numpy()
            images.append(image)
            observed_magnitudes.append(zero_filled_magnitude(image, mask).ravel())
        scale = float(np.percentile(np.concatenate(observed_magnitudes), args.percentile))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid observed-data normalization scale: {volume_rows[0]['volume_id']}")

        for row, image in zip(volume_rows, images):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["sample_id"]):
                raise ValueError(f"Unsafe sample ID: {row['sample_id']}")
            relative = Path(row["dataset"]) / row["split"] / f"{row['sample_id']}.npz"
            destination = args.output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, image=(image / scale).astype(np.complex64), scale=scale)
            output_rows.append({**row, "path": relative.as_posix()})

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Prepared {len(output_rows)} slices in {args.output_root}")


if __name__ == "__main__":
    main()
