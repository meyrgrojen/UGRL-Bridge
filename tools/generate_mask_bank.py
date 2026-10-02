"""Export fixed Cartesian sampling indices for a manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ugrl_mri.data import RandomMaskFunc, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--acceleration", type=int, choices=(4, 8, 12), required=True)
    parser.add_argument("--matrix-size", type=int, choices=(256, 320), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    center_fraction = {4: 0.08, 8: 0.04, 12: 0.02}[args.acceleration]
    generator = RandomMaskFunc(args.acceleration, center_fraction, args.seed)
    rows = [row for row in read_manifest(args.manifest) if row["dataset"] == args.dataset]
    if not rows:
        raise ValueError(f"No rows for {args.dataset}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "split", "sample_id", "acceleration", "matrix_size", "sampled_lines", "effective_acceleration", "line_indices"))
        for row in rows:
            mask = generator((args.matrix_size, args.matrix_size), f"{row['dataset']}:{row['sample_id']}")
            indices = mask[0, 0].nonzero().flatten().tolist()
            writer.writerow((
                row["dataset"], row["split"], row["sample_id"], args.acceleration,
                args.matrix_size, len(indices), f"{generator.effective_acceleration(mask):.5f}",
                ";".join(map(str, indices)),
            ))
    print(f"Wrote {len(rows)} masks to {args.out}")


if __name__ == "__main__":
    main()
