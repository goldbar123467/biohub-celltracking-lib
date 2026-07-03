#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.data.paths import discover_datasets
from biohub_ct.data.zarr_io import open_zarr_volume


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    for record in discover_datasets(args.data_dir, require_geff=False):
        volume = open_zarr_volume(record.zarr_path)
        print(record.name, "shape=", volume.shape, "chunks=", volume.chunks, "scale=", volume.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

