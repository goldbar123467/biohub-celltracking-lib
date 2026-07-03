from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from biohub_ct.config import KAGGLE_COMPETITION_DIRS


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    zarr_path: Path
    geff_path: Path | None = None


def default_data_dir(split: str = "test") -> Path:
    env = os.environ.get("BIOHUB_CT_DATA_DIR") or os.environ.get("CELLMOT_DATA_DIR")
    if env:
        return Path(env)
    for root in KAGGLE_COMPETITION_DIRS:
        candidate = root / split
        if candidate.exists():
            return candidate
    return Path("data") / split


def discover_datasets(
    data_dir: Path | str | None = None,
    *,
    split: str | None = None,
    require_geff: bool = False,
) -> list[DatasetRecord]:
    root = Path(data_dir) if data_dir is not None else default_data_dir(split or "test")
    if split and (root / split).exists():
        root = root / split
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    records: list[DatasetRecord] = []
    for zarr_path in sorted(root.glob("*.zarr")):
        geff_path = root / f"{zarr_path.stem}.geff"
        if require_geff and not geff_path.exists():
            continue
        records.append(
            DatasetRecord(
                name=zarr_path.stem,
                zarr_path=zarr_path,
                geff_path=geff_path if geff_path.exists() else None,
            )
        )
    return records

