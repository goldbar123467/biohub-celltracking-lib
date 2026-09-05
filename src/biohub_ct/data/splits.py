from __future__ import annotations

import json
from pathlib import Path

from biohub_ct.data.paths import discover_datasets


def embryo_id(dataset_name: str) -> str:
    return dataset_name.split("_", 1)[0]


def make_dataset_splits(data_dir: Path | str, *, k: int = 5) -> dict[str, dict[str, list[str]]]:
    records = discover_datasets(data_dir, require_geff=True)
    names = sorted(record.name for record in records)
    if not names:
        raise FileNotFoundError(f"No paired .zarr/.geff train datasets found under {data_dir}")
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(embryo_id(name), []).append(name)
    group_names = sorted(groups)
    if k < 2 or len(group_names) < 2:
        raise ValueError("Embryo-held-out evaluation requires at least two groups and k >= 2")
    fold_count = min(k, len(group_names))
    splits: dict[str, dict[str, list[str]]] = {}

    first_val_group = group_names[0]
    splits["debug"] = {
        "train": [name for group in group_names[1:] for name in groups[group]],
        "val": groups[first_val_group][:1],
    }
    splits["fold0"] = {
        "train": sorted([name for group in group_names[1:] for name in groups[group]]),
        "val": sorted(groups[first_val_group]),
    }
    for fold_index in range(fold_count):
        val_groups = group_names[fold_index::fold_count]
        val = [name for group in val_groups for name in groups[group]]
        train = [name for group in group_names if group not in val_groups for name in groups[group]]
        splits[f"fold{fold_index}"] = {"train": sorted(train), "val": sorted(val)}
    return splits


def validate_split(split: dict[str, list[str]], available: set[str]) -> None:
    train, val = split["train"], split["val"]
    if not train or not val:
        raise ValueError("Train and validation sets must both be nonempty")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Duplicate split IDs")
    missing = (set(train) | set(val)) - available
    if missing:
        raise ValueError(f"Missing requested datasets: {sorted(missing)}")
    if {embryo_id(n) for n in train} & {embryo_id(n) for n in val}:
        raise ValueError("Train and validation embryos overlap")


def write_splits(
    data_dir: Path | str, output_path: Path | str, *, k: int = 5
) -> dict[str, dict[str, list[str]]]:
    splits = make_dataset_splits(data_dir, k=k)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return splits
