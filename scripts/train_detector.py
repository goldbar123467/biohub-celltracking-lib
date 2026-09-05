"""Train one explicit split. Campaign orchestration belongs to its caller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from biohub_ct.data.paths import discover_datasets
from biohub_ct.training.data import DataConfig, SparsePatchSampler
from biohub_ct.training.model import ModelConfig
from biohub_ct.training.trainer import TrainConfig, train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    data = raw.get("data", {})
    for key in ("patch_shape", "sigma_voxels"):
        if key in data:
            data[key] = tuple(data[key])
    sampler = SparsePatchSampler(
        discover_datasets(raw["data_dir"], require_geff=True),
        raw["train_ids"],
        raw.get("val_ids", []),
        DataConfig(**data),
        dev_ids=raw.get("dev_ids", []),
    )
    report = train_detector(
        TrainConfig(**raw.get("training", {})),
        sampler,
        raw["output_dir"],
        ModelConfig(**raw.get("model", {})),
        raw.get("identity", {}),
        resume=args.resume,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
