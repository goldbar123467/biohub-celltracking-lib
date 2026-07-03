#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.data.splits import write_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    splits = write_splits(args.data_dir, args.output, k=args.folds)
    print(f"wrote {args.output} ({len(splits)} splits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

