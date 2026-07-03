#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.pipelines.evaluate import evaluate_fold


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--pipeline", default="classical", choices=["classical"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary = evaluate_fold(
        data_dir=args.data_dir,
        splits_path=args.split,
        fold=args.fold,
        pipeline=args.pipeline,
        output_path=args.output,
    )
    print(
        f"wrote {args.output} "
        f"score={summary.score:.6g} edge={summary.edge_tp}/{summary.edge_fp}/{summary.edge_fn}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
