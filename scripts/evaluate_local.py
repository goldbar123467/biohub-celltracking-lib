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
    parser.add_argument("--config")
    parser.add_argument("--metric-backend", choices=["official", "local-probe"], default="official")
    parser.add_argument("--metadata-smoke-only", action="store_true")
    args = parser.parse_args(argv)
    import json

    from biohub_ct.pipelines.baseline_classical import ClassicalConfig

    config = ClassicalConfig(**json.loads(Path(args.config).read_text())) if args.config else None
    summary = evaluate_fold(
        data_dir=args.data_dir,
        splits_path=args.split,
        fold=args.fold,
        pipeline=args.pipeline,
        output_path=args.output,
        metric_backend=args.metric_backend,
        config=config,
        metadata_smoke_only=args.metadata_smoke_only,
    )
    print(
        f"wrote {args.output} "
        f"score={summary.score:.6g} edge={summary.edge_tp}/{summary.edge_fp}/{summary.edge_fn}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
