#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.pipelines.submission_pipeline import run_submission_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--config", help="JSON ClassicalConfig values")
    parser.add_argument("--cache-dir")
    parser.add_argument("--deadline-seconds", type=float)
    args = parser.parse_args(argv)
    import json

    from biohub_ct.pipelines.baseline_classical import ClassicalConfig

    config = ClassicalConfig(**json.loads(Path(args.config).read_text())) if args.config else None
    out = run_submission_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        debug=args.debug,
        config=config,
        cache_dir=args.cache_dir,
        deadline_seconds=args.deadline_seconds,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
