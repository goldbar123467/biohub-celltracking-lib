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
    args = parser.parse_args(argv)
    out = run_submission_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        debug=args.debug,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

