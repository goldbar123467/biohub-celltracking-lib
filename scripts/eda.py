#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.data.eda import write_eda_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-intensity-frames", type=int, default=3)
    args = parser.parse_args(argv)
    rows = write_eda_report(
        args.data_dir,
        args.output,
        max_intensity_frames=args.max_intensity_frames,
    )
    print(f"wrote {args.output} ({len(rows)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

