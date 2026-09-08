#!/usr/bin/env python
"""Audit strict campaign evidence or the historical 61-of-71 outer report."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.campaign.evidence import (
    EvidenceValidationError,
    audit_evidence_bundle,
    audit_historical_outer_progress,
    load_json,
    strict_json_dumps,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("bundle", help="audit a schema-v1 bundle of per-clip evidence")
    bundle.add_argument("bundle", type=Path)
    bundle.add_argument(
        "--artifact-root",
        type=Path,
        help="base directory for relative graph artifact paths",
    )
    bundle.add_argument("--output", type=Path, help="write the recomputed aggregate")

    historical = subparsers.add_parser(
        "historical-outer",
        help="audit legacy outer-progress JSON without upgrading its evidence class",
    )
    historical.add_argument("outer_progress", type=Path)
    historical.add_argument("--split", type=Path, required=True)
    historical.add_argument("--fold", default="fold0")
    historical.add_argument("--output", type=Path, help="write the historical audit")
    return parser


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "bundle":
            result = audit_evidence_bundle(load_json(args.bundle), artifact_root=args.artifact_root)
        else:
            result = audit_historical_outer_progress(
                args.outer_progress, args.split, fold=args.fold
            )
        rendered = strict_json_dumps(result, indent=2) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            output = args.output.resolve()
            input_paths = (
                {args.bundle.resolve()}
                if args.command == "bundle"
                else {args.outer_progress.resolve(), args.split.resolve()}
            )
            if output in input_paths:
                raise EvidenceValidationError("output must not overwrite an input evidence file")
            _atomic_write(output, rendered)
            print(f"validated evidence written to {output}")
    except (EvidenceValidationError, OSError) as exc:
        print(f"evidence audit failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
