"""Explicit checkpoint migration and cumulative recovery budget accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

from biohub_ct.training.checkpoint import load_checkpoint

CAMPAIGN_BUDGET_SECONDS = 27000
REPAIR_RESERVE_SECONDS = 600


def recovery_plan(
    parent_id,
    *,
    root=Path("reports/campaigns"),
    jobs=Path("reports/jobs"),
    repair_seconds=REPAIR_RESERVE_SECONDS,
):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", parent_id):
        raise ValueError("Invalid recovery parent ID")
    if not math.isfinite(repair_seconds) or repair_seconds < REPAIR_RESERVE_SECONDS:
        raise ValueError("Repair budget charge must be at least 600 seconds")
    parent = root / parent_id
    manifest = json.loads((parent / "manifest.json").read_text())
    status = json.loads((parent / "status.json").read_text())
    job = jobs / parent_id
    if not (job / "exit-code.txt").exists():
        raise RuntimeError("Parent job has no terminal exit code; refuse duplicate training")
    if status["stage"] not in {"failed", "budget_exhausted"}:
        raise ValueError("Recovery requires a failed or interrupted campaign")
    started = datetime.fromisoformat((job / "started_at.txt").read_text().strip())
    finished = datetime.fromisoformat((job / "finished_at.txt").read_text().strip())
    elapsed = max(float(status.get("elapsed_seconds", 0)), (finished - started).total_seconds())
    spent = math.ceil(
        float(manifest.get("budget_spent_before_seconds", 0)) + elapsed + repair_seconds
    )
    remaining = CAMPAIGN_BUDGET_SECONDS - spent
    if remaining < 300:
        raise RuntimeError("Cumulative campaign budget exhausted; no restart authorized")
    return {
        "parent_id": parent_id,
        "parent_elapsed_seconds": elapsed,
        "repair_reserve_seconds": repair_seconds,
        "budget_spent_before_seconds": spent,
        "remaining_seconds": remaining,
    }


def copy_verified_checkpoints(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "manifest.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    for pointer in {
        p["file"]: p for k in ("latest", "previous", "best") if (p := manifest.get(k)) is not None
    }.values():
        if Path(pointer["file"]).name != pointer["file"]:
            raise ValueError("Invalid checkpoint pointer")
        path = source / pointer["file"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != pointer["sha256"]:
                raise ValueError("Checkpoint pointer SHA256 mismatch")
        load_checkpoint(path)
        shutil.copy2(path, destination / path.name)
        shutil.copy2(
            path.with_suffix(".sha256.json"), destination / path.with_suffix(".sha256.json").name
        )
    shutil.copy2(source / "manifest.json", destination / "manifest.json")
    return load_checkpoint(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_id")
    parser.add_argument("--repair-seconds", type=float, default=REPAIR_RESERVE_SECONDS)
    args = parser.parse_args()
    print(recovery_plan(args.parent_id, repair_seconds=args.repair_seconds)["remaining_seconds"])
