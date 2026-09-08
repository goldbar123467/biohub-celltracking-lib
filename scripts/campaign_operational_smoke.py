"""One real, bounded arithmetic worker for campaign launch/recovery verification."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from biohub_ct.campaign.watchdog import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-units", type=int, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.max_units != 1 or not 0 < args.max_wall_seconds <= 30:
        parser.error("This operational check permits exactly one unit and at most 30 seconds")
    started = time.monotonic()
    attempt = Path(os.environ["BIOHUB_ATTEMPT_DIR"]).resolve(strict=True)
    identity = {"run_id": os.environ["BIOHUB_RUN_ID"], "run_spec_sha256": os.environ["BIOHUB_RUN_SPEC_SHA256"],
                "intent_id": os.environ["BIOHUB_INTENT_ID"], "fencing_token": int(os.environ["BIOHUB_FENCING_TOKEN"])}
    progress = Path(os.environ["BIOHUB_PROGRESS_PATH"])
    atomic_json(progress, {**identity, "completed_units": 0, "observed_at": datetime.now(UTC).isoformat(), "error": None})
    if args.device == "cuda":
        import torch
        value = torch.arange(8, dtype=torch.float32, device="cuda").square().sum()
        torch.cuda.synchronize()
        value = value.item()
        runtime = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    else:
        value = sum(n * n for n in range(8))
        runtime = {"device": "cpu"}
    if value != 140 or time.monotonic() - started > args.max_wall_seconds:
        raise RuntimeError("Operational arithmetic or application deadline check failed")
    output = attempt / "arithmetic.json"
    atomic_json(output, {"value": value, "expected": 140, "runtime": runtime,
                         "elapsed_seconds": time.monotonic() - started})
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    atomic_json(attempt / "result.json", {**identity, "status": "COMPLETE", "completed_units": 1,
        "artifact_sha256": {output.relative_to(ROOT).as_posix(): digest}})
    atomic_json(progress, {**identity, "completed_units": 1, "observed_at": datetime.now(UTC).isoformat(), "error": None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
