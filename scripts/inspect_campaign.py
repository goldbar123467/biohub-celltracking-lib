#!/usr/bin/env python
"""Read a campaign and identify immutable checkpoints for durable backup."""

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        raise ValueError("Invalid run ID")
    root = Path("reports/campaigns") / args.run_id
    reports = {}
    checkpoints = {}
    for relative in (
        "manifest.json",
        "status.json",
        "summary.json",
        "fold0/status.json",
        "fold1/status.json",
        "refit/status.json",
        "fold0/selection.json",
        "fold1/selection.json",
        "fold0/outer-progress.json",
        "fold1/outer-progress.json",
    ):
        path = root / relative
        if path.exists():
            reports[relative] = json.loads(path.read_text())
    for fold in ("fold0", "fold1", "refit"):
        path = root / fold / "checkpoints/manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        reports[f"{fold}/checkpoints/manifest.json"] = manifest
        for pointer in (manifest.get(key) for key in ("latest", "previous", "best")):
            if pointer is None:
                continue
            if not re.fullmatch(r"step-\d+-[0-9a-f]+\.pt", pointer["file"]):
                raise ValueError("Invalid checkpoint filename")
            relative = f"{fold}/checkpoints/{pointer['file']}"
            checkpoints[relative] = {
                **pointer,
                "bytes": (root / relative).stat().st_size,
                "mtime": (root / relative).stat().st_mtime,
            }
    job = Path("reports/jobs") / args.run_id
    exit_path = job / "exit-code.txt"
    pid_path = job / "pid.txt"
    pid = int(pid_path.read_text()) if pid_path.exists() else None
    process_path = Path(f"/proc/{pid}/cmdline")
    process_alive = process_path.exists() and args.run_id.encode() in process_path.read_bytes()
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "checked_unix": time.time(),
                "exit_code": exit_path.read_text().strip() if exit_path.exists() else None,
                "job_pid": pid,
                "process_alive": process_alive,
                "disk_free_bytes": shutil.disk_usage(".").free,
                "gpu": gpu.stdout.strip(),
                "reports": reports,
                "checkpoints": checkpoints,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
