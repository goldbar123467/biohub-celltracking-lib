"""Read-only, nonsecret inventory on the existing Linux worker allocation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def run_read(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def process_still_matches(expected: dict | None) -> bool:
    if not expected:
        return False
    try:
        stat = Path(f"/proc/{int(expected['pid'])}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        actual = {"pid": int(expected["pid"]), "process_group": int(fields[2]),
                  "session": int(fields[3]), "start_ticks": int(fields[19]),
                  "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
        return fields[0] != "Z" and actual == expected
    except (FileNotFoundError, KeyError, ValueError, IndexError):
        return False


def inventory(root: Path, *, check_cuda: bool, worker_roots: list[Path] | None = None) -> dict:
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "project_root": str(root),
        "host": os.uname().nodename,
        "read_only": True,
        "provider": None,
        "provider_error": None,
        "git_commit": run_read(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_status": run_read(["git", "-C", str(root), "status", "--short"]),
        "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(root))),
        "gpu": run_read(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv,noheader,nounits"]),
        "gpu_processes": run_read(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]),
        "cpu_max": None,
        "memory_max": None,
        "cuda_operation": None,
        "launch_receipts": {},
        "worker_completions": {},
        "worker_heartbeats": {},
    }
    for field, path in (("cpu_max", "/sys/fs/cgroup/cpu.max"), ("memory_max", "/sys/fs/cgroup/memory.max")):
        if Path(path).is_file():
            result[field] = Path(path).read_text().strip()
    container_id = os.environ.get("CONTAINER_ID")
    credential = os.environ.get("CONTAINER_API_KEY")
    if container_id and credential and shutil.which("vastai"):
        raw = run_read(["vastai", "show", "instance", container_id, "--api-key", credential, "--raw"])
        if raw is not None:
            try:
                provider = json.loads(raw)
                if isinstance(provider, dict) and isinstance(provider.get("instances"), dict):
                    provider = provider["instances"]
                if isinstance(provider, dict) and isinstance(provider.get("instances"), list):
                    provider = provider["instances"]
                if isinstance(provider, list):
                    provider = next((p for p in provider if str(p.get("id")) == str(container_id)), {})
                allowed = {"id", "actual_status", "cur_state", "next_state", "gpu_name", "num_gpus", "dph_total", "dph_base", "dph_storage", "storage_cost", "total_cost", "end_date", "start_date", "duration", "disk_space", "cost_per_hour"}
                result["provider"] = {key: value for key, value in provider.items() if key in allowed}
            except (ValueError, AttributeError, TypeError):
                result["provider_error"] = "Provider response was not the expected JSON object; raw output withheld"
        else:
            result["provider_error"] = "Authenticated read-only instance query failed; raw output withheld"
    else:
        result["provider_error"] = "Required provider identity, credential or CLI unavailable"
    # Exact persisted tickets, not process-name guesses. Completion remains an
    # artifact claim until the controller checks the downloaded files.
    for worker_project in {root, *(worker_roots or [])}:
        if not worker_project.resolve().is_relative_to(root.resolve()):
            raise ValueError("Worker inventory must stay within this project")
        worker_root = worker_project / "reports/campaign-workers"
        if not worker_root.is_dir():
            continue
        for directory in sorted(worker_root.iterdir()):
            if not directory.is_dir():
                continue
            dispatch_path = directory / "dispatch.json"
            if dispatch_path.is_file() and dispatch_path.stat().st_size < 100_000:
                receipt = json.loads(dispatch_path.read_text())
                receipt["process_alive"] = process_still_matches(receipt.get("process_identity"))
                receipt["live_checked_at"] = result["observed_at"]
                result["launch_receipts"][receipt["intent_id"]] = receipt
            completion_path = directory / "worker/completion.json"
            if completion_path.is_file() and completion_path.stat().st_size < 1_000_000:
                completion = json.loads(completion_path.read_text())
                result["worker_completions"][completion["run_id"]] = completion
            heartbeat_path = directory / "worker/heartbeat.json"
            if heartbeat_path.is_file() and heartbeat_path.stat().st_size < 100_000:
                heartbeat = json.loads(heartbeat_path.read_text())
                result["worker_heartbeats"][heartbeat["run_id"]] = heartbeat
    if check_cuda:
        import torch

        start = time.monotonic()
        value = torch.arange(8, device="cuda", dtype=torch.float32).square().sum()
        torch.cuda.synchronize()
        if value.item() != 140:
            raise RuntimeError("CUDA arithmetic check failed")
        properties = torch.cuda.get_device_properties(0)
        result["cuda_operation"] = {
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": properties.name, "vram_bytes": properties.total_memory,
            "elapsed_seconds": time.monotonic() - start, "result": value.item(),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-cuda", action="store_true")
    parser.add_argument("--worker-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = inventory(args.project_root.resolve(), check_cuda=args.check_cuda, worker_roots=args.worker_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
