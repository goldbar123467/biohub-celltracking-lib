"""Run one bounded review from the verified Windows campaign controller."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VAST_WORKER_HOST = "existing-vast-4070-super"
VAST_PROVIDER_INSTANCE_ID = 49994628


def _is_configured_vast_execution(execution: object) -> bool:
    return (
        isinstance(execution, Mapping)
        and execution.get("host") == VAST_WORKER_HOST
        and str(execution.get("provider_instance_id")) == str(VAST_PROVIDER_INSTANCE_ID)
    )


def _configured_vast_worker_roots(
    runs: Sequence[Mapping], proposed_spec: Mapping | None = None
) -> set[str]:
    """Select only roots belonging to this reviewer's configured Vast worker."""
    roots: set[str] = set()
    for run in runs:
        spec = run.get("spec")
        execution = spec.get("execution") if isinstance(spec, Mapping) else None
        if not _is_configured_vast_execution(execution):
            continue
        working_directory = execution.get("working_directory")
        if not isinstance(working_directory, str) or not working_directory:
            raise ValueError("Configured Vast run has no valid working directory")
        roots.add(working_directory)
    if proposed_spec is not None:
        execution = proposed_spec.get("execution")
        if not _is_configured_vast_execution(execution):
            raise ValueError("Proposed launch must target the configured existing Vast instance")
        working_directory = execution.get("working_directory")
        if not isinstance(working_directory, str) or not working_directory:
            raise ValueError("Proposed Vast launch has no valid working directory")
        roots.add(working_directory)
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--mutation-enabled", action="store_true")
    parser.add_argument("--release-id")
    parser.add_argument("--run-spec", type=Path)
    parser.add_argument("--ledger-id")
    parser.add_argument("--reservation-amount")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.store.is_file():
        parser.error("Initialize and review the canonical campaign store before this entrypoint")
    if not args.worker:
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                                    timeout=550, check=False, shell=False)
        except subprocess.TimeoutExpired:
            print(json.dumps({"status": "ITERATION_TIMEOUT", "reconcile_before_new_mutations": True}))
            return 124
        return result.returncode

    from biohub_ct.campaign.kaggle_cli import KaggleCLI
    from biohub_ct.campaign.reviewer import ReviewConfig, review_once
    from biohub_ct.campaign.state import CampaignStore

    if os.name != "nt":
        parser.error("This entrypoint is configured for the verified Windows controller")
    cli = KaggleCLI(["wsl", "-d", "Ubuntu-24.04", "--", "/home/thecl/.local/bin/kaggle"])
    store = CampaignStore(args.store)
    spec = None
    if args.run_spec:
        if not args.ledger_id or not args.reservation_amount:
            parser.error("A launch requires explicit ledger ID and worst-case reservation")
        spec = json.loads(args.run_spec.read_text(encoding="utf-8"))
        _configured_vast_worker_roots((), spec)

    def inventory():
        # Fixed verified remote command. No user input or secrets enter shell text.
        command_argv = ["/workspace/biohub-cell-tracking/.venv/bin/python3.12",
            "/workspace/biohub-cell-tracking/work/campaign_inventory-20260908.py",
            "--project-root", "/workspace/biohub-cell-tracking", "--output",
            "/workspace/biohub-cell-tracking/reports/campaign-review-live-inventory.json"]
        worker_roots = _configured_vast_worker_roots(store.list_runs(), spec)
        for root in sorted(worker_roots):
            command_argv.extend(["--worker-root", root])
        command = shlex.join(command_argv)
        result = subprocess.run(["ssh", "-F", str(ROOT / "configs/ssh_config"),
                                 "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "vast-biohub", command],
                                capture_output=True, text=True, timeout=100, check=False, shell=False)
        if result.returncode:
            raise RuntimeError("Read-only Vast inventory failed; raw transport output withheld")
        value = json.loads(result.stdout)
        if value.get("provider_error") or not value.get("provider", {}).get("id"):
            raise RuntimeError("Authenticated provider inventory is incomplete")
        return value

    launch_request = None
    from biohub_ct.campaign.dispatch import ExistingVastTransport
    if spec is not None:
        launch_request = {"spec": spec, "ledger_id": args.ledger_id, "reservation_amount": args.reservation_amount,
            "transport": ExistingVastTransport(ROOT / "configs/ssh_config", spec["execution"]["working_directory"])}

    def download_completion(spec, completion):
        transport = ExistingVastTransport(ROOT / "configs/ssh_config", spec["execution"]["working_directory"])
        # Keep room for attempt-local cache sidecars under Windows MAX_PATH.
        # The canonical state still binds the full independent download path.
        destination = ROOT / "reports/campaign-downloads" / spec["run_id"]
        return transport.retrieve_completed(spec, completion, destination), destination
    result = review_once(store, cli,
        ReviewConfig("windows-campaign-reviewer", args.artifact_root.resolve(), args.receipts.resolve(), args.mutation_enabled),
        inventory=inventory, release_id=args.release_id, launch_request=launch_request,
        download_completion=download_completion)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
