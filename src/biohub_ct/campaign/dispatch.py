"""Controller-side at-most-once launch on the already authorized Vast instance."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath

from biohub_ct.campaign.admission import (
    AdmissionError,
    decimal_nonnegative,
    file_sha256,
    require_fresh,
)
from biohub_ct.campaign.contracts import canonical_json, run_spec_digest, validate_identifier
from biohub_ct.campaign.state import CampaignStore, ReviewLease
from biohub_ct.campaign.watchdog import atomic_json


class LaunchUnknown(RuntimeError):
    """Remote launch may have succeeded; retain reservation and reconcile."""


class ExistingVastTransport:
    def __init__(self, ssh_config: Path, remote_root: str, *, runner: Callable = subprocess.run):
        self.ssh_config = ssh_config.resolve()
        self.remote_root = remote_root
        self.runner = runner

    def retrieve_completed(self, spec: dict, completion: dict, destination: Path) -> Path:
        """Download only the approved attempt's files and independently hash them."""
        run_id = validate_identifier(spec["run_id"], "run_id")
        prefix = PurePosixPath("reports/campaign-workers") / run_id
        files = [str(prefix / "worker/completion.json"), str(prefix / "worker/output.log")]
        if completion.get("status") == "COMPLETE":
            files.extend([spec["result_manifest_path"], *spec["expected_artifacts"]])
        for name in files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or not path.is_relative_to(prefix):
                raise AdmissionError("Downloaded files must stay within the approved attempt")
        files = list(dict.fromkeys(files))
        limit = int(decimal_nonnegative(spec.get("max_output_bytes"), "maximum output bytes")) + 20 * 1024**2
        code = """import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
names=json.loads(sys.argv[2]); limit=int(sys.argv[3]); result={}; total=0
for name in names:
    path=(root/name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Invalid artifact path')
    size=path.stat().st_size; total+=size
    if total>limit: raise ValueError('Artifact download exceeds approved size bound')
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    result[name]={'bytes':size,'sha256':digest}
print(json.dumps(result))
"""
        command = shlex.join([f"{self.remote_root}/.venv/bin/python3.12", "-c", code,
                              self.remote_root, json.dumps(files), str(limit)])
        result = self.runner(["ssh", "-F", str(self.ssh_config), "-o", "BatchMode=yes", "vast-biohub", command],
                             capture_output=True, text=True, timeout=100, check=False, shell=False)
        if result.returncode:
            raise AdmissionError("Remote artifact inventory failed; output withheld")
        try:
            described = json.loads(result.stdout)
        except (ValueError, TypeError) as exc:
            raise AdmissionError("Remote artifact inventory is not valid JSON") from exc
        if not isinstance(described, dict) or set(described) != set(files):
            raise AdmissionError("Remote artifact inventory differs from the approved file set")
        total = 0
        for entry in described.values():
            if (not isinstance(entry, dict) or type(entry.get("bytes")) is not int
                    or entry["bytes"] < 0 or not isinstance(entry.get("sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None):
                raise AdmissionError("Remote artifact inventory has invalid size or hash metadata")
            total += entry["bytes"]
        if total > limit:
            raise AdmissionError("Artifact inventory exceeds approved size bound")
        destination.mkdir(parents=True, exist_ok=True)
        def download_one(name: str) -> int:
            local = destination / name
            if not local.resolve().is_relative_to(destination.resolve()):
                raise AdmissionError("Local artifact path escapes download root")
            if (local.is_file() and not local.is_symlink()
                    and local.stat().st_size == described[name]["bytes"]
                    and file_sha256(local) == described[name]["sha256"]):
                return local.stat().st_size
            local.parent.mkdir(parents=True, exist_ok=True)
            temporary = local.with_name(local.name + ".download")
            if os.path.lexists(temporary) or not temporary.resolve().is_relative_to(destination.resolve()):
                raise AdmissionError("Unsafe temporary artifact path outside download root")
            # Reserve a fresh inode. Never let scp truncate an existing hardlink.
            try:
                with temporary.open("xb"):
                    pass
            except FileExistsError as exc:
                raise AdmissionError("Temporary artifact path already exists") from exc
            result = self.runner(["scp", "-F", str(self.ssh_config), "-o", "BatchMode=yes",
                "vast-biohub:" + shlex.quote(self.remote_root + "/" + name), str(temporary)],
                capture_output=True, text=True, timeout=120, check=False, shell=False)
            if result.returncode or temporary.is_symlink() or not temporary.is_file():
                raise AdmissionError("Artifact download is incomplete; reservation retained")
            downloaded_bytes = temporary.stat().st_size
            if downloaded_bytes > limit:
                raise AdmissionError("Downloaded artifacts exceed approved size bound")
            if temporary.stat().st_size != described[name]["bytes"] or file_sha256(temporary) != described[name]["sha256"]:
                raise AdmissionError("Downloaded artifact hash or size differs from remote receipt")
            os.replace(temporary, local)
            return downloaded_bytes

        # Each file is independent and hash-bound. Bound SSH concurrency so
        # small sidecars do not each add a serial connection round trip.
        with ThreadPoolExecutor(max_workers=min(4, len(files))) as pool:
            downloaded_bytes = sum(pool.map(download_one, files))
        if downloaded_bytes > limit:
            raise AdmissionError("Downloaded artifacts exceed approved size bound")
        receipt_path = destination / prefix / "worker/completion.json"
        # Bind the downloaded completion itself to the live observation used to
        # choose this file set. A changed receipt requires a new read-only review.
        try:
            downloaded_completion = json.loads(receipt_path.read_text())
        except (ValueError, UnicodeError) as exc:
            raise AdmissionError("Downloaded completion is invalid JSON") from exc
        if canonical_json(downloaded_completion) != canonical_json(completion):
            raise AdmissionError("Completion changed between observation and download")
        return receipt_path

    def stage(self, ticket_path: Path, run_id: str) -> str:
        validate_identifier(run_id, "run_id")
        remote = f"{self.remote_root}/work/campaign-ticket-{run_id}.json"
        result = self.runner(["scp", "-F", str(self.ssh_config), "-o", "BatchMode=yes",
                              str(ticket_path), "vast-biohub:" + shlex.quote(remote)],
                             capture_output=True, text=True, timeout=60, check=False, shell=False)
        if result.returncode:
            raise AdmissionError("Ticket staging failed before launch admission; transport output withheld")
        return remote

    def command(self, operation: str, remote_ticket: str) -> dict:
        if operation not in {"preflight", "dispatch"}:
            raise ValueError("Unsupported worker operation")
        argv = [f"{self.remote_root}/.venv/bin/python3.12", f"{self.remote_root}/scripts/campaign_worker.py",
                "--" + operation, remote_ticket, "--root", self.remote_root]
        result = self.runner(["ssh", "-F", str(self.ssh_config), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                              "vast-biohub", shlex.join(argv)],
                             capture_output=True, text=True, timeout=60, check=False, shell=False)
        if result.returncode:
            raise LaunchUnknown(f"Remote {operation} did not return a successful receipt")
        try:
            value = json.loads(result.stdout)
        except ValueError as exc:
            raise LaunchUnknown("Remote worker response was not a valid receipt") from exc
        if not isinstance(value, dict):
            raise LaunchUnknown("Remote worker response must be an object")
        return value


def launch_existing_vast(store: CampaignStore, transport: ExistingVastTransport,
                        spec: dict, *, owner: str, ledger_id: str,
                        reservation_amount: str, receipts: Path,
                        inventory: Callable[[], dict], review_lease: ReviewLease | None = None) -> dict:
    """Reserve and dispatch one bounded run; never retry uncertain remote launch."""
    digest = run_spec_digest(spec)
    run_id = spec["run_id"]
    validate_identifier(run_id, "run_id")
    with (store.review_lock(owner) if review_lease is None else nullcontext(review_lease)) as lease:
        campaign = store.get_campaign()
        if campaign.get("stop_requested") is True:
            raise AdmissionError("Campaign stop flag is set")
        if any(r["state"] in {"APPROVED", "LAUNCH_INTENT", "LAUNCH_UNKNOWN", "RUNNING"}
               and r["spec"]["execution"]["host"] == spec["execution"]["host"] for r in store.list_runs()):
            raise AdmissionError("An admitted job on this host is active or unresolved")
        live = inventory()
        require_fresh(live.get("observed_at"), now=datetime.now(UTC), max_age_seconds=120)
        provider = live.get("provider", {})
        expected_instance = spec["execution"].get("provider_instance_id")
        if expected_instance is None or provider.get("id") is None or provider.get("actual_status") != "running" or str(provider["id"]) != str(expected_instance):
            raise AdmissionError("Existing provider allocation identity or status changed")
        if live.get("gpu_processes") is None or live["gpu_processes"].strip():
            raise AdmissionError("GPU process inventory is unavailable or the GPU is occupied")
        free = decimal_nonnegative(live.get("disk", {}).get("free"), "free disk bytes")
        required = Decimal(15 * 1024**3) + decimal_nonnegative(spec.get("max_output_bytes"), "maximum output bytes")
        if free < required:
            raise AdmissionError("Job would consume the protected 15 GiB disk reserve")
        try:
            ledger = store.budget_snapshot(ledger_id)
        except KeyError as exc:
            raise AdmissionError("Named resource ledger has not been initialized and reconciled") from exc
        requested = decimal_nonnegative(reservation_amount, "reservation")
        # Include the watchdog's graceful shutdown tail in worst-case reservation.
        seconds = decimal_nonnegative(spec["execution"]["max_wall_seconds"], "walltime") + Decimal(15)
        worst = seconds / Decimal(3600)
        if ledger.unit == "USD":
            worst *= decimal_nonnegative(provider.get("dph_total"), "current instance hourly rate")
        elif ledger.unit != "instance_hours":
            raise AdmissionError("Vast launch ledger must use USD or instance_hours")
        if requested < worst:
            raise AdmissionError("Reservation is below worst-case runtime including shutdown")
        store.register_run(spec)
        ticket = {"spec": spec, "run_spec_sha256": digest, "fencing_token": lease.fencing_token,
                  "intent_id": "launch-" + run_id}
        directory = receipts / run_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "ticket.json"
        atomic_json(path, ticket)
        remote = transport.stage(path, run_id)
        preflight = transport.command("preflight", remote)
        if (preflight.get("preflight") != "PASS" or preflight.get("run_spec_sha256") != digest
                or preflight.get("intent_id") != ticket["intent_id"] or preflight.get("fencing_token") != lease.fencing_token
                or preflight.get("files", {}).get("verified_files") != spec.get("verified_files")
                or not spec.get("source_file_sha256")
                or preflight.get("files", {}).get("verified_source_files") != spec["source_file_sha256"]):
            raise AdmissionError("Remote byte-identity preflight failed")
        # Refresh the live balance immediately before approval/dispatch. The
        # ledger reservation itself is atomic under the same controller lock.
        refreshed = inventory()
        require_fresh(refreshed.get("observed_at"), now=datetime.now(UTC), max_age_seconds=120)
        current_provider = refreshed.get("provider", {})
        if current_provider.get("id") != provider["id"] or current_provider.get("actual_status") != "running":
            raise AdmissionError("Provider identity/status changed during preflight")
        if refreshed.get("gpu_processes") is None or refreshed["gpu_processes"].strip():
            raise AdmissionError("GPU became occupied during preflight")
        if decimal_nonnegative(refreshed.get("disk", {}).get("free"), "free disk bytes") < required:
            raise AdmissionError("Disk reserve changed during preflight")
        if ledger.unit == "USD" and requested < seconds / Decimal(3600) * decimal_nonnegative(current_provider.get("dph_total"), "refreshed instance hourly rate"):
            raise AdmissionError("Provider cost increased beyond the worst-case reservation")
        intent = store.authorize_launch(run_id, decision_id="decision-" + run_id,
            reviewer=owner, reservations=[{"ledger_id": ledger_id, "reservation_id": "compute-" + run_id,
                                          "amount": str(requested)}],
            reason="Existing allocation, exact remote byte identity, bounded work and reconciled reservation",
            intent_id=ticket["intent_id"], request_id="request-" + run_id,
            description_tag="biohub-" + digest[:32], fencing_token=lease.fencing_token)
        store.validate_dispatch(intent["intent_id"], run_id, digest, lease.fencing_token)
        try:
            receipt = transport.command("dispatch", remote)
            if (receipt.get("status") != "DISPATCHED" or receipt.get("run_spec_sha256") != digest
                    or receipt.get("fencing_token") != lease.fencing_token or not receipt.get("process_identity")
                    or receipt.get("run_id") != run_id or receipt.get("intent_id") != intent["intent_id"]):
                raise LaunchUnknown("Remote dispatch identity is incomplete or uncertain")
            store.confirm_launch(intent["intent_id"], provider_job_id=receipt["provider_job_id"], receipt=receipt)
            atomic_json(directory / "dispatch.json", receipt)
            return receipt
        except Exception as exc:  # noqa: BLE001 - persist unknown external launch outcome regardless of failure type
            receipt = {"status": "LAUNCH_UNKNOWN", "error_type": type(exc).__name__, "retry_allowed": False}
            store.mark_intent_unknown(intent["intent_id"], receipt=receipt)
            atomic_json(directory / "dispatch.json", receipt)
            return receipt
