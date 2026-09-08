"""Linux receiver for one controller-admitted, immutable worker ticket."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from biohub_ct.campaign.admission import AdmissionError, file_sha256, verify_run_files
from biohub_ct.campaign.contracts import run_spec_digest, validate_identifier, validate_run_spec
from biohub_ct.campaign.watchdog import atomic_json, process_identity, run_bounded_job


def validate_ticket(ticket: dict, root: Path) -> dict:
    spec = ticket["spec"]
    validate_run_spec(spec, for_approval=True)
    if ticket.get("run_spec_sha256") != run_spec_digest(spec):
        raise AdmissionError("Worker ticket specification hash mismatch")
    validate_identifier(ticket.get("intent_id"), "intent_id")
    token = ticket.get("fencing_token")
    if isinstance(token, bool) or not isinstance(token, int) or token <= 0:
        raise AdmissionError("Worker ticket requires a positive dispatch fencing token")
    execution = spec["execution"]
    if Path(execution["working_directory"]).resolve() != root.resolve():
        raise AdmissionError("Worker ticket targets a different root")
    progress = Path(execution.get("worker_progress_path", ""))
    if not progress.is_absolute() or not progress.resolve().is_relative_to(root.resolve()):
        raise AdmissionError("Production workers require a progress path within the project")
    checked = verify_run_files(spec, root)
    if spec.get("source_file_sha256") != checked["verified_source_files"]:
        raise AdmissionError("Checked source files differ from the immutable source-file map")
    supervisor_name = "scripts/campaign_worker.py"
    module_names = ("worker_ticket.py", "watchdog.py", "admission.py", "contracts.py")
    required_sources = [supervisor_name, *("src/biohub_ct/campaign/" + name for name in module_names)]
    if any(name not in checked["verified_source_files"] for name in required_sources):
        raise AdmissionError("Worker source manifest must bind the supervisor and its verification modules")
    for name in module_names:
        imported = Path(__file__).with_name(name)
        if file_sha256(imported) != checked["verified_source_files"]["src/biohub_ct/campaign/" + name]:
            raise AdmissionError("Loaded worker verification code differs from the reviewed source")
    attempt_dir = root.resolve() / "reports/campaign-workers" / spec["run_id"]
    manifest_name = Path(spec.get("result_manifest_path", ""))
    if manifest_name.is_absolute() or ".." in manifest_name.parts:
        raise AdmissionError("Result manifest path must stay inside the exclusive attempt directory")
    manifest_path = (root / manifest_name).resolve()
    if not manifest_path.is_relative_to(attempt_dir):
        raise AdmissionError("Result manifest must belong to the exclusive attempt directory")
    if not progress.resolve().is_relative_to(attempt_dir):
        raise AdmissionError("Progress must belong to the exclusive attempt directory")
    entry = execution.get("entrypoint_file")
    if entry not in checked["verified_source_files"]:
        raise AdmissionError("Worker entrypoint is not included in the checked source manifest")
    argv = execution["argv"]
    if len(argv) < 2 or Path(argv[1]).resolve() != (root / entry).resolve():
        raise AdmissionError("Worker argv does not execute the verified entrypoint")
    if file_sha256(Path(argv[0])) != execution.get("executable_sha256"):
        raise AdmissionError("Worker executable differs from the reviewed runtime")
    deadline = datetime.fromisoformat(execution["deadline_utc"])
    if deadline <= datetime.now(UTC):
        raise AdmissionError("Worker ticket deadline passed")
    return checked


def dispatch_ticket(ticket_path: Path, root: Path, *, supervisor_script: Path) -> dict:
    """Persist remote claim before Popen. Duplicate delivery never starts another job."""
    if os.name != "posix":
        raise AdmissionError("Tickets execute only on the verified Linux worker host")
    import fcntl

    ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
    validate_ticket(ticket, root)
    if supervisor_script.resolve() != (root / "scripts/campaign_worker.py").resolve():
        raise AdmissionError("Dispatch must use the source-bound supervisor entrypoint")
    run_id = ticket["spec"]["run_id"]
    parent = root / "reports/campaign-workers"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / "dispatch.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        directory = parent / run_id
        if directory.exists():
            prior = json.loads((directory / "ticket.json").read_text())
            if prior != ticket:
                raise AdmissionError("Existing run ID belongs to a different immutable ticket")
            receipt_path = directory / "dispatch.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                expected_identity = {"run_id": run_id, "intent_id": ticket["intent_id"],
                    "run_spec_sha256": ticket["run_spec_sha256"], "fencing_token": ticket["fencing_token"]}
                if any(receipt.get(key) != value for key, value in expected_identity.items()):
                    raise AdmissionError("Persisted dispatch receipt differs from the exact ticket")
                return receipt
            return {"status": "LAUNCH_UNKNOWN", "intent_id": ticket["intent_id"],
                    "run_spec_sha256": ticket["run_spec_sha256"], "retry_allowed": False}
        watermark = parent / "fencing.json"
        previous = json.loads(watermark.read_text())["fencing_token"] if watermark.exists() else 0
        if ticket["fencing_token"] < previous:
            raise AdmissionError("Stale controller fencing token")
        atomic_json(watermark, {"fencing_token": ticket["fencing_token"]})
        directory.mkdir(exist_ok=False)
        atomic_json(directory / "ticket.json", ticket)
        # Durable one-shot marker survives an SSH disconnect or supervisor crash.
        with (directory / "supervisor.log").open("ab") as stream:
            process = subprocess.Popen([sys.executable, str(supervisor_script), "--execute", str(directory / "ticket.json"),
                                        "--root", str(root)],
                cwd=root, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, shell=False)
        identity = process_identity(process.pid)
        receipt = {"status": "DISPATCHED" if identity is not None else "LAUNCH_UNKNOWN", "intent_id": ticket["intent_id"],
            "run_id": run_id, "run_spec_sha256": ticket["run_spec_sha256"],
            "fencing_token": ticket["fencing_token"], "provider_job_id": f"linux-supervisor-{process.pid}",
            "process_identity": identity, "observed_at": datetime.now(UTC).isoformat()}
        atomic_json(directory / "dispatch.json", receipt)
        return receipt


def execute_ticket(ticket_path: Path, root: Path) -> dict:
    ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
    validate_ticket(ticket, root)
    spec = ticket["spec"]
    attempt_dir = root.resolve() / "reports/campaign-workers" / spec["run_id"]
    if ticket_path.resolve() != attempt_dir / "ticket.json":
        raise AdmissionError("Execution ticket must be the exclusively claimed attempt record")
    if (root / spec["result_manifest_path"]).exists():
        raise AdmissionError("An existing result manifest cannot be reused for a new execution")
    environment = spec["execution"].get("nonsecret_environment", {})
    required_env = {"BIOHUB_RUN_ID": spec["run_id"], "BIOHUB_RUN_SPEC_SHA256": ticket["run_spec_sha256"],
                    "BIOHUB_PROGRESS_PATH": spec["execution"]["worker_progress_path"],
                    "BIOHUB_INTENT_ID": ticket["intent_id"], "BIOHUB_FENCING_TOKEN": str(ticket["fencing_token"]),
                    "BIOHUB_ATTEMPT_DIR": str(attempt_dir)}
    # Digest cannot self-reference its own value. These derived values are
    # injected by the checked supervisor after the exact specification check.
    launched_spec = {**spec, "execution": {**spec["execution"],
                                         "nonsecret_environment": {**environment, **required_env}}}

    def artifacts():
        # An application manifest is required, then every declared file is hashed.
        result_path = root / spec["result_manifest_path"]
        result = json.loads(result_path.read_text())
        if result.get("run_id") != spec["run_id"] or result.get("run_spec_sha256") != ticket["run_spec_sha256"]:
            raise AdmissionError("Worker result manifest has a different identity")
        if result.get("intent_id") != ticket["intent_id"] or result.get("fencing_token") != ticket["fencing_token"]:
            raise AdmissionError("Worker result manifest belongs to a different dispatch attempt")
        if result.get("status") != "COMPLETE":
            raise AdmissionError("Worker result manifest is incomplete")
        completed = result.get("completed_units")
        expected = spec["expected_completed_units"]
        if isinstance(completed, bool) or completed != expected or completed > spec["execution"]["max_steps_or_clips"]:
            raise AdmissionError("Worker result coverage is incomplete or beyond its cap")
        from biohub_ct.campaign.admission import verify_file_manifest
        files = result.get("artifact_sha256", {})
        if set(files) != set(spec["expected_artifacts"]):
            raise AdmissionError("Worker result does not cover the exact expected artifacts")
        if any(not (root / name).resolve().is_relative_to(attempt_dir) for name in files):
            raise AdmissionError("Worker outputs must belong to the exclusive attempt directory")
        verify_file_manifest(root, files)
        return {"complete": True, "result_manifest_sha256": file_sha256(result_path),
                "artifact_sha256": files, "completed_units": completed,
                "quality_review_required": True}

    return run_bounded_job(launched_spec, ticket_path.parent / "worker",
        run_spec_sha256=ticket["run_spec_sha256"], fencing_token=ticket["fencing_token"],
        validate_dispatch=lambda: validate_ticket(ticket, root), validate_artifacts=artifacts)
