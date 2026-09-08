from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from biohub_ct.campaign.admission import AdmissionError, file_sha256
from biohub_ct.campaign.contracts import run_spec_digest
from biohub_ct.campaign.dispatch import ExistingVastTransport, launch_existing_vast
from biohub_ct.campaign.state import CampaignStore

HASHES = [f"{index:x}" * 64 for index in range(1, 10)]


def run_spec(run_id: str = "run-1") -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": "bounded-transport-test",
        "work_kind": "gpu-verification",
        "source": {
            "git_commit": "1" * 40,
            "source_bundle_sha256": HASHES[0],
            "dependency_manifest_sha256": HASHES[1],
            "effective_config_sha256": HASHES[2],
        },
        "data": {
            "input_manifest_sha256": HASHES[3],
            "split_manifest_sha256": HASHES[4],
        },
        "execution": {
            "host": "existing-vast-4070-super",
            "provider_instance_id": 456,
            "working_directory": "/workspace/biohub-cell-tracking",
            "argv": ["python3.12", "worker.py", "--run-id", run_id],
            "max_wall_seconds": 60,
            "max_steps_or_clips": 1,
            "deadline_utc": "2030-09-08T00:00:00Z",
            "worker_progress_path": f"reports/campaign-workers/{run_id}/progress.json",
        },
        "stop_rules": ["deadline"],
        "success_rules": ["one bounded unit completed"],
        "expected_artifacts": [f"reports/campaign-workers/{run_id}/result.json"],
        "max_output_bytes": 1024,
        "verified_files": {
            "source.json": HASHES[0],
            "requirements.lock": HASHES[1],
            "config.json": HASHES[2],
            "input.json": HASHES[3],
            "split.json": HASHES[4],
        },
        "identity_file_bindings": {
            "source.source_bundle_sha256": "source.json",
            "source.dependency_manifest_sha256": "requirements.lock",
            "source.effective_config_sha256": "config.json",
            "data.input_manifest_sha256": "input.json",
            "data.split_manifest_sha256": "split.json",
        },
        "source_file_sha256": {
            "worker.py": HASHES[5],
            "scripts/campaign_worker.py": HASHES[6],
            "src/biohub_ct/campaign/worker_ticket.py": HASHES[7],
            "src/biohub_ct/campaign/watchdog.py": HASHES[7],
            "src/biohub_ct/campaign/admission.py": HASHES[8],
            "src/biohub_ct/campaign/contracts.py": HASHES[8],
        },
    }


def ready_store(tmp_path, *, unit: str = "instance_hours") -> CampaignStore:
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    store.initialize_campaign(
        "campaign-dispatch-test",
        "biohub-cell-tracking-during-development",
        status="ACTIVE",
        authorization={
            "routine_runs_and_submissions": False,
            "existing_allocation_authorized": True,
            "source": "explicit authorization for the bounded existing-allocation campaign",
        },
    )
    store.create_ledger(
        "compute",
        resource_scope="existing Vast instance",
        unit=unit,
        authorized_total="10",
    )
    return store


def live_inventory(spec: dict) -> dict:
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "provider": {
            "actual_status": "running",
            "id": spec["execution"]["provider_instance_id"],
            "dph_total": "0.50",
        },
        "gpu_processes": "",
        "disk": {"free": 20 * 1024**3},
    }


def preflight_files(spec: dict) -> dict:
    return {
        "verified_files": dict(spec["verified_files"]),
        "verified_source_files": dict(spec["source_file_sha256"]),
    }


class FakeTransport:
    def __init__(self, store: CampaignStore, spec: dict, *, observe=None,
                 dispatch_error: Exception | None = None, receipt_changes: dict | None = None,
                 preflight_changes: dict | None = None):
        self.store = store
        self.spec = spec
        self.observe = observe
        self.dispatch_error = dispatch_error
        self.receipt_changes = receipt_changes or {}
        self.preflight_changes = preflight_changes or {}
        self.calls: list[str] = []
        self.ticket = None

    def _record(self, operation: str) -> None:
        self.calls.append(operation)
        if self.observe is not None:
            self.observe(operation)

    def stage(self, ticket_path, run_id):
        self._record("stage")
        assert ticket_path.is_file()
        assert run_id == self.spec["run_id"]
        self.ticket = json.loads(ticket_path.read_text())
        return "/remote/ticket.json"

    def command(self, operation, remote_ticket):
        self._record(operation)
        assert remote_ticket == "/remote/ticket.json"
        digest = run_spec_digest(self.spec)
        intent_id = "launch-" + self.spec["run_id"]
        assert self.ticket is not None
        if operation == "preflight":
            result = {
                "preflight": "PASS",
                "run_id": self.spec["run_id"],
                "intent_id": intent_id,
                "fencing_token": self.ticket["fencing_token"],
                "run_spec_sha256": digest,
                "files": preflight_files(self.spec),
            }
            result.update(self.preflight_changes)
            return result
        intent = self.store.get_intent(intent_id)
        if self.dispatch_error is not None:
            raise self.dispatch_error
        result = {
            "status": "DISPATCHED",
            "run_id": self.spec["run_id"],
            "intent_id": intent_id,
            "run_spec_sha256": digest,
            "fencing_token": intent["fencing_token"],
            "provider_job_id": "linux-supervisor-123",
            "process_identity": {"pid": 123, "start_time_ticks": 99},
        }
        result.update(self.receipt_changes)
        return result


def launch(store, transport, spec, tmp_path, *, amount="1"):
    return launch_existing_vast(
        store,
        transport,
        spec,
        owner="transport-reviewer",
        ledger_id="compute",
        reservation_amount=amount,
        receipts=tmp_path / "receipts",
        inventory=lambda: live_inventory(spec),
    )


def test_persists_reservation_and_claim_before_remote_dispatch(tmp_path):
    store = ready_store(tmp_path)
    spec = run_spec()

    def observe(operation):
        if operation == "dispatch":
            assert store.get_intent("launch-run-1")["state"] == "DISPATCHED"
            assert store.budget_snapshot("compute").outstanding_reservations == Decimal(1)

    transport = FakeTransport(store, spec, observe=observe)
    receipt = launch(store, transport, spec, tmp_path)

    assert transport.calls == ["stage", "preflight", "dispatch"]
    assert receipt["status"] == "DISPATCHED"
    assert store.get_intent("launch-run-1")["state"] == "CONFIRMED"
    assert store.get_run("run-1")["state"] == "RUNNING"


def test_dispatch_timeout_is_unknown_reserved_and_never_retried(tmp_path):
    store = ready_store(tmp_path)
    spec = run_spec()
    transport = FakeTransport(store, spec, dispatch_error=TimeoutError("lost response"))

    receipt = launch(store, transport, spec, tmp_path)

    assert receipt == {
        "status": "LAUNCH_UNKNOWN",
        "error_type": "TimeoutError",
        "retry_allowed": False,
    }
    assert store.get_intent("launch-run-1")["state"] == "UNKNOWN"
    assert store.budget_snapshot("compute").outstanding_reservations == Decimal(1)
    with pytest.raises(AdmissionError, match="active or unresolved"):
        launch(store, transport, spec, tmp_path)
    assert transport.calls.count("dispatch") == 1


@pytest.mark.parametrize(
    ("spec_id", "provider_id"),
    [(456, 999), (None, None)],
)
def test_provider_identity_must_be_present_and_exact(tmp_path, spec_id, provider_id):
    store = ready_store(tmp_path)
    spec = run_spec()
    spec["execution"]["provider_instance_id"] = spec_id
    transport = FakeTransport(store, spec)
    live = live_inventory(run_spec())
    live["provider"]["id"] = provider_id

    with pytest.raises(AdmissionError, match="identity or status"):
        launch_existing_vast(
            store, transport, spec, owner="transport-reviewer", ledger_id="compute",
            reservation_amount="1", receipts=tmp_path / "receipts", inventory=lambda: live,
        )
    assert transport.calls == []
    assert store.budget_snapshot("compute").outstanding_reservations == 0


@pytest.mark.parametrize("blocked", ["gpu", "disk"])
def test_gpu_concurrency_and_disk_reserve_gate_before_transport(tmp_path, blocked):
    store = ready_store(tmp_path)
    spec = run_spec()
    transport = FakeTransport(store, spec)
    live = live_inventory(spec)
    if blocked == "gpu":
        live["gpu_processes"] = "123, python, 4096"
    else:
        live["disk"]["free"] = 15 * 1024**3 + spec["max_output_bytes"] - 1

    with pytest.raises(AdmissionError):
        launch_existing_vast(
            store, transport, spec, owner="transport-reviewer", ledger_id="compute",
            reservation_amount="1", receipts=tmp_path / "receipts", inventory=lambda: live,
        )
    assert transport.calls == []
    assert store.budget_snapshot("compute").outstanding_reservations == 0


@pytest.mark.parametrize("changed", ["provider", "gpu", "disk", "rate"])
def test_resource_gates_are_rechecked_after_preflight(tmp_path, changed):
    unit = "USD" if changed == "rate" else "instance_hours"
    amount = "0.02" if changed == "rate" else "1"
    store = ready_store(tmp_path, unit=unit)
    spec = run_spec()
    transport = FakeTransport(store, spec)
    first = live_inventory(spec)
    refreshed = live_inventory(spec)
    if changed == "provider":
        refreshed["provider"]["id"] = 999
    elif changed == "gpu":
        refreshed["gpu_processes"] = "123, python, 4096"
    elif changed == "disk":
        refreshed["disk"]["free"] = 15 * 1024**3 + spec["max_output_bytes"] - 1
    else:
        refreshed["provider"]["dph_total"] = "1.00"
    observations = iter((first, refreshed))

    with pytest.raises(AdmissionError):
        launch_existing_vast(
            store, transport, spec, owner="transport-reviewer", ledger_id="compute",
            reservation_amount=amount, receipts=tmp_path / "receipts",
            inventory=lambda: next(observations),
        )
    assert transport.calls == ["stage", "preflight"]
    assert store.budget_snapshot("compute").outstanding_reservations == 0


def test_running_store_job_blocks_same_host_before_transport(tmp_path):
    store = ready_store(tmp_path)
    prior = run_spec("prior-run")
    registered = store.register_run(prior)
    with store.review_lock("prior-reviewer") as lease:
        intent = store.authorize_launch(
            "prior-run", decision_id="prior-decision", reviewer="prior-reviewer",
            reservations=[{"ledger_id": "compute", "reservation_id": "prior-reservation", "amount": "1"}],
            reason="legitimate bounded prior launch", intent_id="launch-prior-run",
            request_id="request-prior-run", description_tag="prior-run",
            fencing_token=lease.fencing_token,
        )
        store.validate_dispatch(
            intent["intent_id"], "prior-run", registered["run_spec_sha256"], lease.fencing_token,
        )
        store.confirm_launch(intent["intent_id"], provider_job_id="prior-provider-job", receipt={})
    spec = run_spec()
    transport = FakeTransport(store, spec)

    with pytest.raises(AdmissionError, match="active or unresolved"):
        launch(store, transport, spec, tmp_path)
    assert transport.calls == []


@pytest.mark.parametrize(
    ("unit", "amount"),
    [("instance_hours", "1"), ("USD", "0.5")],
)
def test_reservation_covers_walltime_plus_shutdown_grace(tmp_path, unit, amount):
    store = ready_store(tmp_path, unit=unit)
    spec = run_spec()
    spec["execution"]["max_wall_seconds"] = 3600
    transport = FakeTransport(store, spec)

    with pytest.raises(AdmissionError, match="below worst-case"):
        launch(store, transport, spec, tmp_path, amount=amount)
    assert transport.calls == []
    assert store.budget_snapshot("compute").outstanding_reservations == 0


@pytest.mark.parametrize(
    "preflight_changes",
    [
        {"intent_id": "launch-other"},
        {"fencing_token": 999},
        {"files": {"verified_files": {"source.json": "0" * 64},
                   "verified_source_files": run_spec()["source_file_sha256"]}},
        {"files": {"verified_files": run_spec()["verified_files"],
                   "verified_source_files": {"worker.py": "0" * 64}}},
    ],
)
def test_preflight_identity_and_verified_files_must_match_ticket(tmp_path, preflight_changes):
    store = ready_store(tmp_path)
    spec = run_spec()
    transport = FakeTransport(store, spec, preflight_changes=preflight_changes)

    with pytest.raises(AdmissionError, match="preflight"):
        launch(store, transport, spec, tmp_path)
    assert "dispatch" not in transport.calls


@pytest.mark.parametrize(
    "receipt_changes",
    [{"intent_id": "launch-other"}, {"run_id": "other-run"}],
)
def test_dispatch_receipt_is_bound_to_exact_intent_and_run(tmp_path, receipt_changes):
    store = ready_store(tmp_path)
    spec = run_spec()
    transport = FakeTransport(store, spec, receipt_changes=receipt_changes)

    receipt = launch(store, transport, spec, tmp_path)

    assert receipt["status"] == "LAUNCH_UNKNOWN"
    assert receipt["retry_allowed"] is False
    assert store.get_intent("launch-run-1")["state"] == "UNKNOWN"
    assert store.get_run("run-1")["state"] == "LAUNCH_UNKNOWN"
    assert store.budget_snapshot("compute").outstanding_reservations == Decimal(1)


def completed_download_fixture(tmp_path):
    spec = run_spec()
    prefix = Path("reports/campaign-workers/run-1")
    spec["result_manifest_path"] = (prefix / "result.json").as_posix()
    spec["expected_artifacts"] = [(prefix / "answer.bin").as_posix()]
    remote = tmp_path / "remote"
    worker = remote / prefix / "worker"
    worker.mkdir(parents=True)
    (worker / "output.log").write_bytes(b"bounded output\n")
    (remote / prefix / "answer.bin").write_bytes(b"verified artifact")
    (remote / prefix / "result.json").write_text('{"status":"COMPLETE"}')
    completion = {
        "run_id": "run-1",
        "run_spec_sha256": run_spec_digest(spec),
        "fencing_token": 4,
        "status": "COMPLETE",
        "output_log_sha256": file_sha256(worker / "output.log"),
    }
    (worker / "completion.json").write_text(json.dumps(completion))
    return spec, completion, remote


class RetrievalRunner:
    def __init__(
        self,
        remote: Path,
        *,
        corrupt_on_copy: str | None = None,
        inventory_stdout: str | None = None,
        malformed_entry: str | None = None,
    ):
        self.remote = remote
        self.corrupt_on_copy = corrupt_on_copy
        self.inventory_stdout = inventory_stdout
        self.malformed_entry = malformed_entry
        self.calls = []

    def __call__(self, argv, **_kwargs):
        self.calls.append(argv)
        if argv[0] == "ssh":
            names = [
                "reports/campaign-workers/run-1/worker/completion.json",
                "reports/campaign-workers/run-1/worker/output.log",
                "reports/campaign-workers/run-1/result.json",
                "reports/campaign-workers/run-1/answer.bin",
            ]
            described = {}
            for name in names:
                path = self.remote / name
                if path.is_file():
                    described[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            if self.malformed_entry is not None:
                described[self.malformed_entry] = {}
            stdout = self.inventory_stdout or json.dumps(described)
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        source = argv[-2].split(":", 1)[1].strip("'")
        name = source.split("/remote/root/", 1)[1]
        destination = Path(argv[-1])
        if name == self.corrupt_on_copy:
            destination.write_bytes(b"changed during transfer")
        else:
            shutil.copyfile(self.remote / name, destination)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_retrieve_completed_downloads_exact_files_with_hash_and_size_checks(tmp_path):
    spec, completion, remote = completed_download_fixture(tmp_path)
    runner = RetrievalRunner(remote)
    transport = ExistingVastTransport(
        tmp_path / "ssh.config", "/remote/root", runner=runner
    )
    destination = tmp_path / "downloaded"

    receipt = transport.retrieve_completed(spec, completion, destination)

    assert receipt == (
        destination / "reports/campaign-workers/run-1/worker/completion.json"
    )
    assert json.loads(receipt.read_text()) == completion
    assert (destination / spec["expected_artifacts"][0]).read_bytes() == b"verified artifact"
    assert len(runner.calls) == 5

    # A read-only reconciliation reuses only bytes matching fresh remote hashes.
    transport.retrieve_completed(spec, completion, destination)
    assert len(runner.calls) == 6


def test_retrieve_rejects_attempt_path_escape_before_network(tmp_path):
    spec, completion, remote = completed_download_fixture(tmp_path)
    spec["expected_artifacts"] = ["reports/campaign-workers/run-1/../outside.bin"]
    runner = RetrievalRunner(remote)
    transport = ExistingVastTransport(
        tmp_path / "ssh.config", "/remote/root", runner=runner
    )

    with pytest.raises(AdmissionError, match="approved attempt"):
        transport.retrieve_completed(spec, completion, tmp_path / "downloaded")
    assert runner.calls == []


def test_retrieve_rejects_file_changed_after_remote_inventory(tmp_path):
    spec, completion, remote = completed_download_fixture(tmp_path)
    changed = "reports/campaign-workers/run-1/answer.bin"
    runner = RetrievalRunner(remote, corrupt_on_copy=changed)
    transport = ExistingVastTransport(
        tmp_path / "ssh.config", "/remote/root", runner=runner
    )

    with pytest.raises(AdmissionError, match="hash or size"):
        transport.retrieve_completed(spec, completion, tmp_path / "downloaded")


def test_retrieve_rejects_completion_changed_since_observation(tmp_path):
    spec, completion, remote = completed_download_fixture(tmp_path)
    observed = dict(completion)
    completion["status"] = "FAILED"
    completion_path = remote / "reports/campaign-workers/run-1/worker/completion.json"
    completion_path.write_text(json.dumps(completion))
    runner = RetrievalRunner(remote)
    transport = ExistingVastTransport(
        tmp_path / "ssh.config", "/remote/root", runner=runner
    )

    with pytest.raises(AdmissionError, match="changed between observation"):
        transport.retrieve_completed(spec, observed, tmp_path / "downloaded")


def test_retrieve_detects_boolean_integer_completion_identity_alias(tmp_path):
    spec, observed, remote = completed_download_fixture(tmp_path)
    observed["fencing_token"] = True
    remote_completion = dict(observed)
    remote_completion["fencing_token"] = 1
    completion_path = remote / "reports/campaign-workers/run-1/worker/completion.json"
    completion_path.write_text(json.dumps(remote_completion))
    runner = RetrievalRunner(remote)
    transport = ExistingVastTransport(
        tmp_path / "ssh.config", "/remote/root", runner=runner
    )

    with pytest.raises(AdmissionError, match="completion|changed"):
        transport.retrieve_completed(spec, observed, tmp_path / "downloaded")


@pytest.mark.parametrize("malformed", ["invalid-json", "missing-metadata"])
def test_retrieve_normalizes_malformed_remote_inventory_to_admission_error(tmp_path, malformed):
    spec, completion, remote = completed_download_fixture(tmp_path)
    completion_name = "reports/campaign-workers/run-1/worker/completion.json"
    runner = RetrievalRunner(
        remote,
        inventory_stdout="not-json" if malformed == "invalid-json" else None,
        malformed_entry=completion_name if malformed == "missing-metadata" else None,
    )
    transport = ExistingVastTransport(
        tmp_path / "ssh.config", "/remote/root", runner=runner
    )

    with pytest.raises(AdmissionError, match="inventory"):
        transport.retrieve_completed(spec, completion, tmp_path / "downloaded")


def test_retrieve_independently_enforces_total_local_size_bound(tmp_path):
    spec, completion, remote = completed_download_fixture(tmp_path)
    artifact = remote / spec["expected_artifacts"][0]
    artifact.write_bytes(b"x" * (20 * 1024**2 + spec["max_output_bytes"] + 1))
    runner = RetrievalRunner(remote)
    transport = ExistingVastTransport(
        tmp_path / "ssh.config", "/remote/root", runner=runner
    )

    with pytest.raises(AdmissionError, match="size bound"):
        transport.retrieve_completed(spec, completion, tmp_path / "downloaded")


def test_retrieve_rejects_preexisting_temporary_symlink(tmp_path):
    spec, completion, remote = completed_download_fixture(tmp_path)
    destination = tmp_path / "downloaded"
    worker = destination / "reports/campaign-workers/run-1/worker"
    worker.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("do not modify")
    try:
        (worker / "completion.json.download").symlink_to(outside)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account lacks symlink privilege; exercised on Linux")
        raise
    runner = RetrievalRunner(remote)
    transport = ExistingVastTransport(
        tmp_path / "ssh.config", "/remote/root", runner=runner
    )

    with pytest.raises(AdmissionError, match="temporary|download root"):
        transport.retrieve_completed(spec, completion, destination)
    assert outside.read_text() == "do not modify"


def test_retrieve_rejects_preexisting_temporary_hardlink(tmp_path):
    spec, completion, remote = completed_download_fixture(tmp_path)
    destination = tmp_path / "downloaded"
    worker = destination / "reports/campaign-workers/run-1/worker"
    worker.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("do not modify")
    os.link(outside, worker / "completion.json.download")
    runner = RetrievalRunner(remote)
    transport = ExistingVastTransport(
        tmp_path / "ssh.config", "/remote/root", runner=runner
    )

    with pytest.raises(AdmissionError, match="temporary|download root"):
        transport.retrieve_completed(spec, completion, destination)
    assert outside.read_text() == "do not modify"
