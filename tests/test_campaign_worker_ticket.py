import json
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from biohub_ct.campaign.admission import AdmissionError, file_sha256
from biohub_ct.campaign.contracts import run_spec_digest
from biohub_ct.campaign.worker_ticket import dispatch_ticket, execute_ticket, validate_ticket

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Actual worker is Linux")


def fixture(tmp_path):
    code = '''import hashlib, json, os
from pathlib import Path
root = Path.cwd()
attempt = Path(os.environ["BIOHUB_ATTEMPT_DIR"])
result = attempt / "answer.txt"
result.write_text(str(sum(n*n for n in range(8))))
identity = {"run_id": os.environ["BIOHUB_RUN_ID"], "run_spec_sha256": os.environ["BIOHUB_RUN_SPEC_SHA256"],
    "intent_id":os.environ["BIOHUB_INTENT_ID"], "fencing_token":int(os.environ["BIOHUB_FENCING_TOKEN"])}
(attempt / "result.json").write_text(json.dumps({**identity, "status":"COMPLETE", "completed_units":1,
    "artifact_sha256":{result.relative_to(root).as_posix():hashlib.sha256(result.read_bytes()).hexdigest()}}))
Path(os.environ["BIOHUB_PROGRESS_PATH"]).write_text(json.dumps({**identity,"completed_units":1,"error":None}))
'''
    (tmp_path / "worker.py").write_text(code)
    actual_root = Path(__file__).resolve().parents[1]
    names = ["scripts/campaign_worker.py", *("src/biohub_ct/campaign/" + name for name in
             ("worker_ticket.py", "watchdog.py", "admission.py", "contracts.py"))]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual_root / name, tmp_path / name)
    (tmp_path / "source.json").write_text(json.dumps({name: file_sha256(tmp_path / name) for name in ["worker.py", *names]}))
    (tmp_path / "config.json").write_text("{}")
    source_hash, other = file_sha256(tmp_path / "source.json"), file_sha256(tmp_path / "config.json")
    spec = {"run_id": "run-test", "experiment_id": "operational-smoke", "work_kind": "cpu-verification",
        "source_file_sha256": json.loads((tmp_path / "source.json").read_text()),
        "source": {"source_bundle_sha256": source_hash, "dependency_manifest_sha256": other, "effective_config_sha256": other},
        "data": {"input_manifest_sha256": other, "split_manifest_sha256": other},
        "execution": {"host": "test-linux", "working_directory": str(tmp_path), "argv": [sys.executable, str(tmp_path / "worker.py")],
            "entrypoint_file": "worker.py", "executable_sha256": file_sha256(Path(sys.executable)),
            "max_wall_seconds": 5, "max_steps_or_clips": 1,
            "worker_progress_path": str(tmp_path / "reports/campaign-workers/run-test/progress.json"),
            "deadline_utc": (datetime.now(UTC) + timedelta(seconds=30)).isoformat()},
        "stop_rules": ["deadline"], "success_rules": ["arithmetic matches"],
        "expected_artifacts": ["reports/campaign-workers/run-test/answer.txt"], "expected_completed_units": 1,
        "result_manifest_path": "reports/campaign-workers/run-test/result.json",
        "verified_files": {"source.json": source_hash, "config.json": other},
        "identity_file_bindings": {"source.source_bundle_sha256": "source.json",
            "source.dependency_manifest_sha256": "config.json", "source.effective_config_sha256": "config.json",
            "data.input_manifest_sha256": "config.json", "data.split_manifest_sha256": "config.json"}}
    ticket = {"spec": spec, "run_spec_sha256": run_spec_digest(spec), "fencing_token": 1, "intent_id": "intent-1"}
    ticket_path = tmp_path / "ticket.json"
    ticket_path.write_text(json.dumps(ticket))
    return ticket, ticket_path


def test_actual_bounded_worker_with_identity_bound_outputs(tmp_path):
    _ticket, path = fixture(tmp_path)
    attempt = tmp_path / "reports/campaign-workers/run-test"
    attempt.mkdir(parents=True)
    shutil.copyfile(path, attempt / "ticket.json")
    path = attempt / "ticket.json"
    result = execute_ticket(path, tmp_path)
    assert result["status"] == "COMPLETE"
    assert (attempt / "answer.txt").read_text() == "140"
    assert result["artifacts"]["quality_review_required"] is True
    with pytest.raises(AdmissionError, match="existing result"):
        execute_ticket(path, tmp_path)


def test_ticket_rejects_changed_entrypoint_runtime_and_hash(tmp_path):
    ticket, _ = fixture(tmp_path)
    validate_ticket(ticket, tmp_path)
    (tmp_path / "worker.py").write_text("raise SystemExit(1)")
    with pytest.raises(AdmissionError, match="hash mismatch"):
        validate_ticket(ticket, tmp_path)


def test_remote_duplicate_ticket_never_spawns_twice(tmp_path, monkeypatch):
    ticket, path = fixture(tmp_path)
    calls = []
    class Process:
        pid = os.getpid()
    def launch(*args, **kwargs):
        calls.append(args)
        return Process()
    monkeypatch.setattr("biohub_ct.campaign.worker_ticket.subprocess.Popen", launch)
    first = dispatch_ticket(path, tmp_path, supervisor_script=tmp_path / "scripts/campaign_worker.py")
    second = dispatch_ticket(path, tmp_path, supervisor_script=tmp_path / "scripts/campaign_worker.py")
    assert first == second
    assert len(calls) == 1
    ticket["fencing_token"] = 2
    path.write_text(json.dumps(ticket))
    with pytest.raises(AdmissionError, match="different immutable"):
        dispatch_ticket(path, tmp_path, supervisor_script=tmp_path / "scripts/campaign_worker.py")


def test_interrupted_remote_claim_stays_unknown(tmp_path, monkeypatch):
    _, path = fixture(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("simulated failure after durable claim")
    monkeypatch.setattr("biohub_ct.campaign.worker_ticket.subprocess.Popen", fail)
    with pytest.raises(OSError):
        dispatch_ticket(path, tmp_path, supervisor_script=tmp_path / "scripts/campaign_worker.py")
    result = dispatch_ticket(path, tmp_path, supervisor_script=tmp_path / "scripts/campaign_worker.py")
    assert result["status"] == "LAUNCH_UNKNOWN" and not result["retry_allowed"]


def test_result_path_escape_and_receipt_substitution_rejected(tmp_path, monkeypatch):
    ticket, path = fixture(tmp_path)
    ticket["spec"]["result_manifest_path"] = "../outside.json"
    ticket["run_spec_sha256"] = run_spec_digest(ticket["spec"])
    with pytest.raises(AdmissionError, match="exclusive attempt"):
        validate_ticket(ticket, tmp_path)
    ticket, path = fixture(tmp_path)
    class Process:
        pid = os.getpid()
    monkeypatch.setattr("biohub_ct.campaign.worker_ticket.subprocess.Popen", lambda *a, **k: Process())
    dispatch_ticket(path, tmp_path, supervisor_script=tmp_path / "scripts/campaign_worker.py")
    receipt_path = tmp_path / "reports/campaign-workers/run-test/dispatch.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["fencing_token"] += 1
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(AdmissionError, match="receipt differs"):
        dispatch_ticket(path, tmp_path, supervisor_script=tmp_path / "scripts/campaign_worker.py")
