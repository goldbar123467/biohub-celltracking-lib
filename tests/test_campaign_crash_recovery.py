"""Real process-death checks for transactional state and atomic review exports."""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from biohub_ct.campaign.state import CampaignStateError, CampaignStore
from test_campaign_state import ready_store

ROOT = Path(__file__).resolve().parents[1]
CRASH_WORKER = r"""
import json
import sys
from pathlib import Path
import biohub_ct.campaign.state as state_module
from biohub_ct.campaign.state import CampaignStore

store = CampaignStore(sys.argv[1])
mode, destination = sys.argv[2], Path(sys.argv[3])
with store.review_lock("crash-owner") as lease:
    def boundary():
        print(json.dumps({"mode": mode, "fencing_token": lease.fencing_token}), flush=True)
        sys.stdin.buffer.read(1)
        raise RuntimeError("The parent must kill this process at the verified boundary")

    store.create_ledger("committed", resource_scope="test", unit="instance_hours",
                        authorized_total="3")
    if mode == "transaction":
        original_event = store._event
        def paused_event(connection, event_type, **kwargs):
            result = original_event(connection, event_type, **kwargs)
            if event_type == "LEDGER_CREATED" and kwargs.get("subject_id") == "uncommitted":
                boundary()
            return result
        store._event = paused_event
        store.create_ledger("uncommitted", resource_scope="test", unit="instance_hours",
                            authorized_total="9")
    else:
        original_replace = state_module.os.replace
        def paused_replace(source, target):
            assert Path(target) == destination
            if mode == "export-after":
                original_replace(source, target)
            boundary()
        state_module.os.replace = paused_replace
        store.export_json(destination)
"""


def kill_at_verified_boundary(store: CampaignStore, mode: str, destination: Path) -> dict:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with subprocess.Popen(
        [sys.executable, "-c", CRASH_WORKER, str(store.path), mode, str(destination)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        cwd=ROOT,
        text=True,
    ) as process:
        lines: queue.Queue[str] = queue.Queue()
        reader = threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True)
        reader.start()
        try:
            line = lines.get(timeout=15)
            if not line:
                raise AssertionError("Crash worker exited before the intended boundary")
            boundary = json.loads(line)
            assert boundary["mode"] == mode
            assert process.poll() is None
            process.kill()
            assert process.wait(timeout=10) != 0
            return boundary
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            reader.join(timeout=2)


def assert_recovered_store(store: CampaignStore, boundary: dict) -> CampaignStore:
    recovered = CampaignStore(store.path)
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert recovered.budget_snapshot("committed").authorized_total == 3
    with recovered.review_lock("replacement-reviewer") as lease:
        assert lease.fencing_token > boundary["fencing_token"]
    events = recovered.list_events()
    assert [event["state_version_after"] for event in events] == list(range(1, len(events) + 1))
    return recovered


def test_killed_transaction_rolls_back_ledger_and_event_but_keeps_prior_commit(tmp_path):
    store = ready_store(tmp_path)
    boundary = kill_at_verified_boundary(store, "transaction", tmp_path / "unused.json")
    recovered = assert_recovered_store(store, boundary)
    with pytest.raises(KeyError):
        recovered.budget_snapshot("uncommitted")
    assert not any(event["subject_id"] == "uncommitted" for event in recovered.list_events())


@pytest.mark.parametrize("mode", ["export-before", "export-after"])
def test_killed_export_preserves_complete_old_or_new_snapshot_and_releases_os_lock(tmp_path, mode):
    store = ready_store(tmp_path)
    destination = store.export_json(tmp_path / "campaign.json")
    previous_bytes = destination.read_bytes()
    boundary = kill_at_verified_boundary(store, mode, destination)
    assert_recovered_store(store, boundary)
    actual_bytes = destination.read_bytes()
    exported = json.loads(actual_bytes)
    assert exported["state_version"] == exported["events"][-1]["state_version_after"]
    ledgers = {ledger["ledger_id"] for ledger in exported["ledgers"]}
    if mode == "export-before":
        assert actual_bytes == previous_bytes
        assert "committed" not in ledgers
    else:
        assert actual_bytes != previous_bytes
        assert "committed" in ledgers
    # A later normal export must recover even if process death left a temporary file.
    store.export_json(destination)
    assert "committed" in {
        item["ledger_id"] for item in json.loads(destination.read_text())["ledgers"]
    }


def test_export_rejects_database_journal_and_review_lock_destinations(tmp_path):
    store = ready_store(tmp_path)
    protected = [store.path, store.path.with_suffix(".review.lock")]
    protected.extend(Path(str(store.path) + suffix) for suffix in ["-wal", "-shm", "-journal"])
    for destination in protected:
        before = destination.read_bytes() if destination.exists() else None
        with pytest.raises(CampaignStateError, match="canonical store or lock"):
            store.export_json(destination)
        after = destination.read_bytes() if destination.exists() else None
        assert before == after
    assert store.get_campaign()["campaign_id"] == "biohub-2026-09-08"
