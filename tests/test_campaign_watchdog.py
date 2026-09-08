from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from biohub_ct.campaign.watchdog import (
    WorkerError,
    _stop_owned_group,
    process_identity,
    run_bounded_job,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/stat").exists(),
    reason="Worker runs on Linux Vast; Windows is the controller",
)
DIGEST = "b" * 64


def specification(tmp_path, code, *, max_wall=5):
    script = tmp_path / "worker.py"
    script.write_text(code)
    return {
        "run_id": "test-worker",
        "execution": {
            "argv": [sys.executable, str(script)],
            "working_directory": str(tmp_path),
            "max_wall_seconds": max_wall,
            "max_steps_or_clips": 5,
            "deadline_utc": (datetime.now(UTC) + timedelta(seconds=20)).isoformat(),
            "nonsecret_environment": {},
        },
    }


def execute(spec, output, **kwargs):
    return run_bounded_job(
        spec, output, run_spec_sha256=DIGEST, fencing_token=1,
        validate_dispatch=kwargs.pop("validate_dispatch", lambda: None),
        poll_seconds=0.02, heartbeat_seconds=0.05, shutdown_grace_seconds=0.15,
        **kwargs,
    )


def test_zero_exit_requires_artifact_verification_and_literal_argv(tmp_path):
    spec = specification(tmp_path, "import sys; print(sys.argv[1])")
    spec["execution"]["argv"].append("$(touch UNAUTHORIZED); echo unsafe")
    result = execute(spec, tmp_path / "out")
    assert result["status"] == "PARTIAL"
    assert result["exit_code"] == 0
    assert not (tmp_path / "UNAUTHORIZED").exists()
    assert "$(touch UNAUTHORIZED)" in (tmp_path / "out/output.log").read_text()
    verified = execute(spec, tmp_path / "verified", validate_artifacts=lambda: {"complete": True})
    assert verified["status"] == "COMPLETE"


def test_final_unit_may_flush_and_exit_within_bounded_grace(tmp_path):
    progress = tmp_path / "progress.json"
    code = f"""import json,time
from pathlib import Path
Path({str(progress)!r}).write_text(json.dumps({{'run_id':'test-worker','run_spec_sha256':{DIGEST!r},'completed_units':5}}))
time.sleep(0.08)
Path('artifact.txt').write_text('finished')
"""
    spec = specification(tmp_path, code)
    spec["execution"]["worker_progress_path"] = str(progress)
    def validate():
        return {"complete": (tmp_path / "artifact.txt").read_text() == "finished"}
    result = execute(spec, tmp_path / "out", validate_artifacts=validate)
    assert result["status"] == "COMPLETE"
    assert result["stop_reason"] is None


def test_rejected_dispatch_does_not_launch_or_create_output(tmp_path):
    spec = specification(tmp_path, "from pathlib import Path; Path('launched').touch()")
    def reject():
        raise WorkerError("stale token")
    with pytest.raises(WorkerError, match="stale token"):
        execute(spec, tmp_path / "out", validate_dispatch=reject)
    assert not (tmp_path / "launched").exists()
    assert not (tmp_path / "out").exists()


def test_deadline_kills_ignoring_worker_and_surviving_descendant(tmp_path):
    spec = specification(tmp_path, """import signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
time.sleep(60)
""", max_wall=0.4)
    result = execute(spec, tmp_path / "out")
    assert result["status"] == "STOPPED"
    assert result["stop_reason"] == "deadline"
    assert result["elapsed_wall_seconds"] < 3
    child_pid = int((tmp_path / "child.pid").read_text())
    stat_path = Path(f"/proc/{child_pid}/stat")
    if stat_path.exists():
        stat = stat_path.read_text()
        assert stat[stat.rfind(")") + 2 :].split()[0] == "Z"
    receipt = json.loads((tmp_path / "out/completion.json").read_text())
    assert receipt["run_spec_sha256"] == DIGEST


def test_stale_pid_identity_cannot_kill_unrelated_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
    try:
        identity = process_identity(process.pid)
        identity["start_ticks"] -= 1
        with pytest.raises(WorkerError, match="identity"):
            _stop_owned_group(process, identity, 0.1)
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=3)


def test_work_units_stop_before_wall_limit_and_progress_identity_is_checked(tmp_path):
    progress = tmp_path / "progress.json"
    code = f"""import json, time
from pathlib import Path
Path({str(progress)!r}).write_text(json.dumps({{'run_id':'test-worker','run_spec_sha256':{DIGEST!r},'completed_units':5}}))
time.sleep(10)
"""
    spec = specification(tmp_path, code)
    spec["execution"]["worker_progress_path"] = str(progress)
    result = execute(spec, tmp_path / "out")
    assert result["stop_reason"] == "work_unit_limit"
    assert result["elapsed_wall_seconds"] < 2
    progress.write_text(json.dumps({"run_id": "wrong", "run_spec_sha256": DIGEST, "completed_units": 2}))
    spec = specification(tmp_path, "import time; time.sleep(10)")
    spec["execution"]["worker_progress_path"] = str(progress)
    result = execute(spec, tmp_path / "bad-progress")
    assert result["status"] == "FAILED"
    assert "progress identity mismatch" in result["error"]["message"]


def test_output_directory_cannot_be_reused_and_bad_artifacts_fail(tmp_path):
    spec = specification(tmp_path, "print('done')")
    result = execute(spec, tmp_path / "out", validate_artifacts=lambda: {"complete": False})
    assert result["status"] == "FAILED"
    assert result["stop_reason"] == "artifact_validation_failed"
    with pytest.raises(FileExistsError):
        execute(spec, tmp_path / "out")


def test_orphaned_child_is_stopped_even_after_worker_exits(tmp_path):
    spec = specification(tmp_path, """import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
Path('child.pid').write_text(str(child.pid))
""")
    result = execute(spec, tmp_path / "out", validate_artifacts=lambda: {"complete": True})
    assert result["status"] == "FAILED"
    assert result["stop_reason"] == "orphaned_descendants"
    child_pid = int((tmp_path / "child.pid").read_text())
    path = Path(f"/proc/{child_pid}/stat")
    if path.exists():
        stat = path.read_text()
        assert stat[stat.rfind(")") + 2 :].split()[0] == "Z"


def test_backstop_stops_worker_after_supervisor_is_killed(tmp_path):
    spec = specification(tmp_path, """import os, time
from pathlib import Path
Path('actual-worker.pid').write_text(str(os.getpid()))
time.sleep(60)
""", max_wall=1.2)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    supervisor_code = """import json, sys
from pathlib import Path
from biohub_ct.campaign.watchdog import run_bounded_job
run_bounded_job(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]),
    run_spec_sha256='b'*64, fencing_token=1, validate_dispatch=lambda: None,
    poll_seconds=0.02, heartbeat_seconds=0.05, shutdown_grace_seconds=0.15)
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in sys.path if p)}
    supervisor = subprocess.Popen([sys.executable, "-c", supervisor_code, str(spec_path), str(tmp_path / "out")], env=env)
    child_pid = None
    child_identity = None
    try:
        wait_until = time.monotonic() + 5
        while not (tmp_path / "actual-worker.pid").exists() and time.monotonic() < wait_until:
            time.sleep(0.02)
        child_pid = int((tmp_path / "actual-worker.pid").read_text())
        child_identity = process_identity(child_pid)
        supervisor.kill()
        supervisor.wait(timeout=3)
        wait_until = time.monotonic() + 4
        live = True
        while time.monotonic() < wait_until:
            path = Path(f"/proc/{child_pid}/stat")
            if not path.exists():
                live = False
                break
            stat = path.read_text()
            if stat[stat.rfind(")") + 2 :].split()[0] == "Z":
                live = False
                break
            time.sleep(0.02)
        assert not live, "Independent GNU timeout failed after supervisor death"
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=3)
        if child_identity is not None and process_identity(child_pid) == child_identity:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("field,value", [("max_wall_seconds", float("nan")), ("max_steps_or_clips", 0), ("deadline_utc", "2000-01-01T00:00:00Z")])
def test_invalid_limits_fail_before_launch(tmp_path, field, value):
    spec = specification(tmp_path, "raise RuntimeError('must not run')")
    spec["execution"][field] = value
    with pytest.raises(WorkerError):
        execute(spec, tmp_path / "out")
    assert not (tmp_path / "out").exists()
