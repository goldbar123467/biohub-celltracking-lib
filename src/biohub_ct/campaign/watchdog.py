"""Deadline supervision for the actual POSIX GPU worker host.

The Windows controller dispatches this worker over SSH. It never relies on the
hourly reviewer to enforce minute-scale limits. No shell parses the job argv.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path


class WorkerError(RuntimeError):
    """Invalid worker configuration or unverifiable process ownership."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    """Durably replace a JSON record; do not allow NaN to masquerade as evidence."""
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def process_identity(pid: int) -> dict | None:
    """Linux PID identity includes boot ID and kernel start ticks, not PID alone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # comm may contain spaces and parentheses. Fields after the last ')'
        # begin with state (field 3); starttime is field 22.
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) < 20:
            raise WorkerError("Malformed process identity record")
        return {
            "pid": pid,
            "process_group": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except FileNotFoundError:
        return None


def identity_matches(expected: Mapping) -> bool:
    return process_identity(int(expected["pid"])) == dict(expected)


def _owned_group_members(identity: Mapping) -> list[dict]:
    """A dedicated session/group ID cannot be recycled while it has members."""
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != identity["boot_id"]:
        raise WorkerError("Boot identity changed; refusing stale process-group ownership")
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        member = process_identity(int(entry.name))
        if member is None or member["process_group"] != identity["process_group"]:
            continue
        if member["session"] != identity["session"] or member["start_ticks"] < identity["start_ticks"]:
            raise WorkerError("Unexpected process-group membership; no group signal sent")
        try:
            stat = (entry / "stat").read_text()
        except FileNotFoundError:
            continue
        if stat[stat.rfind(")") + 2 :].split()[0] != "Z":
            members.append(member)
    return members


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise WorkerError(f"{name} must be a finite positive number")
    return value


def _read_progress(path: Path | None, run_id: str, digest: str) -> dict | None:
    if path is None or not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkerError("Worker progress is unreadable; refusing invented progress") from exc
    if not isinstance(result, dict):
        raise WorkerError("Worker progress must be an object")
    if result.get("run_id") != run_id or result.get("run_spec_sha256") != digest:
        raise WorkerError("Worker progress identity mismatch")
    units = result.get("completed_units")
    if isinstance(units, bool) or not isinstance(units, int) or units < 0:
        raise WorkerError("Worker completed_units must be a nonnegative integer")
    if result.get("error"):
        raise WorkerError("Worker reported an error")
    return result


def _stop_owned_group(process: subprocess.Popen, identity: dict, grace_seconds: float) -> str:
    """Signal only a still-owned child session. Never kill by an old PID file."""
    if process.poll() is None and not identity_matches(identity):
        raise WorkerError("Cannot confirm process identity; no termination signal sent")
    if identity["process_group"] != process.pid or identity["session"] != process.pid:
        raise WorkerError("Refusing to signal a non-dedicated process group")
    if not _owned_group_members(identity):
        return "already_exited"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    stop_at = time.monotonic() + grace_seconds
    while time.monotonic() < stop_at:
        process.poll()
        if not _owned_group_members(identity):
            process.wait(timeout=10)
            return "terminated"
        time.sleep(min(0.05, max(0.001, stop_at - time.monotonic())))
    # The leader may have exited, but its dedicated session/group is still
    # reserved by surviving descendants and checked immediately before signaling.
    if _owned_group_members(identity):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)
    for _ in range(100):
        if not _owned_group_members(identity):
            return "killed_after_grace"
        time.sleep(0.01)
    raise WorkerError("Owned group still has live processes after SIGKILL")


def run_bounded_job(
    spec: Mapping,
    output_dir: Path,
    *,
    run_spec_sha256: str,
    fencing_token: int,
    validate_dispatch: Callable[[], None],
    validate_artifacts: Callable[[], dict] | None = None,
    poll_seconds: float = 0.5,
    heartbeat_seconds: float = 30.0,
    shutdown_grace_seconds: float = 15.0,
) -> dict:
    """Run one already-admitted worker and persist supervision/completion receipts.

    An exit code of zero alone is not COMPLETE: a caller-supplied artifact gate
    must also return {"complete": True, ...}. Otherwise the output is PARTIAL.
    Application progress is distinct from this supervisor's liveness heartbeat.
    """
    if os.name != "posix" or not Path("/proc/sys/kernel/random/boot_id").is_file():
        raise WorkerError("This worker requires Linux /proc; use the verified Vast host")
    timeout_executable = shutil.which("timeout")
    if timeout_executable is None:
        raise WorkerError("GNU timeout is required as an independent deadline backstop")
    timeout_version = subprocess.run(
        [timeout_executable, "--version"], capture_output=True, text=True, timeout=5, check=False
    )
    if timeout_version.returncode != 0 or "GNU coreutils" not in timeout_version.stdout:
        raise WorkerError("The timeout executable must be GNU coreutils")
    execution = spec.get("execution", {})
    run_id = spec.get("run_id")
    if not isinstance(run_id, str) or not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise WorkerError("Invalid run ID")
    if len(run_spec_sha256) != 64 or any(c not in "0123456789abcdef" for c in run_spec_sha256):
        raise WorkerError("Invalid immutable specification digest")
    if isinstance(fencing_token, bool) or not isinstance(fencing_token, int) or fencing_token <= 0:
        raise WorkerError("A valid consumed dispatch token is required")
    argv = execution.get("argv")
    if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(a, str) or not a or "\x00" in a for a in argv):
        raise WorkerError("Job argv must contain nonempty literal strings")
    cwd = Path(execution.get("working_directory") or "")
    if not cwd.is_absolute() or not cwd.is_dir():
        raise WorkerError("Working directory must resolve to an existing absolute directory")
    if not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file():
        raise WorkerError("Executable must be an existing absolute path")
    max_wall = _positive_number(execution.get("max_wall_seconds"), "max_wall_seconds")
    max_units = execution.get("max_steps_or_clips")
    if isinstance(max_units, bool) or not isinstance(max_units, int) or max_units <= 0:
        raise WorkerError("A positive work-unit bound is required")
    try:
        deadline = datetime.fromisoformat(execution["deadline_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerError("An absolute UTC deadline is required") from exc
    if deadline.tzinfo is None or deadline.utcoffset().total_seconds() != 0:
        raise WorkerError("Deadline must be timezone-aware UTC")
    remaining = (deadline - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise WorkerError("Job deadline already passed")
    max_wall = min(max_wall, remaining)
    poll_seconds = _positive_number(poll_seconds, "poll_seconds")
    heartbeat_seconds = _positive_number(heartbeat_seconds, "heartbeat_seconds")
    shutdown_grace_seconds = _positive_number(shutdown_grace_seconds, "shutdown_grace_seconds")
    if heartbeat_seconds > 60 or poll_seconds > heartbeat_seconds:
        raise WorkerError("Supervisor must observe and heartbeat at least once per minute")
    environment = execution.get("nonsecret_environment", {})
    if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) or "\x00" in k + v or "=" in k for k, v in environment.items()):
        raise WorkerError("Explicit environment must be a string mapping")
    progress_value = execution.get("worker_progress_path")
    progress_path = Path(progress_value) if progress_value else None
    if progress_path is not None and not progress_path.is_absolute():
        raise WorkerError("Worker progress path must be absolute")

    # The token and all immutable fields are checked by the authority immediately
    # before launch. An output directory is exclusive to one attempt, not reused.
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    validate_dispatch()
    supervised_argv = [
        timeout_executable, "--signal=TERM", f"--kill-after={shutdown_grace_seconds}s",
        f"{max_wall}s", *argv,
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "launch.json", {
        "run_id": run_id, "run_spec_sha256": run_spec_sha256,
        "fencing_token": fencing_token, "launch_started_at": utc_now(),
        "argv": list(argv), "supervised_argv": supervised_argv, "working_directory": str(cwd),
        "max_wall_seconds": max_wall, "max_steps_or_clips": max_units,
        "deadline_utc": execution["deadline_utc"],
    })
    started = time.monotonic()
    next_heartbeat = started
    process = None
    identity = None
    stop_reason = None
    progress = None
    unit_limit_observed_at = None
    failure = None
    stop_action = None
    with (output_dir / "output.log").open("wb") as log:
        try:
            process = subprocess.Popen(
                supervised_argv, cwd=cwd, env={**os.environ, **environment},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, shell=False,
            )
            identity = process_identity(process.pid)
            if identity is None:
                # Fast terminal children are still waitable through Popen.
                process.wait(timeout=10)
            else:
                atomic_json(output_dir / "process.json", {
                    **identity, "run_id": run_id, "run_spec_sha256": run_spec_sha256,
                    "fencing_token": fencing_token, "observed_at": utc_now(),
                })
            while process.poll() is None:
                now = time.monotonic()
                progress = _read_progress(progress_path, run_id, run_spec_sha256)
                if now - started >= max_wall or datetime.now(UTC) >= deadline:
                    stop_reason = "deadline"
                elif progress is not None and progress["completed_units"] > max_units:
                    stop_reason = "work_unit_limit"
                elif progress is not None and progress["completed_units"] == max_units:
                    # Applications enforce their own unit cap. Let the final
                    # admitted unit flush artifacts and exit during bounded
                    # shutdown grace; this never extends the wall deadline.
                    if unit_limit_observed_at is None:
                        unit_limit_observed_at = now
                    if now - unit_limit_observed_at >= shutdown_grace_seconds:
                        stop_reason = "work_unit_limit"
                if stop_reason:
                    if identity is None:
                        raise WorkerError("Live child has no verified identity")
                    stop_action = _stop_owned_group(process, identity, shutdown_grace_seconds)
                    break
                if now >= next_heartbeat:
                    atomic_json(output_dir / "heartbeat.json", {
                        "run_id": run_id, "run_spec_sha256": run_spec_sha256,
                        "fencing_token": fencing_token, "pid": process.pid,
                        "process_identity": identity, "observed_at": utc_now(),
                        "supervisor_alive": True, "elapsed_wall_seconds": now - started,
                        "worker_progress": progress,
                    })
                    next_heartbeat = now + heartbeat_seconds
                time.sleep(min(poll_seconds, max(0.001, max_wall - (now - started))))
            exit_code = process.wait(timeout=10)
            if identity is not None and _owned_group_members(identity):
                stop_reason = stop_reason or "orphaned_descendants"
                stop_action = _stop_owned_group(process, identity, shutdown_grace_seconds)
            if stop_reason is None and exit_code in (124, 137):
                stop_reason = "deadline"
        except BaseException as exc:  # noqa: BLE001 - supervisor must clean up owned children even on interruption
            failure = {"type": type(exc).__name__, "message": str(exc)}
            if process is not None and process.poll() is None and identity is not None:
                stop_action = _stop_owned_group(process, identity, shutdown_grace_seconds)
            exit_code = process.returncode if process is not None else None
            stop_reason = stop_reason or "supervision_error"

    artifacts = None
    if exit_code == 0 and not stop_reason and validate_artifacts is not None:
        try:
            artifacts = validate_artifacts()
            if not isinstance(artifacts, dict) or artifacts.get("complete") is not True:
                raise WorkerError("Artifact gate did not confirm complete outputs")
            # Ensure an unserializable/NaN gate result fails before final status.
            json.dumps(artifacts, allow_nan=False)
        except Exception as exc:  # noqa: BLE001 - any artifact-gate failure yields FAILED, never invented completion
            failure = {"type": type(exc).__name__, "message": str(exc)}
            stop_reason = "artifact_validation_failed"
    if stop_reason in {"deadline", "work_unit_limit"}:
        status = "STOPPED"
    elif failure or exit_code != 0 or stop_reason is not None:
        status = "FAILED"
    elif artifacts is None:
        status = "PARTIAL"
    else:
        status = "COMPLETE"
    log_digest = hashlib.sha256()
    with (output_dir / "output.log").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            log_digest.update(chunk)
    result = {
        "run_id": run_id, "run_spec_sha256": run_spec_sha256,
        "fencing_token": fencing_token, "status": status,
        "exit_code": exit_code, "stop_reason": stop_reason, "stop_action": stop_action,
        "finished_at": utc_now(), "elapsed_wall_seconds": time.monotonic() - started,
        "process_identity": identity, "worker_progress": progress,
        "artifacts": artifacts, "error": failure,
        "output_log_sha256": log_digest.hexdigest(),
    }
    atomic_json(output_dir / "completion.json", result)
    return result
