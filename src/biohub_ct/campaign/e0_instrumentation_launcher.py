"""Narrow support-script instrumentation immediately before public shard launch.

This module can be embedded in the notebook without importing the project.
It observes the existing launch path and preserves the subprocess arguments.
It neither chooses a workload nor grants compute authorization.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
from pathlib import Path


class InstrumentationLaunchError(RuntimeError):
    """The exact support-source or launch identity changed."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SupportScriptInterceptor:
    """Patch one reviewed script once and observe its finite public launch set."""

    def __init__(
        self,
        script_path,
        *,
        expected_public_sha256: str,
        patcher,
        helper_source: str,
        runtime_module_name: str,
        output_dir,
    ) -> None:
        self.script_path = Path(script_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_public_sha256):
            raise InstrumentationLaunchError("Expected public support hash is unresolved")
        if not re.fullmatch(r"_[A-Za-z0-9_]+", runtime_module_name):
            raise InstrumentationLaunchError(
                "Runtime helper name is not a private module identifier"
            )
        if not callable(patcher) or not isinstance(helper_source, str) or not helper_source:
            raise InstrumentationLaunchError("A reviewed patcher and helper source are required")
        compile(helper_source, runtime_module_name, "exec")
        self.expected_public_sha256 = expected_public_sha256
        self.patcher = patcher
        self.helper_bytes = helper_source.encode("utf-8")
        self.helper_path = self.script_path.with_name(runtime_module_name + ".py")
        self.runtime_module_name = runtime_module_name
        self._original_popen = None
        self._observed_popen = None
        self._previous_environment = None
        self._environment_installed = False
        self._guard = threading.Lock()
        self._stream = None
        self._output_sha256 = None
        self._closed = False
        self.launch_count = 0
        self.events = []

    def _event(self, event: dict) -> None:
        encoded = (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
        if self._stream is None:
            raise InstrumentationLaunchError("Interceptor evidence stream is closed")
        self._stream.write(encoded)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self.events.append(event)

    def _matches(self, args, kwargs) -> bool:
        if not isinstance(args, (tuple, list)) or len(args) < 2:
            return False
        if not isinstance(args[1], (str, os.PathLike)):
            return False
        cwd = Path(kwargs.get("cwd") or os.getcwd()).resolve()
        candidate = Path(args[1])
        candidate = candidate if candidate.is_absolute() else cwd / candidate
        if candidate.resolve() != self.script_path:
            return False
        if kwargs.get("shell") or Path(args[0]).resolve() != Path(sys.executable).resolve():
            raise InstrumentationLaunchError(
                "Reviewed support entrypoint needs the exact Python argv"
            )
        if cwd != self.script_path.parent.parent:
            raise InstrumentationLaunchError(
                "Reviewed support entrypoint has the wrong working directory"
            )
        return True

    def _prepare(self) -> None:
        if self.script_path.is_symlink() or not self.script_path.is_file():
            raise InstrumentationLaunchError("Materialized support source must be a regular file")
        raw = self.script_path.read_bytes()
        actual = _digest(raw)
        if self._output_sha256 is not None:
            if actual != self._output_sha256 or self.helper_path.is_symlink():
                raise InstrumentationLaunchError(
                    "Instrumented source changed between shard launches"
                )
            if not self.helper_path.is_file() or self.helper_path.read_bytes() != self.helper_bytes:
                raise InstrumentationLaunchError("Runtime helper changed between shard launches")
            return
        if actual != self.expected_public_sha256:
            raise InstrumentationLaunchError(
                "Public dynamic support-source hash differs from the review"
            )
        patched = self.patcher(raw.decode("utf-8"))
        if patched.public_patched_input_sha256 != actual:
            raise InstrumentationLaunchError(
                "Patcher source chain is not bound to the actual input"
            )
        output_bytes = patched.source.encode("utf-8")
        if _digest(output_bytes) != patched.telemetry_output_sha256:
            raise InstrumentationLaunchError("Patcher output hash does not bind its source")
        if (
            patched.runtime_module_name != self.runtime_module_name
            or patched.runtime_module_sha256 != _digest(self.helper_bytes)
        ):
            raise InstrumentationLaunchError(
                "Patcher runtime helper identity differs from the review"
            )
        compile(patched.source, str(self.script_path), "exec")
        with self.helper_path.open("xb") as helper:
            helper.write(self.helper_bytes)
            helper.flush()
            os.fsync(helper.fileno())
        temporary = self.script_path.with_name(
            "." + self.script_path.name + "." + secrets.token_hex(8) + ".telemetry.tmp"
        )
        with temporary.open("xb") as stream:
            stream.write(output_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        # Only the previously checked, exact support file is replaced.
        os.replace(temporary, self.script_path)
        self._output_sha256 = patched.telemetry_output_sha256
        self._event(
            {
                "event": "SUPPORT_PATCHED",
                "source_chain": patched.source_chain(),
                "script_path": str(self.script_path),
                "helper_path": str(self.helper_path),
            }
        )

    def install(self) -> None:
        if self._original_popen is not None or self._closed:
            raise InstrumentationLaunchError("Interceptor can be installed only once")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._stream = (self.output_dir / "launch-events.jsonl").open("xb")
        self._original_popen = subprocess.Popen
        self._previous_environment = os.environ.get("BIOHUB_E0_TELEMETRY_DIR")
        os.environ["BIOHUB_E0_TELEMETRY_DIR"] = str(self.output_dir)
        self._environment_installed = True
        owner = self
        original = self._original_popen

        class ObservedPopen(original):
            def __init__(self, args, *positional, **kwargs):
                matched = owner._matches(args, kwargs)
                if matched:
                    with owner._guard:
                        owner._prepare()
                        environment = kwargs.get("env")
                        if environment is not None:
                            kwargs["env"] = {
                                **environment,
                                "BIOHUB_E0_TELEMETRY_DIR": str(owner.output_dir),
                            }
                        try:
                            super().__init__(args, *positional, **kwargs)
                        except BaseException as exc:
                            owner._event(
                                {"event": "SHARD_LAUNCH_FAILED", "error_type": type(exc).__name__}
                            )
                            raise
                        try:
                            owner._event(
                                {
                                    "event": "SHARD_LAUNCHED",
                                    "pid": self.pid,
                                    "argv": [os.fspath(value) for value in args],
                                    "source_sha256": owner._output_sha256,
                                }
                            )
                            owner.launch_count += 1
                        except BaseException:
                            # Constructor failure must not strand the child whose
                            # verified handle would never reach the public caller.
                            if self.poll() is None:
                                self.terminate()
                            try:
                                self.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                self.kill()
                                self.wait(timeout=5)
                            raise
                else:
                    super().__init__(args, *positional, **kwargs)

        self._observed_popen = ObservedPopen
        subprocess.Popen = ObservedPopen

    def restore_launcher(self, *, expected_launches: int) -> dict:
        """Restore Popen after production; keep output environment for later public cells."""
        if isinstance(expected_launches, bool) or expected_launches not in {1, 2}:
            raise InstrumentationLaunchError("Expected production launch count must be one or two")
        if self._original_popen is None or self._closed:
            raise InstrumentationLaunchError("Interceptor has no active launch observation")
        if subprocess.Popen is not self._observed_popen:
            raise InstrumentationLaunchError("Another writer replaced the subprocess launcher")
        subprocess.Popen = self._original_popen
        result = {
            "event": "LAUNCH_OBSERVATION_FINISHED",
            "launch_count": self.launch_count,
            "expected_launch_count": expected_launches,
            "status": "PASS" if self.launch_count == expected_launches else "ERROR",
        }
        self._event(result)
        if self.launch_count != expected_launches:
            raise InstrumentationLaunchError(
                "Production launch count differs from the public branch"
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        if self._observed_popen is not None and subprocess.Popen is self._observed_popen:
            subprocess.Popen = self._original_popen
        if self._environment_installed and os.environ.get("BIOHUB_E0_TELEMETRY_DIR") == str(
            self.output_dir
        ):
            if self._previous_environment is None:
                os.environ.pop("BIOHUB_E0_TELEMETRY_DIR", None)
            else:
                os.environ["BIOHUB_E0_TELEMETRY_DIR"] = self._previous_environment
        if self._stream is not None:
            self._stream.close()
        self._closed = True
