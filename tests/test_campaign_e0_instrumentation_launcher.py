from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from biohub_ct.campaign.e0_instrumentation_launcher import (
    InstrumentationLaunchError,
    SupportScriptInterceptor,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(tmp_path: Path):
    repository = tmp_path / "repository"
    script = repository / "scripts/predict_unet_transformer.py"
    script.parent.mkdir(parents=True)
    source = (
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps({'value': 7, 'args': sys.argv[2:], "
        "'pythonpath': os.environ.get('PYTHONPATH'), "
        "'telemetry_dir': os.environ.get('BIOHUB_E0_TELEMETRY_DIR')}))\n"
    )
    script.write_bytes(source.encode())
    helper = "MARKER = 'fixture instrumentation'\n"
    patched = "import _test_e0_telemetry_runtime\n" + source
    calls = []

    def patcher(actual):
        calls.append(actual)
        return SimpleNamespace(
            source=patched,
            public_patched_input_sha256=digest(source),
            telemetry_output_sha256=digest(patched),
            runtime_module_name="_test_e0_telemetry_runtime",
            runtime_module_sha256=digest(helper),
            source_chain=lambda: {"input": digest(source), "output": digest(patched)},
        )

    interceptor = SupportScriptInterceptor(
        script,
        expected_public_sha256=digest(source),
        patcher=patcher,
        helper_source=helper,
        runtime_module_name="_test_e0_telemetry_runtime",
        output_dir=tmp_path / "telemetry",
    )
    return repository, script, interceptor, calls


def run_script(repository, output_name):
    return subprocess.run(
        [sys.executable, "scripts/predict_unet_transformer.py", output_name, "--unchanged", "7"],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )


def test_exact_launch_interception_preserves_behavior_and_restores(tmp_path, monkeypatch):
    repository, script, interceptor, calls = fixture(tmp_path)
    monkeypatch.setenv("BIOHUB_E0_TELEMETRY_DIR", "prior-value")
    original = subprocess.Popen
    interceptor.install()
    try:
        unrelated = subprocess.run(
            [sys.executable, "-c", "print('unrelated')"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        assert unrelated.stdout.strip() == "unrelated"
        assert interceptor.launch_count == 0
        for filename in ("first.json", "second.json"):
            run_script(repository, filename)
            result = json.loads((repository / filename).read_text())
            assert result == {
                "value": 7,
                "args": ["--unchanged", "7"],
                "pythonpath": "src",
                "telemetry_dir": str(interceptor.output_dir),
            }
        assert len(calls) == 1
        assert interceptor.restore_launcher(expected_launches=2)["status"] == "PASS"
        assert subprocess.Popen is original
        # Later public validation can still import the materialized helper and
        # retains the supplemental output directory without another patch.
        run_script(repository, "later-validation.json")
        assert interceptor.launch_count == 2
        assert script.read_text().startswith("import _test_e0_telemetry_runtime")
    finally:
        interceptor.close()
    assert os.environ["BIOHUB_E0_TELEMETRY_DIR"] == "prior-value"
    events = [
        json.loads(line)
        for line in (interceptor.output_dir / "launch-events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == [
        "SUPPORT_PATCHED",
        "SHARD_LAUNCHED",
        "SHARD_LAUNCHED",
        "LAUNCH_OBSERVATION_FINISHED",
    ]


def test_wrong_public_source_never_launches(tmp_path):
    repository, script, interceptor, calls = fixture(tmp_path)
    script.write_text("raise RuntimeError('changed')")
    interceptor.install()
    try:
        with pytest.raises(InstrumentationLaunchError, match="hash differs"):
            run_script(repository, "not-created.json")
        assert interceptor.launch_count == 0
        assert not calls
        assert not (repository / "not-created.json").exists()
    finally:
        interceptor.close()


def test_helper_changes_between_shards_fail_closed(tmp_path):
    repository, _, interceptor, _ = fixture(tmp_path)
    interceptor.install()
    try:
        run_script(repository, "first.json")
        interceptor.helper_path.write_text("raise RuntimeError('changed helper')")
        with pytest.raises(InstrumentationLaunchError, match="helper changed"):
            run_script(repository, "not-created.json")
        assert interceptor.launch_count == 1
        assert not (repository / "not-created.json").exists()
    finally:
        interceptor.close()


def test_missing_expected_launch_restores_original_launcher(tmp_path):
    _, _, interceptor, _ = fixture(tmp_path)
    original = subprocess.Popen
    interceptor.install()
    try:
        with pytest.raises(InstrumentationLaunchError, match="launch count"):
            interceptor.restore_launcher(expected_launches=1)
        assert subprocess.Popen is original
    finally:
        interceptor.close()


def test_failed_launch_evidence_reaps_the_child_handle(tmp_path, monkeypatch):
    created = []
    original = subprocess.Popen

    class CapturingPopen(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(subprocess, "Popen", CapturingPopen)
    repository, _, interceptor, _ = fixture(tmp_path)
    original_event = interceptor._event

    def fail_on_launch(event):
        if event["event"] == "SHARD_LAUNCHED":
            raise OSError("fixture evidence storage failure")
        original_event(event)

    monkeypatch.setattr(interceptor, "_event", fail_on_launch)
    interceptor.install()
    try:
        with pytest.raises(OSError, match="storage failure"):
            run_script(repository, "maybe-created.json")
        assert len(created) == 1
        assert created[0].poll() is not None
        assert interceptor.launch_count == 0
    finally:
        interceptor.close()
