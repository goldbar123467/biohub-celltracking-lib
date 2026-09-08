from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.package_instrumented_public_reference import (
    INCOMPLETE_SENTINEL,
    LAUNCHER_MODULE_NAME,
    PUBLIC_CELL_COUNT,
    SUPPORT_MODULE_NAME,
    _close_interceptor_source,
    _install_interceptor_source,
    _kernel_title,
    build_instrumented_package,
)
from scripts.package_public_reference import (
    build_package,
    canonical_json_bytes,
    sha256_file,
)

REF = "owner/reference-data"


class _FakeEvents:
    def __init__(self) -> None:
        self.callbacks: dict[str, list[object]] = {"post_run_cell": []}

    def register(self, event: str, callback: object) -> None:
        self.callbacks[event].append(callback)

    def unregister(self, event: str, callback: object) -> None:
        self.callbacks[event].remove(callback)

    def emit_swallowing(self, event: str, value: object) -> list[BaseException]:
        errors: list[BaseException] = []
        for callback in list(self.callbacks[event]):
            try:
                callback(value)  # type: ignore[operator]
            except BaseException as error:  # noqa: BLE001 - emulate IPython EventManager
                errors.append(error)
        return errors


class _FakeShell:
    def __init__(self) -> None:
        self.events = _FakeEvents()


_FAKE_SUPPORT_SOURCE = '''
def patch_support_source(source, *, expected_public_patched_sha256):
    return (source, expected_public_patched_sha256)

def runtime_helper_source():
    return "HELPER = True\\n"
'''


_FAKE_LAUNCHER_SOURCE = '''
from pathlib import Path

class SupportScriptInterceptor:
    def __init__(self, script_path, **kwargs):
        self.script_path = script_path
        self.output_dir = Path(kwargs["output_dir"])
        self.closed = False
        self.restore_calls = []

    def install(self):
        self.output_dir.mkdir(parents=True, exist_ok=False)

    def restore_launcher(self, *, expected_launches):
        self.restore_calls.append(expected_launches)
        return {
            "launch_count": expected_launches,
            "expected_launch_count": expected_launches,
            "status": "PASS",
        }

    def close(self):
        self.closed = True
'''


def _glue_namespace(tmp_path: Path) -> tuple[dict[str, object], _FakeShell, str]:
    launch_source = "reviewed public launch cell\n"
    launch_hash = hashlib.sha256(launch_source.encode()).hexdigest()
    config = {
        "schema_version": 1,
        "target_support_script": str(tmp_path / "repo/scripts/predict_unet_transformer.py"),
        "support_output_dir": str(tmp_path / "telemetry"),
        "callback_after_public_cell_sha256": launch_hash,
        "expected_public_patched_support_sha256": "1" * 64,
        "support_runtime_module_name": "_fake_runtime",
        "expected_launches": "2 if worker_count >= 2 and not SLICE else 1",
    }
    shell = _FakeShell()
    namespace: dict[str, object] = {
        "get_ipython": lambda: shell,
        "worker_count": 2,
        "SLICE": "",
    }
    source = _install_interceptor_source(
        config=config,
        support_source=_FAKE_SUPPORT_SOURCE,
        launcher_source=_FAKE_LAUNCHER_SOURCE,
    )
    exec(compile(source, "test-interceptor-install-cell", "exec"), namespace)  # noqa: S102
    return namespace, shell, launch_source


def _cell_result(source: str, error: BaseException | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        info=SimpleNamespace(raw_cell=source),
        error_before_exec=None,
        error_in_exec=error,
    )


def _write_notebook(path: Path) -> list[str]:
    sources = [f"public_value_{index} = {index}\n" for index in range(PUBLIC_CELL_COUNT)]
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": index + 1,
                "metadata": {"public_index": index},
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": ["stale\n"]}
                ],
                "source": [source],
            }
            for index, source in enumerate(sources)
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")
    return sources


def _write_archive(path: Path) -> dict[str, str]:
    payloads = {
        "ARTIFACT_MANIFEST.json": b'{"schema_version":1}',
        "repo/scripts/predict_unet_transformer.py": b"print('public support')\n",
        "weights/model.bin": b"exact-weights",
        "wheels/dependency.whl": b"offline-wheel",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    return {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}


def _inputs(tmp_path: Path) -> tuple[Path, Path, dict, list[str]]:
    notebook = tmp_path / "source.ipynb"
    archive = tmp_path / "reference.zip"
    sources = _write_notebook(notebook)
    members = _write_archive(archive)
    pin = {
        "schema_version": 1,
        "evidence_class": "overlap_unknown",
        "notebook": {
            "ref": "owner/reference-notebook",
            "version": 1,
            "script_version_id": 123,
            "source_sha256": sha256_file(notebook),
            "displayed_public_score": 0.946,
            "displayed_runtime": "1h",
            "machine_shape": "NvidiaTeslaT4",
            "displayed_accelerators": 2,
            "docker_image": "example@sha256:" + "a" * 64,
            "enable_internet": False,
        },
        "datasets": {
            REF: {
                "dataset_id": 456,
                "version": 3,
                "last_updated_utc": "2026-01-01T00:00:00Z",
                "license": "CC0-1.0",
                "archive_sha256": sha256_file(archive),
                "required_members": {
                    "ARTIFACT_MANIFEST.json": members["ARTIFACT_MANIFEST.json"],
                    "repo/scripts/predict_unet_transformer.py": members[
                        "repo/scripts/predict_unet_transformer.py"
                    ],
                    "weights/model.bin": members["weights/model.bin"],
                },
                "minimum_wheel_count": 1,
            }
        },
        "competition": "example-competition",
        "submission_columns": [
            "id",
            "dataset",
            "row_type",
            "node_id",
            "t",
            "z",
            "y",
            "x",
            "source_id",
            "target_id",
        ],
    }
    return notebook, archive, pin, sources


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_two_builds_are_deterministic_and_preserve_exact_public_cells(
    tmp_path: Path,
) -> None:
    notebook, archive, pin, public_sources = _inputs(tmp_path)
    launch_hash = hashlib.sha256(public_sources[4].encode()).hexdigest()
    first = tmp_path / "candidate-a"
    second = tmp_path / "candidate-b"

    first_manifest = build_instrumented_package(
        notebook,
        {REF: archive},
        first,
        "owner/e0-instrumented",
        pin=pin,
        launch_cell_sha256=launch_hash,
    )
    second_manifest = build_instrumented_package(
        notebook,
        {REF: archive},
        second,
        "owner/e0-instrumented",
        pin=pin,
        launch_cell_sha256=launch_hash,
    )

    assert _tree_bytes(first) == _tree_bytes(second)
    assert first_manifest == second_manifest
    assert not (first / INCOMPLETE_SENTINEL).exists()
    kernel_metadata = json.loads(
        (first / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert kernel_metadata["id"] == "owner/e0-instrumented"
    assert kernel_metadata["title"] == "E0 Instrumented"
    assert kernel_metadata["title"].lower().replace(" ", "-") == (
        kernel_metadata["id"].split("/", 1)[1]
    )
    notebook_value = json.loads((first / "submission.ipynb").read_text(encoding="utf-8"))
    assert ["".join(cell["source"]) for cell in notebook_value["cells"][3:15]] == (
        public_sources
    )
    assert [
        cell["metadata"]["biohub_e0_instrumentation"]
        for cell in notebook_value["cells"][:3]
    ] == [
        "e0-telemetry-bootstrap",
        "e0-support-interceptor-install",
        "e0-input-integrity",
    ]
    assert [
        cell["metadata"]["biohub_e0_instrumentation"]
        for cell in notebook_value["cells"][-3:]
    ] == [
        "e0-release-validation",
        "e0-support-interceptor-close",
        "e0-telemetry-finalizer",
    ]
    for cell in notebook_value["cells"]:
        compile("".join(cell["source"]), "candidate-test-cell", "exec")

    lock = json.loads((first / "artifact-lock.json").read_text(encoding="utf-8"))
    release_digest = lock.pop("release_digest")
    assert release_digest == hashlib.sha256(canonical_json_bytes(lock)).hexdigest()
    assert first_manifest["base_release_digest"] != release_digest
    instrumentation = lock["instrumentation"]
    assert instrumentation["config"]["callback_after_public_cell_sha256"] == launch_hash
    assert instrumentation["config_sha256"] == hashlib.sha256(
        canonical_json_bytes(instrumentation["config"])
    ).hexdigest()
    assert set(instrumentation["source_sha256"]) == {
        "e0_support_telemetry.py",
        "e0_support_runtime_helper.py",
        "e0_instrumentation_launcher.py",
        "e0_notebook_telemetry.py",
        "package_instrumented_public_reference.py",
    }
    assert instrumentation["config"]["notebook_telemetry"] == {
        "coordinate_glob": "detector_coordinates_*.jsonl",
        "max_samples": 20_000,
        "nvidia_smi_timeout_seconds": 2.0,
        "output_dir": "/kaggle/working",
        "require_coordinate_artifacts": True,
        "require_submission_crosscheck": True,
        "retention_glob": "retention_guard_*.jsonl",
        "sample_interval_seconds": 5.0,
        "support_telemetry_dirname": "e0_support_telemetry",
    }
    assert len(first_manifest["instrumentation"]["allowed_auxiliary_cell_sha256"]) == 4
    assert first_manifest["algorithm_changes"] == []


def test_running_instrumented_builder_does_not_change_base_builder_outputs(
    tmp_path: Path,
) -> None:
    notebook, archive, pin, public_sources = _inputs(tmp_path)
    launch_hash = hashlib.sha256(public_sources[4].encode()).hexdigest()
    base_before = tmp_path / "base-before"
    base_after = tmp_path / "base-after"
    build_package(notebook, {REF: archive}, base_before, "owner/e0-reference", pin=pin)

    build_instrumented_package(
        notebook,
        {REF: archive},
        tmp_path / "candidate",
        "owner/e0-instrumented",
        pin=pin,
        launch_cell_sha256=launch_hash,
    )

    build_package(notebook, {REF: archive}, base_after, "owner/e0-reference", pin=pin)
    assert _tree_bytes(base_before) == _tree_bytes(base_after)


def test_failed_build_retains_incomplete_sentinel(tmp_path: Path) -> None:
    notebook, archive, pin, _ = _inputs(tmp_path)
    output = tmp_path / "failed-candidate"
    with pytest.raises(RuntimeError, match="launch-cell SHA-256"):
        build_instrumented_package(
            notebook,
            {REF: archive},
            output,
            "owner/e0-instrumented",
            pin=pin,
            launch_cell_sha256="0" * 64,
        )
    assert (output / INCOMPLETE_SENTINEL).is_file()


def test_output_directory_is_exclusive(tmp_path: Path) -> None:
    notebook, archive, pin, public_sources = _inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_instrumented_package(
            notebook,
            {REF: archive},
            output,
            "owner/e0-instrumented",
            pin=pin,
            launch_cell_sha256=hashlib.sha256(public_sources[4].encode()).hexdigest(),
        )
    assert marker.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize(
    ("kernel_id", "title", "message"),
    [
        ("owner/e0_instrumented", None, "hyphenated slug"),
        ("owner/e0-instrumented", "Different Candidate", "canonicalizes"),
        ("owner/e0-instrumented", "E0 Instrumented!", "ASCII letters/digits"),
        ("owner/e0-instrumented", "E0 Instruménted", "ASCII letters/digits"),
    ],
)
def test_kernel_title_rejects_noncanonical_aliases(
    kernel_id: str, title: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _kernel_title(kernel_id, title)


def test_valid_optional_title_canonicalizes_to_exact_slug() -> None:
    assert (
        _kernel_title(
            "clarkkitchen/biohub-e0-instrumented-reference",
            "Biohub E0 Instrumented Reference",
        )
        == "Biohub E0 Instrumented Reference"
    )


def test_direct_cli_help_resolves_repository_imports() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/package_instrumented_public_reference.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--source-notebook" in completed.stdout


def test_generated_callback_restores_after_exact_cell_and_close_cleans_up(
    tmp_path: Path,
) -> None:
    try:
        namespace, shell, launch_source = _glue_namespace(tmp_path)
        assert shell.events.emit_swallowing(
            "post_run_cell", _cell_result("different cell\n")
        ) == []
        assert namespace["_E0_LAUNCH_CALLBACK_STATE"]["status"] == "PENDING"  # type: ignore[index]

        assert shell.events.emit_swallowing(
            "post_run_cell", _cell_result(launch_source)
        ) == []
        state = namespace["_E0_LAUNCH_CALLBACK_STATE"]
        assert state["status"] == "PASS"  # type: ignore[index]
        assert state["launch_count"] == 2  # type: ignore[index]
        interceptor = namespace["_E0_SUPPORT_INTERCEPTOR"]
        assert interceptor.restore_calls == [2]  # type: ignore[attr-defined]

        exec(  # noqa: S102 - generated close cell is the behavior under test.
            compile(_close_interceptor_source(), "test-interceptor-close-cell", "exec"),
            namespace,
        )
        assert interceptor.closed is True  # type: ignore[attr-defined]
        assert shell.events.callbacks["post_run_cell"] == []
        persisted = json.loads(
            (tmp_path / "telemetry/launcher-callback-status.json").read_text()
        )
        assert persisted["status"] == "PASS"
        assert persisted["cleanup_status"] == "PASS"
    finally:
        sys.modules.pop(SUPPORT_MODULE_NAME, None)
        sys.modules.pop(LAUNCHER_MODULE_NAME, None)


@pytest.mark.parametrize("mode", ["missing", "public_error", "repeated"])
def test_generated_close_rejects_swallowed_callback_failures_and_still_cleans_up(
    tmp_path: Path,
    mode: str,
) -> None:
    try:
        namespace, shell, launch_source = _glue_namespace(tmp_path)
        callback_errors: list[BaseException] = []
        if mode == "public_error":
            callback_errors = shell.events.emit_swallowing(
                "post_run_cell", _cell_result(launch_source, RuntimeError("public failed"))
            )
        elif mode == "repeated":
            assert shell.events.emit_swallowing(
                "post_run_cell", _cell_result(launch_source)
            ) == []
            callback_errors = shell.events.emit_swallowing(
                "post_run_cell", _cell_result(launch_source)
            )
        if mode != "missing":
            assert len(callback_errors) == 1

        interceptor = namespace["_E0_SUPPORT_INTERCEPTOR"]
        with pytest.raises(RuntimeError, match="launch callback did not pass"):
            exec(  # noqa: S102 - generated close cell is the behavior under test.
                compile(
                    _close_interceptor_source(),
                    "test-failing-interceptor-close-cell",
                    "exec",
                ),
                namespace,
            )
        assert interceptor.closed is True  # type: ignore[attr-defined]
        assert shell.events.callbacks["post_run_cell"] == []
        persisted = json.loads(
            (tmp_path / "telemetry/launcher-callback-status.json").read_text()
        )
        assert persisted["status"] in {"PENDING", "ERROR"}
        assert persisted["cleanup_status"] == "ERROR"
    finally:
        sys.modules.pop(SUPPORT_MODULE_NAME, None)
        sys.modules.pop(LAUNCHER_MODULE_NAME, None)
