import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from biohub_ct.campaign.kaggle_outputs import (
    _SDK_HELPER,
    DOWNLOAD_DIRECTORY,
    EXACT_VERSION_PROOF_FILENAME,
    TRANSPORT_RECEIPT_FILENAME,
    ExactOutputSpec,
    KaggleExactOutputClient,
    KaggleOutputError,
)

DIGEST = "d" * 64
REFERENCE = {
    "schema_version": 1,
    "competition": "biohub-cell-tracking-during-development",
    "notebook": {"ref": "owner/source", "version": 3, "enable_internet": False},
}
SHAPES = {"hidden-a": [5, 6, 7, 8]}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def specification(**changes) -> ExactOutputSpec:
    values = {
        "notebook_slug": "owner/exact-output",
        "notebook_version": 7,
        "release_digest": DIGEST,
        "expected_reference": REFERENCE,
        "max_files": 4,
        "max_file_bytes": 100_000,
        "max_total_bytes": 200_000,
        "max_runtime_seconds": 30,
        "page_size": 2,
    }
    values.update(changes)
    return ExactOutputSpec(**values)


def manifest_bytes(
    submission: bytes,
    *,
    release_digest: str = DIGEST,
    reference: object = REFERENCE,
    shapes: object = SHAPES,
) -> bytes:
    value = {
        "schema_version": 1,
        "status": "COMPLETE",
        "reference": reference,
        "release_digest": release_digest,
        "actual_input_shapes_tzyx": shapes,
        "dataset_shapes": shapes,
        "csv_sha256": sha(submission),
        "submission": {
            "path": "/kaggle/working/submission.csv",
            "sha256": sha(submission),
            "rows": 1,
            "datasets": {"hidden-a": 1},
            "validation": "PASS",
        },
    }
    return (json.dumps(value, sort_keys=True) + "\n").encode()


class OutputRunner:
    def __init__(
        self,
        *,
        submission: bytes = b"id,dataset\n1,hidden-a\n",
        manifest_changes: dict | None = None,
        result_changes: dict | None = None,
        extra_files: dict[str, bytes] | None = None,
    ) -> None:
        self.submission = submission
        self.manifest_changes = manifest_changes or {}
        self.result_changes = result_changes or {}
        self.extra_files = extra_files or {}
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        config = json.loads(kwargs["input"])
        output = Path(config["output_dir"])
        manifest = manifest_bytes(self.submission, **self.manifest_changes)
        files = {
            "nested/public_reference_run_manifest.json": manifest,
            "submission.csv": self.submission,
            **self.extra_files,
        }
        records = []
        for relative, content in files.items():
            path = output.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            records.append({"path": relative, "bytes": len(content), "sha256": sha(content)})
        helper = {
            "schema_version": 1,
            "status": "PASS",
            "kaggle_package_version": "2.2.4",
            "notebook_slug": config["notebook_slug"],
            "notebook_version": config["notebook_version"],
            "version_label": config["version_label"],
            "provider_request_fields": {
                "user_name": "owner",
                "kernel_slug": "exact-output",
                "version_label": config["version_label"],
                "page_size": config["page_size"],
            },
            "pages": 1,
            "files": records,
            "total_bytes": sum(row["bytes"] for row in records),
            "log_observations": [],
            "marker_sha256": config["marker_sha256"],
            "helper_deadline_seconds": config["helper_deadline_seconds"],
            "deadline_enforcement": "POSIX_SETITIMER_OS_EXIT",
        }
        helper.update(self.result_changes)
        return subprocess.CompletedProcess(argv, 0, json.dumps(helper), "")


def client(runner) -> KaggleExactOutputClient:
    return KaggleExactOutputClient(["trusted-kaggle-python"], runner=runner)


def test_exact_version_transport_and_proof_are_separate(tmp_path: Path) -> None:
    runner = OutputRunner()
    result = client(runner).retrieve_and_verify_outputs(specification(), tmp_path / "evidence")

    proof = result["proof"]
    assert proof["status"] == "PASS"
    assert proof["transport_status"] == "VERIFIED"
    assert proof["notebook_slug"] == "owner/exact-output"
    assert proof["notebook_version"] == 7
    assert proof["version_label"] == "v7"
    assert proof["release_digest"] == DIGEST
    assert proof["embedded_reference_verified"] is True
    assert proof["embedded_submission_hash_verified"] is True
    assert proof["shape_identity_status"] == "UNAVAILABLE"
    assert proof["expected_input_shapes_tzyx"] is None
    assert proof["release_validation_status"] == "UNRESOLVED_PENDING_INDEPENDENT_VALIDATOR"
    assert {row["basename"] for row in proof["output_files"]} == {
        "public_reference_run_manifest.json",
        "submission.csv",
    }
    assert Path(result["proof_path"]).name == EXACT_VERSION_PROOF_FILENAME
    assert Path(result["transport_receipt_path"]).name == TRANSPORT_RECEIPT_FILENAME
    assert Path(result["download_directory"]).name == DOWNLOAD_DIRECTORY
    assert sha(Path(result["proof_path"]).read_bytes()) == result["proof_sha256"]

    argv, kwargs = runner.calls[0]
    assert argv[:2] == ["trusted-kaggle-python", "-c"]
    assert "request.version_label = version_label" in argv[2]
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 30


def test_independent_shape_contract_is_bound_and_checked(tmp_path: Path) -> None:
    spec = specification(
        expected_input_shapes_tzyx=SHAPES,
        expected_input_shapes_source="reviewed hidden-input inventory receipt sha256:abc",
    )
    proof = client(OutputRunner()).retrieve_and_verify_outputs(spec, tmp_path / "ok")["proof"]
    assert proof["shape_identity_status"] == "VERIFIED"
    assert proof["expected_input_shapes_tzyx"] == SHAPES
    assert proof["expected_input_shapes_sha256"] == sha(
        json.dumps(SHAPES, separators=(",", ":"), sort_keys=True).encode()
    )

    bad = OutputRunner(manifest_changes={"shapes": {"hidden-a": [5, 6, 7, 9]}})
    with pytest.raises(KaggleOutputError, match="failed exact-version proof"):
        client(bad).retrieve_and_verify_outputs(spec, tmp_path / "bad")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"notebook_slug": "latest-only"}, "slug"),
        ({"notebook_version": True}, "version"),
        ({"notebook_version": 0}, "version"),
        ({"release_digest": "unknown"}, "digest"),
        ({"max_files": 0}, "limits"),
        ({"max_file_bytes": 10, "max_total_bytes": 9}, "total byte"),
        ({"max_runtime_seconds": True}, "runtime limit"),
        ({"max_runtime_seconds": "10"}, "runtime limit"),
        ({"run_manifest_filename": ""}, "basename"),
        ({"submission_filename": "stream:alternate"}, "basename"),
        ({"expected_input_shapes_tzyx": SHAPES}, "source label"),
    ],
)
def test_invalid_contract_fails_before_creating_destination(tmp_path, changes, message) -> None:
    destination = tmp_path / "unused"
    with pytest.raises(KaggleOutputError, match=message):
        client(OutputRunner()).retrieve_and_verify_outputs(specification(**changes), destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "manifest_changes",
    [
        {"release_digest": "e" * 64},
        {"reference": {"different": True}},
    ],
)
def test_embedded_release_identity_mismatch_is_not_a_proof(tmp_path, manifest_changes) -> None:
    with pytest.raises(KaggleOutputError, match="failed exact-version proof"):
        client(OutputRunner(manifest_changes=manifest_changes)).retrieve_and_verify_outputs(
            specification(), tmp_path / "evidence"
        )
    receipt = json.loads((tmp_path / "evidence" / TRANSPORT_RECEIPT_FILENAME).read_text())
    assert receipt["status"] == "PASS"
    assert not (tmp_path / "evidence" / EXACT_VERSION_PROOF_FILENAME).exists()


def test_local_rehash_rejects_helper_hash_lie_and_unreported_extra(tmp_path: Path) -> None:
    lie = OutputRunner(result_changes={"total_bytes": 1})
    with pytest.raises(KaggleOutputError, match="failed exact-version proof"):
        client(lie).retrieve_and_verify_outputs(specification(), tmp_path / "lie")

    class ExtraRunner(OutputRunner):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            output = Path(json.loads(kwargs["input"])["output_dir"])
            (output / "unreported.bin").write_bytes(b"untrusted")
            return result

    with pytest.raises(KaggleOutputError, match="failed exact-version proof"):
        client(ExtraRunner()).retrieve_and_verify_outputs(specification(), tmp_path / "extra")


def test_helper_version_substitution_is_rejected(tmp_path: Path) -> None:
    runner = OutputRunner(result_changes={"notebook_version": 8, "version_label": "v8"})
    with pytest.raises(KaggleOutputError, match="failed exact-version proof"):
        client(runner).retrieve_and_verify_outputs(specification(), tmp_path / "evidence")


def test_failed_provider_read_persists_only_sanitized_evidence(tmp_path: Path) -> None:
    signed = "https://storage.example.invalid/object?credential=secret"

    def fail(argv, **kwargs):
        error = {
            "schema_version": 1,
            "status": "ERROR",
            "error_type": "HTTPError",
            "error_sha256": sha(("404 " + signed).encode()),
            "http_status": 404,
        }
        return subprocess.CompletedProcess(argv, 2, json.dumps(error), signed)

    with pytest.raises(KaggleOutputError, match="transport failed"):
        client(fail).retrieve_and_verify_outputs(specification(), tmp_path / "evidence")
    receipt_text = (tmp_path / "evidence" / TRANSPORT_RECEIPT_FILENAME).read_text()
    receipt = json.loads(receipt_text)
    assert receipt["status"] == "ERROR"
    assert receipt["helper_result"]["http_status"] == 404
    assert receipt["signed_urls_persisted"] is False
    assert signed not in receipt_text
    assert "secret" not in receipt_text


def test_timeout_is_not_reported_as_absence_or_success(tmp_path: Path) -> None:
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output="possibly sensitive")

    with pytest.raises(KaggleOutputError, match="transport failed"):
        client(timeout).retrieve_and_verify_outputs(specification(), tmp_path / "evidence")
    receipt = json.loads((tmp_path / "evidence" / TRANSPORT_RECEIPT_FILENAME).read_text())
    assert receipt["status"] == "ERROR"
    assert receipt["process_error_type"] == "TimeoutExpired"
    assert receipt["returncode"] is None


def test_destination_must_be_unused(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(KaggleOutputError, match="new directory"):
        client(OutputRunner()).retrieve_and_verify_outputs(specification(), existing)


def _fake_sdk(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    modules = tmp_path / "fake-sdk"
    (modules / "kaggle").mkdir(parents=True)
    (modules / "kagglesdk/kernels/types").mkdir(parents=True)
    for package in ("kagglesdk", "kagglesdk/kernels", "kagglesdk/kernels/types"):
        (modules / package / "__init__.py").write_text("")
    (modules / "kaggle/__init__.py").write_text(
        """import os, time
class Item:
    file_name = os.environ['FAKE_OUTPUT_NAME']
    url = 'https://objects.example.invalid/signed?credential=secret'
class Response:
    files = [Item()]
    log = 'bounded log'
    next_page_token = None
class KernelsApi:
    def list_kernel_session_output(self, request):
        assert request.version_label == 'v7'
        try:
            time.sleep(float(os.environ.get('FAKE_LIST_SLEEP_SECONDS', '0')))
        except Exception:
            # Emulate a provider retry handler that would swallow TimeoutError.
            time.sleep(5)
        return Response()
class Client:
    class Kernels: pass
    kernels = Kernels()
    kernels.kernels_api_client = KernelsApi()
class Api:
    def build_kaggle_client(self): return Client()
api = Api()
"""
    )
    (modules / "kagglesdk/kernels/types/kernels_api_service.py").write_text(
        "class ApiListKernelSessionOutputRequest: pass\n"
    )
    (modules / "sitecustomize.py").write_text(
        """import io, urllib.request
class Response(io.BytesIO):
    headers = {'Content-Length': '7'}
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def geturl(self): return 'https://objects.example.invalid/final'
urllib.request.urlopen = lambda *args, **kwargs: Response(b'payload')
"""
    )
    metadata = modules / "kaggle-2.2.4.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: kaggle\nVersion: 2.2.4\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(modules)
    return modules, environment


def test_real_helper_process_emits_one_success_object_and_rejects_drive_path(
    tmp_path: Path,
) -> None:
    _, environment = _fake_sdk(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / ".path-binding"
    marker.write_bytes(b"marker")
    config = {
        "notebook_slug": "owner/exact-output",
        "notebook_version": 7,
        "version_label": "v7",
        "output_dir": str(output),
        "marker_sha256": sha(b"marker"),
        "max_files": 2,
        "max_file_bytes": 100,
        "max_total_bytes": 100,
        "max_runtime_seconds": 10,
        "helper_deadline_seconds": 9,
        "page_size": 2,
    }
    environment["FAKE_OUTPUT_NAME"] = "nested/result.txt"
    completed = subprocess.run(
        [sys.executable, "-c", _SDK_HELPER],
        input=json.dumps(config),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    from biohub_ct.campaign.kaggle_outputs import _sanitized_helper_result

    assert _sanitized_helper_result(completed.stdout) == value
    assert value["status"] == "PASS"
    assert value["version_label"] == "v7"
    assert value["files"] == [{"path": "nested/result.txt", "bytes": 7, "sha256": sha(b"payload")}]
    assert (output / "nested/result.txt").read_bytes() == b"payload"

    unsafe_output = tmp_path / "unsafe-output"
    unsafe_output.mkdir()
    unsafe_marker = unsafe_output / ".path-binding"
    unsafe_marker.write_bytes(b"marker")
    config["output_dir"] = str(unsafe_output)
    environment["FAKE_OUTPUT_NAME"] = "C:/escape.txt"
    rejected = subprocess.run(
        [sys.executable, "-c", _SDK_HELPER],
        input=json.dumps(config),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=environment,
    )
    assert rejected.returncode == 2
    error = json.loads(rejected.stdout)
    assert error["status"] == "ERROR"
    assert error["error_type"] == "RuntimeError"
    assert not (unsafe_output / "C:/escape.txt").exists()


def test_real_helper_process_enforces_its_own_wall_deadline(tmp_path: Path) -> None:
    _, environment = _fake_sdk(tmp_path)
    output = tmp_path / "stalled-output"
    output.mkdir()
    (output / ".path-binding").write_bytes(b"marker")
    config = {
        "notebook_slug": "owner/exact-output",
        "notebook_version": 7,
        "version_label": "v7",
        "output_dir": str(output),
        "marker_sha256": sha(b"marker"),
        "max_files": 2,
        "max_file_bytes": 100,
        "max_total_bytes": 100,
        "max_runtime_seconds": 1,
        "helper_deadline_seconds": 0.15,
        "page_size": 2,
    }
    environment["FAKE_OUTPUT_NAME"] = "nested/result.txt"
    environment["FAKE_LIST_SLEEP_SECONDS"] = "5"
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-c", _SDK_HELPER],
        input=json.dumps(config),
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
        env=environment,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 2
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.returncode == 124
    assert not (output / "nested/result.txt").exists()
