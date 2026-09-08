from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import verify_kaggle_reference_outputs as operator

from biohub_ct.campaign.kaggle_rehearsal import E0_R3_PACKAGE_IDENTITY
from biohub_ct.campaign.rehearsal_packages import E0_R4_PACKAGE_IDENTITY
from biohub_ct.config import SUBMISSION_COLUMNS
from e0_package_fixtures import SyntheticE0Package, build_e0_package

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path("synthetic-package-is-installed-by-fixture")


@pytest.fixture(autouse=True)
def synthetic_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, SyntheticE0Package]:
    packages = {
        generation: build_e0_package(tmp_path / "package-fixtures", generation=generation)
        for generation in ("r3", "r4")
    }
    monkeypatch.setitem(globals(), "PACKAGE", packages["r3"].path)
    monkeypatch.setitem(globals(), "E0_R3_PACKAGE_IDENTITY", packages["r3"].identity)
    monkeypatch.setitem(globals(), "E0_R4_PACKAGE_IDENTITY", packages["r4"].identity)
    monkeypatch.setattr(
        operator, "reviewed_package", lambda generation: packages[generation].identity
    )
    return packages


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class ValidationResult:
    def __init__(self, submission: Path, manifest: Path, notebook_version: int = 7) -> None:
        self.submission = submission
        self.manifest = manifest
        self.notebook_version = notebook_version

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "STRUCTURAL_AND_IDENTITY_PASS_ADMISSION_BLOCKED",
            "identity_status": "PASS",
            "release_digest": E0_R3_PACKAGE_IDENTITY.release_digest,
            "notebook_slug": E0_R3_PACKAGE_IDENTITY.notebook_slug,
            "notebook_version": self.notebook_version,
            "submission_sha256": sha256(self.submission),
            "run_manifest_sha256": sha256(self.manifest),
            "package_manifest_sha256": E0_R3_PACKAGE_IDENTITY.package_manifest_sha256,
            "artifact_lock_sha256": E0_R3_PACKAGE_IDENTITY.artifact_lock_sha256,
            "official_format_status": "PASS",
            "scorer_compatibility_status": "PASS",
            "e0_lineage_contract_status": "PASS",
            "quality_admission": "BLOCKED_MISSING_UPSTREAM_EVIDENCE",
            "resource_admission": "BLOCKED_MISSING_UPSTREAM_EVIDENCE",
        }


def local_args(tmp_path: Path) -> tuple[object, dict[str, Path]]:
    files = {
        "proof": tmp_path / "proof.json",
        "submission": tmp_path / "submission.csv",
        "manifest": tmp_path / "manifest.json",
    }
    write_json(files["proof"], {"kind": "fixture-proof", "notebook_version": 7})
    files["submission"].write_text("fixture\n")
    write_json(files["manifest"], {"kind": "fixture-manifest"})
    args = operator.parse_args(
        [
            "--destination",
            str(tmp_path / "result"),
            "--package-dir",
            str(PACKAGE),
            "--proof-json",
            str(files["proof"]),
            "--submission-csv",
            str(files["submission"]),
            "--run-manifest-json",
            str(files["manifest"]),
        ]
    )
    return args, files


def test_local_flow_calls_validator_with_frozen_r3_and_writes_exclusive_result(
    tmp_path: Path,
) -> None:
    args, files = local_args(tmp_path)
    calls = []

    def validate(submission, manifest, package, **kwargs):
        calls.append((submission, manifest, package, kwargs))
        return ValidationResult(submission, manifest)

    output = operator.execute(args, release_validator=validate)
    assert output["result"]["status"] == "VALIDATION_COMPLETE"
    result_path = Path(output["result_path"])
    assert result_path.name == operator.RESULT_FILENAME
    assert sha256(result_path) == output["result_sha256"]
    assert not (result_path.parent / ".validation-result.json.partial").exists()
    persisted = json.loads(result_path.read_text())
    assert persisted == output["result"]
    assert persisted["mode"] == "LOCAL_VALIDATE"
    assert persisted["mutation_performed"] is False
    assert persisted["submission_performed"] is False
    assert persisted["approval_or_campaign_store_written"] is False
    assert persisted["release_validation"]["identity_status"] == "PASS"
    assert calls[0][0:3] == (
        files["submission"].resolve(),
        files["manifest"].resolve(),
        PACKAGE.resolve(),
    )
    assert calls[0][3]["package_identity"] == E0_R3_PACKAGE_IDENTITY
    assert calls[0][3]["exact_version_proof"] == {
        "kind": "fixture-proof",
        "notebook_version": 7,
    }


def test_validator_failure_is_sanitized_and_never_claims_validation(tmp_path: Path) -> None:
    args, _ = local_args(tmp_path)
    secret_url = "https://objects.invalid/file?credential=secret"

    def fail(*args, **kwargs):
        raise RuntimeError(secret_url)

    output = operator.execute(args, release_validator=fail)
    text = Path(output["result_path"]).read_text()
    assert output["result"]["status"] == "ERROR"
    assert output["result"]["failure_stage"] == "INDEPENDENT_RELEASE_VALIDATION"
    assert secret_url not in text
    assert "credential" not in text


@pytest.mark.parametrize("return_selected_identity", [True, False])
def test_r4_selector_requires_r4_identity_in_final_validator_result(
    tmp_path: Path, return_selected_identity: bool
) -> None:
    args, files = local_args(tmp_path)
    args.generation = "r4"
    args.package_dir = tmp_path / "reviewed-r4-fixture"
    args.package_dir.mkdir()
    selected = E0_R4_PACKAGE_IDENTITY

    def preflight(path, identity):
        assert path == args.package_dir.resolve()
        assert identity == selected
        return {"status": "PASS"}

    def validate(submission, manifest, package, **kwargs):
        assert kwargs["package_identity"] == selected
        result = ValidationResult(submission, manifest).to_dict()
        if return_selected_identity:
            result.update(
                release_digest=selected.release_digest,
                notebook_slug=selected.notebook_slug,
                package_manifest_sha256=selected.package_manifest_sha256,
                artifact_lock_sha256=selected.artifact_lock_sha256,
            )
        return result

    output = operator.execute(args, release_validator=validate, package_preflight=preflight)
    result = output["result"]
    assert result["reviewed_package_generation"] == "E0_R4"
    assert result["package_identity"]["notebook_sha256"] == selected.notebook_sha256
    if return_selected_identity:
        assert result["status"] == "VALIDATION_COMPLETE"
        assert result["inputs"]["submission_sha256"] == sha256(files["submission"])
    else:
        assert result["status"] == "ERROR"
        assert result["failure_stage"] == "FINAL_IDENTITY_RECHECK"


def test_unknown_generation_is_rejected_before_operator_execution() -> None:
    with pytest.raises(SystemExit) as error:
        operator.parse_args(["--destination", "unused", "--generation", "arbitrary"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--destination", "out"],
        ["--destination", "out", "--max-files", "2"],
        ["--destination", "out", "--download", "--notebook-version", "1"],
    ],
)
def test_unresolved_mode_options_fail_before_destination_creation(tmp_path: Path, argv) -> None:
    resolved = [str(tmp_path / item) if item == "out" else item for item in argv]
    args = operator.parse_args(resolved)
    with pytest.raises(operator.OperatorInputError):
        operator.execute(args)
    assert not (tmp_path / "out").exists()


def test_existing_or_package_nested_destination_is_rejected(tmp_path: Path) -> None:
    args, _ = local_args(tmp_path)
    args.destination.mkdir()
    with pytest.raises(operator.OperatorInputError, match="new path"):
        operator.execute(args, release_validator=lambda *a, **k: None)


def test_package_nested_destination_is_rejected_without_writing() -> None:
    args = operator.parse_args(
        [
            "--destination",
            str(PACKAGE / "operator-result"),
            "--package-dir",
            str(PACKAGE),
            "--proof-json",
            str(PACKAGE / "package-manifest.json"),
            "--submission-csv",
            str(PACKAGE / "package-manifest.json"),
            "--run-manifest-json",
            str(PACKAGE / "artifact-lock.json"),
        ]
    )
    with pytest.raises(operator.OperatorInputError, match="immutable reviewed package"):
        operator.execute(args, release_validator=lambda *a, **k: None)
    assert not (PACKAGE / "operator-result").exists()


def shape_receipt(path: Path, source_receipt: Path) -> dict[str, object]:
    write_json(
        source_receipt,
        {
            "schema_version": 1,
            "kind": "AUTHENTICATED_HIDDEN_INPUT_METADATA_INVENTORY",
            "status": "VERIFIED",
        },
    )
    value = {
        "schema_version": 1,
        "kind": "BIOHUB_INDEPENDENT_INPUT_SHAPES",
        "status": "VERIFIED",
        "source": "authenticated hidden-input metadata inventory",
        "source_receipt_sha256": sha256(source_receipt),
        "shapes_tzyx": {"hidden-a": [2, 3, 4, 5]},
    }
    write_json(path, value)
    return value


def download_args(tmp_path: Path) -> object:
    shapes = tmp_path / "shapes.json"
    source_receipt = tmp_path / "shape-source.json"
    shape_receipt(shapes, source_receipt)
    return operator.parse_args(
        [
            "--destination",
            str(tmp_path / "result"),
            "--package-dir",
            str(PACKAGE),
            "--download",
            "--notebook-version",
            "9",
            "--input-shapes-json",
            str(shapes),
            "--input-shapes-source-receipt",
            str(source_receipt),
            "--max-files",
            "5",
            "--max-file-bytes",
            "1000",
            "--max-total-bytes",
            "2000",
            "--max-runtime-seconds",
            "30",
            "--page-size",
            "2",
            "--wsl-distro",
            "Ubuntu-24.04",
            "--kaggle-sdk-python",
            "/opt/kaggle/bin/python",
            "--destination-sdk-path",
            "/workspace/operator",
        ]
    )


@pytest.mark.parametrize("canonicalized", [False, True])
def test_explicit_download_binds_mapping_bounds_shapes_and_exact_r3_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonicalized: bool,
) -> None:
    args = download_args(tmp_path)
    expected_slug = E0_R3_PACKAGE_IDENTITY.notebook_slug
    if canonicalized:
        from biohub_ct.campaign import kaggle_canonicalization

        expected_slug += "-reproduction"
        args.canonicalization_receipt = tmp_path / "canonicalization.json"
        write_json(args.canonicalization_receipt, {"test_boundary": True})

        def verified_mapping(path, *, identity):
            assert path == args.canonicalization_receipt.resolve()
            assert identity == E0_R3_PACKAGE_IDENTITY
            return SimpleNamespace(
                actual_notebook_slug=expected_slug, notebook_version=9, receipt_sha256=sha256(path)
            )

        monkeypatch.setattr(
            kaggle_canonicalization, "verify_canonicalization_receipt", verified_mapping
        )
    captured: dict[str, object] = {}
    destination = args.destination.resolve()

    class Transport:
        def __init__(self, prefix, *, path_adapter):
            captured["prefix"] = prefix
            self.path_adapter = path_adapter

        def retrieve_and_verify_outputs(self, spec, transport_dir):
            captured["spec"] = spec
            transport_dir.mkdir()
            downloaded = transport_dir / "downloaded"
            downloaded.mkdir()
            captured["mapped_child"] = self.path_adapter(downloaded)
            submission = downloaded / "submission.csv"
            manifest = downloaded / "public_reference_run_manifest.json"
            submission.write_text("fixture\n")
            write_json(manifest, {"fixture": True})
            proof = {
                "submission_path": str(submission),
                "run_manifest_path": str(manifest),
                "marker": "fixture",
                "notebook_version": 9,
            }
            proof_path = transport_dir / "kaggle-exact-version-output-proof.json"
            write_json(proof_path, proof)
            receipt = transport_dir / "kaggle-output-transport-receipt.json"
            write_json(receipt, {"status": "PASS"})
            return {
                "proof": proof,
                "proof_path": str(proof_path),
                "proof_sha256": sha256(proof_path),
                "transport_receipt_path": str(receipt),
            }

    def verifier_factory(distro):
        assert distro == "Ubuntu-24.04"

        def verify(path):
            relative = path.resolve().relative_to(destination)
            suffix = "" if not relative.parts else "/" + "/".join(relative.parts)
            return "/workspace/operator" + suffix

        return verify

    validated = []

    def validate(submission, manifest, package, **kwargs):
        validated.append((submission, manifest, package, kwargs))
        result = ValidationResult(submission, manifest, notebook_version=9).to_dict()
        result["notebook_slug"] = expected_slug
        if canonicalized:
            assert kwargs["canonicalization_receipt"] == args.canonicalization_receipt.resolve()
        return result

    output = operator.execute(
        args,
        transport_factory=Transport,
        path_verifier_factory=verifier_factory,
        release_validator=validate,
        package_preflight=lambda *a: {"status": "PASS"},
    )
    assert output["result"]["status"] == "VALIDATION_COMPLETE"
    assert captured["prefix"] == [
        "wsl",
        "-d",
        "Ubuntu-24.04",
        "--",
        "/opt/kaggle/bin/python",
    ]
    assert captured["mapped_child"] == "/workspace/operator/transport/downloaded"
    spec = captured["spec"]
    assert spec.notebook_slug == expected_slug
    assert spec.notebook_version == 9
    assert spec.release_digest == E0_R3_PACKAGE_IDENTITY.release_digest
    assert spec.max_files == 5
    assert spec.max_file_bytes == 1000
    assert spec.max_total_bytes == 2000
    assert spec.max_runtime_seconds == 30
    assert spec.page_size == 2
    assert spec.expected_input_shapes_tzyx == {"hidden-a": [2, 3, 4, 5]}
    assert "source_receipt_sha256=" in spec.expected_input_shapes_source
    assert validated[0][3]["package_identity"] == E0_R3_PACKAGE_IDENTITY
    assert output["result"]["transport"]["explicit_bounds"]["max_files"] == 5


def test_unresolved_shape_receipt_fails_closed(tmp_path: Path) -> None:
    args = download_args(tmp_path)
    shapes = json.loads(args.input_shapes_json.read_text())
    shapes["source"] = "TODO"
    write_json(args.input_shapes_json, shapes)
    output = operator.execute(
        args,
        path_verifier_factory=lambda _: lambda path: "/workspace/operator",
        package_preflight=lambda *a: {"status": "PASS"},
    )
    assert output["result"]["status"] == "ERROR"
    assert output["result"]["failure_stage"] == "INDEPENDENT_SHAPE_EVIDENCE"


def test_explicit_wsl_mapping_mismatch_blocks_transport(tmp_path: Path) -> None:
    args = download_args(tmp_path)

    def forbidden_transport(*args, **kwargs):
        raise AssertionError("transport must not be constructed after mapping mismatch")

    output = operator.execute(
        args,
        transport_factory=forbidden_transport,
        path_verifier_factory=lambda _: lambda path: "/different/path",
        package_preflight=lambda *a: {"status": "PASS"},
    )
    assert output["result"]["status"] == "ERROR"
    assert output["result"]["failure_stage"] == "WSL_PATH_BINDING"


def test_download_page_and_total_bounds_are_consistent_before_writes(tmp_path: Path) -> None:
    args = download_args(tmp_path)
    args.page_size = args.max_files + 1
    with pytest.raises(operator.OperatorInputError, match="page-size"):
        operator.execute(args)
    assert not args.destination.exists()

    args = download_args(tmp_path)
    args.max_total_bytes = args.max_file_bytes - 1
    with pytest.raises(operator.OperatorInputError, match="max-total-bytes"):
        operator.execute(args)
    assert not args.destination.exists()


def test_atomic_writer_refuses_temporary_collision_created_during_validation(
    tmp_path: Path,
) -> None:
    args, _ = local_args(tmp_path)

    def collide(*args, **kwargs):
        args_path = tmp_path / "result/.validation-result.json.partial"
        args_path.write_text("do not replace")
        return ValidationResult(Path(args[0]), Path(args[1]))

    with pytest.raises(operator.OperatorInputError, match="temporary path"):
        operator.execute(args, release_validator=collide)
    collision = tmp_path / "result/.validation-result.json.partial"
    assert collision.read_text() == "do not replace"
    assert not (tmp_path / "result/validation-result.json").exists()


def test_tampered_package_manifest_fails_final_identity_recheck(tmp_path: Path) -> None:
    args, files = local_args(tmp_path)

    def validate(submission, manifest, package, **kwargs):
        with (package / "package-manifest.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        return ValidationResult(submission, manifest)

    output = operator.execute(args, release_validator=validate)
    assert output["result"]["status"] == "ERROR"
    assert output["result"]["failure_stage"] == "FINAL_IDENTITY_RECHECK"
    assert output["result"]["error_type"] == "RehearsalError"
    assert files["submission"].read_text() == "fixture\n"


def make_real_r3_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    lock = json.loads((PACKAGE / "artifact-lock.json").read_text())
    submission = tmp_path / "submission.csv"
    submission.write_text(
        ",".join(SUBMISSION_COLUMNS) + "\n0,a,node,1,0,0,0,0,-1,-1\n",
        encoding="utf-8",
        newline="",
    )
    shapes = {"a": [1, 1, 1, 1]}
    integrity = {}
    for reference, record in lock["datasets"].items():
        integrity[reference] = {
            "status": "verified",
            "root": f"/kaggle/input/{reference.split('/', 1)[1]}",
            "version": record["version"],
            "files": len(record["files"]),
            "bytes": sum(item["bytes"] for item in record["files"].values()),
        }
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "reference": lock["reference"],
        "release_digest": lock["release_digest"],
        "input_integrity": integrity,
        "actual_input_shapes_tzyx": shapes,
        "dataset_shapes": shapes,
        "expected_dataset_ids": ["a"],
        "completed_dataset_ids": ["a"],
        "csv_sha256": sha256(submission),
        "coordinate_bounds_valid": True,
        "graph_valid": True,
        "hidden_discovery_supported": True,
        "internet_enabled_configured": False,
        "submission": {
            "path": "/kaggle/working/submission.csv",
            "sha256": sha256(submission),
            "rows": 1,
            "datasets": {"a": {"nodes": 1, "edges": 0}},
            "validation": "PASS",
        },
        "elapsed_seconds": 1.0,
        "full_runtime_seconds": 1.0,
    }
    manifest_path = tmp_path / "public_reference_run_manifest.json"
    write_json(manifest_path, manifest)
    proof = {
        "schema_version": 1,
        "kind": "KAGGLE_EXACT_VERSION_OUTPUT_PROOF",
        "status": "PASS",
        "transport_status": "VERIFIED",
        "notebook_slug": E0_R3_PACKAGE_IDENTITY.notebook_slug,
        "notebook_version": 1,
        "version_label": "v1",
        "release_digest": E0_R3_PACKAGE_IDENTITY.release_digest,
        "run_manifest_sha256": sha256(manifest_path),
        "submission_sha256": sha256(submission),
        "embedded_release_identity_verified": True,
        "embedded_reference_verified": True,
        "embedded_submission_hash_verified": True,
        "output_files": [
            {
                "path": str(manifest_path),
                "basename": manifest_path.name,
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256(manifest_path),
            },
            {
                "path": str(submission),
                "basename": submission.name,
                "bytes": submission.stat().st_size,
                "sha256": sha256(submission),
            },
        ],
        "expected_input_shapes_tzyx": shapes,
        "shape_identity_status": "VERIFIED",
        "release_validation_status": "UNRESOLVED_PENDING_INDEPENDENT_VALIDATOR",
    }
    proof_path = tmp_path / "proof.json"
    write_json(proof_path, proof)
    return proof_path, submission, manifest_path


def test_actual_cli_subprocess_validates_local_r3_fixture(
    tmp_path: Path, synthetic_packages: dict[str, SyntheticE0Package]
) -> None:
    proof, submission, manifest = make_real_r3_evidence(tmp_path)
    destination = tmp_path / "cli-result"
    driver = tmp_path / "synthetic_cli_driver.py"
    driver.write_text(
        """\
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from biohub_ct.campaign.kaggle_rehearsal import PackageIdentity
from scripts import verify_kaggle_reference_outputs as operator

raw = json.loads(sys.argv[2])
raw["dataset_versions"] = tuple(tuple(row) for row in raw["dataset_versions"])
identity = PackageIdentity(**raw)
operator.reviewed_package = lambda generation: identity
raise SystemExit(operator.main(sys.argv[3:]))
""",
        encoding="utf-8",
        newline="\n",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            str(ROOT),
            json.dumps(asdict(synthetic_packages["r3"].identity), sort_keys=True),
            "--destination",
            str(destination),
            "--package-dir",
            str(synthetic_packages["r3"].path),
            "--proof-json",
            str(proof),
            "--submission-csv",
            str(submission),
            "--run-manifest-json",
            str(manifest),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    summary = json.loads(completed.stdout)
    assert summary["status"] == "VALIDATION_COMPLETE"
    result_path = destination / operator.RESULT_FILENAME
    assert summary["result_sha256"] == sha256(result_path)
    result = json.loads(result_path.read_text())
    assert result["release_validation"]["identity_status"] == "PASS"
    assert result["release_validation"]["official_format_status"] == "PASS"
    assert result["release_validation"]["quality_admission"] == (
        "BLOCKED_MISSING_UPSTREAM_EVIDENCE"
    )
    assert result["submission_performed"] is False
