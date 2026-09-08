"""Fail-closed launch and recovery for an immutable Kaggle notebook rehearsal.

This module treats a kernel push as an external mutation with at-most-once local
dispatch.  A missing or malformed response is never evidence that no version was
created.  Ambiguous launches remain UNKNOWN with their quota reservation active.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlparse

from biohub_ct.campaign.admission import (
    AdmissionError,
    admit_resource_action,
    decimal_nonnegative,
    release_quota_reserve,
)
from biohub_ct.campaign.contracts import IntentState, JobState, canonical_json
from biohub_ct.campaign.kaggle_cli import KaggleCLI
from biohub_ct.campaign.state import CampaignStore, ReviewLease


class RehearsalError(RuntimeError):
    """The rehearsal cannot safely advance."""


EXPECTED_PACKAGE_FILES = frozenset(
    {"artifact-lock.json", "kernel-metadata.json", "package-manifest.json", "submission.ipynb"}
)
EXACT_VERSION_RECEIPT_FILE = "kaggle-exact-version-read-receipt.json"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SLUG = re.compile(r"[a-z0-9_-]+/[a-z0-9_-]+\Z")
_SUCCESS = re.compile(
    r"^Kernel version ([1-9][0-9]*) successfully pushed\.\s+Please check progress at (\S+)\s*$",
    re.MULTILINE,
)
_REMOTE_IDENTITY_WARNING = re.compile(
    r"(?:not (?:a )?valid (?:dataset|competition|kernel|model) sources?|Kernel push error)",
    re.IGNORECASE,
)
_SDK_SOURCE_MAX_BYTES = 64 * 1024 * 1024
_SDK_METADATA_MAX_BYTES = 1024 * 1024

# Runs inside the explicitly configured Python environment that owns Kaggle
# SDK 2.2.4.  The request carries the version in ``version_label``; unlike the
# broken CLI ``kernels pull owner/slug/N`` path it never appends a version to
# ``kernel_slug``.
_SDK_EXACT_SOURCE_HELPER = r"""
import hashlib, importlib.metadata, json, os, pathlib, signal, sys

def emit(value, code=0):
    sys.stdout.write(json.dumps(value, sort_keys=True, allow_nan=False))
    raise SystemExit(code)

try:
    cfg = json.load(sys.stdin)
    deadline = cfg["helper_deadline_seconds"]
    def expire(_signum, _frame):
        os._exit(124)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, deadline)

    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    root = pathlib.Path(cfg["output_dir"])
    if not root.is_dir() or root.is_symlink() or list(root.iterdir()):
        raise RuntimeError("recovery directory is not empty and exclusive")
    fields = cfg["request"]
    request = ApiGetKernelRequest()
    request.user_name = fields["user_name"]
    request.kernel_slug = fields["kernel_slug"]
    request.version_label = fields["version_label"]
    response = api.build_kaggle_client().kernels.kernels_api_client.get_kernel(request)
    if response is None or response.blob is None or response.metadata is None:
        raise RuntimeError("SDK response omitted source or metadata")
    source = response.blob.source
    if not isinstance(source, str) or not source:
        raise RuntimeError("SDK response source is empty")
    source_bytes = source.encode("utf-8")
    if len(source_bytes) > cfg["max_source_bytes"]:
        raise RuntimeError("SDK source exceeds fixed byte limit")
    metadata = response.metadata
    projected = {
        "id": metadata.id,
        "ref": metadata.ref,
        "title": metadata.title,
        "slug": metadata.slug,
        "language": metadata.language,
        "kernel_type": metadata.kernel_type,
        "is_private": metadata.is_private,
        "enable_gpu": metadata.enable_gpu,
        "enable_internet": metadata.enable_internet,
        "dataset_data_sources": metadata.dataset_data_sources,
        "kernel_data_sources": metadata.kernel_data_sources,
        "competition_data_sources": metadata.competition_data_sources,
        "model_data_sources": metadata.model_data_sources,
        "enable_tpu": metadata.enable_tpu,
        "current_version_number": metadata.current_version_number,
        "docker_image": metadata.docker_image,
        "machine_shape": metadata.machine_shape,
    }
    metadata_bytes = json.dumps(projected, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(metadata_bytes) > cfg["max_metadata_bytes"]:
        raise RuntimeError("SDK metadata exceeds fixed byte limit")
    source_path = root / "source.ipynb"
    with source_path.open("xb") as stream:
        stream.write(source_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    emit({
        "schema_version": 1,
        "status": "PASS",
        "kaggle_package_version": importlib.metadata.version("kaggle"),
        "request": fields,
        "metadata": projected,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_bytes": len(source_bytes),
        "helper_deadline_seconds": deadline,
        "deadline_enforcement": "POSIX_SETITIMER_OS_EXIT",
    })
except Exception as exc:
    emit({
        "schema_version": 1,
        "status": "ERROR",
        "error_type": type(exc).__name__,
        "error_sha256": hashlib.sha256(repr(exc).encode()).hexdigest(),
    }, 2)
"""


@dataclass(frozen=True)
class PackageIdentity:
    notebook_slug: str
    notebook_title: str
    competition: str
    machine_shape: str
    release_digest: str
    package_manifest_sha256: str
    kernel_metadata_sha256: str
    artifact_lock_sha256: str
    notebook_sha256: str
    cli_normalized_notebook_sha256: str
    dataset_versions: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class QuotaContract:
    ledger_id: str
    measured_debit_rate_per_wall_hour: object
    notebook_timeout_seconds: int
    final_attempt_runtime_bound_hours: object
    verified_platform_runtime_limit_hours: object
    documented_debit_rate_upper_bound_per_wall_hour: object | None = None
    quota_rate_source: str | None = None


@dataclass(frozen=True)
class PushAttempt:
    exact_receipt: bool
    notebook_slug: str
    notebook_version: int | None
    result_url: str | None
    returncode: int | None
    stdout_sha256: str
    stderr_sha256: str
    finished_at: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExactVersionPullReceipt:
    """Locally persisted evidence produced by one exact-version SDK source read."""

    artifact_path: str
    artifact_sha256: str
    exact_kernel_ref: str
    notebook_version: int
    source_path: str
    source_sha256: str
    cli_normalized_source_sha256: str
    observed_at: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


E0_R3_PACKAGE_IDENTITY = PackageIdentity(
    notebook_slug="clarkkitchen/biohub-e0-public-reference",
    notebook_title="Biohub E0 Public Reference Reproduction",
    competition="biohub-cell-tracking-during-development",
    machine_shape="NvidiaTeslaT4",
    release_digest="e02e7bd80f17ec89fb6217ea86b1591c6ff949befcb655c66988ecb179e23e78",
    package_manifest_sha256="26065311f67657122f6436111b5fc1d15f3451a1bc20e1da1d6c27a3808c20c4",
    kernel_metadata_sha256="85ef798c0f980027b334f0613818e0ea62e2eb65c848ab27abfe76142249b812",
    artifact_lock_sha256="6f12505158a1c42a5d15ef336e86a4c4b9e99d9087cfed407da84d1e9a1e7241",
    notebook_sha256="b0c51948112fd407a848de44d52923f2fe70f432e5520060338f9f518afe5e30",
    cli_normalized_notebook_sha256=(
        "c4d12a7d6b1ad3de5b4ef0883329d9fdd8c4cad7c7ee005e17c89e1354ba72d4"
    ),
    dataset_versions=(
        ("pilkwang/biohub-deepcenter-unet3d-center-prior-v1", 5),
        ("pilkwang/biohub-temporal-unet3d-seed314159-v1", 2),
        ("pilkwang/biohub-tracking-support-pack-50ep-v1", 10),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(path: Path) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RehearsalError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    def bad_constant(value: str) -> None:
        raise RehearsalError(f"nonfinite JSON value in {path.name}: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=bad_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RehearsalError(f"cannot read strict JSON from {path.name}") from exc
    if not isinstance(value, dict):
        raise RehearsalError(f"{path.name} must contain a JSON object")
    return value


def _cli_normalized_notebook_source(path: Path) -> str:
    """Reproduce Kaggle CLI 2.2.4 notebook normalization exactly.

    The installed client parses JSON, clears code-cell outputs, joins source
    lists, then calls ``json.dumps`` with its defaults before upload.
    """

    notebook = _strict_object(path)
    cells = notebook.get("cells")
    if cells is not None:
        if not isinstance(cells, list):
            raise RehearsalError("notebook cells must be a list")
        for cell in cells:
            if not isinstance(cell, dict):
                raise RehearsalError("notebook cells must be JSON objects")
            if "outputs" in cell and cell.get("cell_type") == "code":
                cell["outputs"] = []
            source = cell.get("source")
            if isinstance(source, list):
                if not all(isinstance(part, str) for part in source):
                    raise RehearsalError("notebook source lists must contain only strings")
                cell["source"] = "".join(source)
    return json.dumps(notebook)


def _versioned_dataset_sources(identity: PackageIdentity) -> list[str]:
    return [f"{ref}/{version}" for ref, version in identity.dataset_versions]


def _validate_identity(identity: PackageIdentity) -> None:
    if not _SLUG.fullmatch(identity.notebook_slug):
        raise RehearsalError("package identity requires an exact owner/notebook slug")
    if not identity.notebook_title or not identity.competition or not identity.machine_shape:
        raise RehearsalError("title, competition, and machine shape must be resolved")
    for field in (
        identity.release_digest,
        identity.package_manifest_sha256,
        identity.kernel_metadata_sha256,
        identity.artifact_lock_sha256,
        identity.notebook_sha256,
        identity.cli_normalized_notebook_sha256,
    ):
        if not _SHA256.fullmatch(field):
            raise RehearsalError("package identity contains an invalid SHA-256")
    if not identity.dataset_versions or len(dict(identity.dataset_versions)) != len(
        identity.dataset_versions
    ):
        raise RehearsalError("dataset versions must be nonempty and unique")
    for ref, version in identity.dataset_versions:
        if not _SLUG.fullmatch(ref) or isinstance(version, bool) or version <= 0:
            raise RehearsalError("dataset identity requires exact positive versions")


def preflight_package(package_dir: Path, identity: PackageIdentity) -> dict[str, object]:
    """Verify the exact local bytes and private, offline GPU metadata before dispatch."""

    _validate_identity(identity)
    try:
        root = package_dir.resolve(strict=True)
    except OSError as exc:
        raise RehearsalError("package directory is unavailable") from exc
    if not root.is_dir():
        raise RehearsalError("package path must be a directory")
    entries = list(root.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise RehearsalError("package cannot contain symlinks or directories")
    names = {entry.name for entry in entries}
    if names != EXPECTED_PACKAGE_FILES:
        raise RehearsalError("package file set does not match the reviewed release")

    hashes = {name: _sha256(root / name) for name in sorted(names)}
    expected_hashes = {
        "artifact-lock.json": identity.artifact_lock_sha256,
        "kernel-metadata.json": identity.kernel_metadata_sha256,
        "package-manifest.json": identity.package_manifest_sha256,
        "submission.ipynb": identity.notebook_sha256,
    }
    if hashes != expected_hashes:
        raise RehearsalError("package bytes do not match the immutable identity")
    normalized_sha256 = hashlib.sha256(
        _cli_normalized_notebook_source(root / "submission.ipynb").encode("utf-8")
    ).hexdigest()
    if normalized_sha256 != identity.cli_normalized_notebook_sha256:
        raise RehearsalError("CLI-normalized notebook differs from the immutable identity")

    metadata = _strict_object(root / "kernel-metadata.json")
    manifest = _strict_object(root / "package-manifest.json")
    artifact_lock = _strict_object(root / "artifact-lock.json")
    required_metadata = {
        "id": identity.notebook_slug,
        "title": identity.notebook_title,
        "code_file": "submission.ipynb",
        "is_private": True,
        "enable_internet": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": identity.machine_shape,
        "kernel_type": "notebook",
        "language": "python",
        "competition_sources": [identity.competition],
        "dataset_sources": _versioned_dataset_sources(identity),
        "kernel_sources": [],
        "model_sources": [],
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise RehearsalError(f"kernel metadata violates exact {key} contract")

    packaged_notebook = manifest.get("packaged_notebook")
    if not isinstance(packaged_notebook, Mapping):
        raise RehearsalError("package manifest has no packaged notebook identity")
    manifest_checks = {
        "schema_version": 1,
        "status": "ready_for_root_review_not_launched",
        "release_digest": identity.release_digest,
        "kernel_metadata_sha256": identity.kernel_metadata_sha256,
        "artifact_lock_sha256": identity.artifact_lock_sha256,
    }
    for key, expected in manifest_checks.items():
        if manifest.get(key) != expected:
            raise RehearsalError(f"package manifest violates exact {key} contract")
    if (
        packaged_notebook.get("path") != "submission.ipynb"
        or packaged_notebook.get("sha256") != identity.notebook_sha256
    ):
        raise RehearsalError("package manifest does not bind the notebook bytes")
    if manifest.get("algorithm_changes") != []:
        raise RehearsalError("reproduction package declares an algorithm change")

    if artifact_lock.get("schema_version") != 1:
        raise RehearsalError("artifact lock schema is unsupported")
    if artifact_lock.get("release_digest") != identity.release_digest:
        raise RehearsalError("artifact lock release digest differs from package")
    if artifact_lock.get("competition") != identity.competition:
        raise RehearsalError("artifact lock competition differs from package")
    datasets = artifact_lock.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {
        ref for ref, _ in identity.dataset_versions
    }:
        raise RehearsalError("artifact lock dataset set differs from package metadata")
    for ref, version in identity.dataset_versions:
        entry = datasets[ref]
        if not isinstance(entry, Mapping) or entry.get("version") != version:
            raise RehearsalError(f"artifact lock does not bind dataset version: {ref}")

    return {
        "status": "PASS",
        "package_root": str(root),
        "notebook_slug": identity.notebook_slug,
        "release_digest": identity.release_digest,
        "file_sha256": hashes,
        "cli_normalized_notebook_sha256": normalized_sha256,
        "private": True,
        "internet": False,
        "gpu": True,
        "machine_shape": identity.machine_shape,
        "dataset_versions": dict(identity.dataset_versions),
    }


class KaggleRehearsalCLI:
    """Installed Kaggle CLI 2.2.4 command adapter with one mutation method."""

    def __init__(
        self,
        argv_prefix: Sequence[str],
        *,
        sdk_python_prefix: Sequence[str] | None = None,
        runner: Callable[..., object] = subprocess.run,
        read_timeout_seconds: float = 45,
        mutation_timeout_seconds: float = 120,
    ) -> None:
        if not argv_prefix or any(
            not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv_prefix
        ):
            raise RehearsalError("a literal installed Kaggle CLI prefix is required")
        for value in (read_timeout_seconds, mutation_timeout_seconds):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RehearsalError("CLI process timeouts must be finite and positive")
            if not math.isfinite(value) or value <= 0:
                raise RehearsalError("CLI process timeouts must be finite and positive")
        self.prefix = list(argv_prefix)
        if sdk_python_prefix is not None and (
            len(sdk_python_prefix) != 5
            or any(
                not isinstance(arg, str) or not arg or "\x00" in arg for arg in sdk_python_prefix
            )
            or sdk_python_prefix[0] != "wsl"
            or sdk_python_prefix[1] != "-d"
            or sdk_python_prefix[3] != "--"
            or not sdk_python_prefix[4].startswith("/")
        ):
            raise RehearsalError("a literal WSL Kaggle SDK Python prefix is required")
        self.sdk_python_prefix = list(sdk_python_prefix) if sdk_python_prefix is not None else None
        self.runner = runner
        self.read_timeout_seconds = float(read_timeout_seconds)
        self.mutation_timeout_seconds = float(mutation_timeout_seconds)

    def push_argv(
        self, *, package_cli_path: str, accelerator: str, notebook_timeout_seconds: int
    ) -> list[str]:
        if not package_cli_path or "\x00" in package_cli_path:
            raise RehearsalError("CLI package path must be resolved")
        if not accelerator or "\x00" in accelerator:
            raise RehearsalError("accelerator must be resolved")
        if (
            isinstance(notebook_timeout_seconds, bool)
            or not isinstance(notebook_timeout_seconds, int)
            or notebook_timeout_seconds <= 0
        ):
            raise RehearsalError("notebook timeout must be a positive integer")
        return [
            *self.prefix,
            "kernels",
            "push",
            "--path",
            package_cli_path,
            "--timeout",
            str(notebook_timeout_seconds),
            "--accelerator",
            accelerator,
        ]

    @staticmethod
    def _safe_hash(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def push_once(
        self,
        *,
        package_cli_path: str,
        accelerator: str,
        notebook_timeout_seconds: int,
        expected_slug: str,
        verify_persisted_intent: Callable[[], None],
    ) -> PushAttempt:
        """Perform one push call after the caller proves a dispatched intent exists."""

        if not _SLUG.fullmatch(expected_slug):
            raise RehearsalError("expected notebook slug is invalid")
        argv = self.push_argv(
            package_cli_path=package_cli_path,
            accelerator=accelerator,
            notebook_timeout_seconds=notebook_timeout_seconds,
        )
        try:
            verify_persisted_intent()
        except Exception as exc:
            raise RehearsalError("pre-mutation identity attestation failed") from exc
        try:
            result = self.runner(
                argv,
                capture_output=True,
                text=True,
                timeout=self.mutation_timeout_seconds,
                check=False,
                shell=False,
            )
        except Exception as exc:
            raise RehearsalError(
                f"kernel push outcome is unknown after {type(exc).__name__}"
            ) from exc

        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
        returncode = getattr(result, "returncode", None)
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise RehearsalError("kernel push returned non-text output")
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise RehearsalError("kernel push returned no process exit status")
        matches = _SUCCESS.findall(stdout)
        version: int | None = None
        result_url: str | None = None
        exact = False
        if len(matches) == 1:
            raw_version, raw_url = matches[0]
            try:
                parsed = urlparse(raw_url)
            except ValueError:
                parsed = None
            expected_path = "/code/" + expected_slug
            if (
                parsed is not None
                and parsed.scheme == "https"
                and parsed.hostname in {"kaggle.com", "www.kaggle.com"}
                and parsed.netloc in {"kaggle.com", "www.kaggle.com"}
                and parsed.path.rstrip("/") == expected_path
                and not parsed.username
                and not parsed.password
                and not parsed.query
                and not parsed.fragment
            ):
                version = int(raw_version)
                result_url = raw_url
                exact = (
                    returncode == 0
                    and not stderr.strip()
                    and _REMOTE_IDENTITY_WARNING.search(stdout) is None
                    and _SUCCESS.fullmatch(stdout.strip()) is not None
                )
        return PushAttempt(
            exact_receipt=exact,
            notebook_slug=expected_slug,
            notebook_version=version,
            result_url=result_url,
            returncode=returncode,
            stdout_sha256=self._safe_hash(stdout),
            stderr_sha256=self._safe_hash(stderr),
            finished_at=datetime.now(UTC).isoformat(),
        )

    def current_status(self, notebook_slug: str) -> str:
        """Read slug-current status.

        Kaggle CLI 2.2.4 accepts a version in its status argument but discards it
        before the API call.  This method therefore deliberately accepts no
        version and cannot be used as exact-version recovery evidence.
        """

        if not _SLUG.fullmatch(notebook_slug):
            raise RehearsalError("notebook slug is invalid")
        client = KaggleCLI(
            self.prefix,
            runner=self.runner,
            read_timeout_seconds=self.read_timeout_seconds,
        )
        return client.notebook_status(notebook_slug)

    def pull_argv(
        self,
        *,
        notebook_slug: str,
        notebook_version: int,
        recovery_cli_path: str,
    ) -> list[str]:
        if not _SLUG.fullmatch(notebook_slug):
            raise RehearsalError("notebook slug is invalid")
        if (
            isinstance(notebook_version, bool)
            or not isinstance(notebook_version, int)
            or notebook_version <= 0
        ):
            raise RehearsalError("exact notebook version must be positive")
        if not recovery_cli_path.startswith("/") or "\x00" in recovery_cli_path:
            raise RehearsalError("recovery CLI path must be an absolute Linux path")
        if self.sdk_python_prefix is None:
            raise RehearsalError("exact source read requires an explicit Kaggle SDK Python prefix")
        return [*self.sdk_python_prefix, "-c", _SDK_EXACT_SOURCE_HELPER]

    def pull_exact_version(
        self,
        *,
        notebook_slug: str,
        notebook_version: int,
        recovery_dir: Path,
        recovery_cli_path: str,
        verify_cli_recovery_path: WSLPathVerifier,
    ) -> ExactVersionPullReceipt:
        """Read one source version through SDK GetKernel and persist bounded evidence."""

        try:
            root = recovery_dir.resolve(strict=True)
        except OSError as exc:
            raise RehearsalError("recovery directory is unavailable") from exc
        if recovery_dir.is_symlink() or not root.is_dir() or list(root.iterdir()):
            raise RehearsalError("recovery directory must be an empty regular directory")
        if not isinstance(verify_cli_recovery_path, WSLPathVerifier):
            raise RehearsalError("exact pull requires the concrete WSL path verifier")
        if verify_cli_recovery_path(root) != recovery_cli_path:
            raise RehearsalError("recovery CLI path does not map to the local directory")
        argv = self.pull_argv(
            notebook_slug=notebook_slug,
            notebook_version=notebook_version,
            recovery_cli_path=recovery_cli_path,
        )
        owner, kernel_slug = notebook_slug.split("/", 1)
        request = {
            "user_name": owner,
            "kernel_slug": kernel_slug,
            "version_label": f"v{notebook_version}",
        }
        helper_deadline = self.read_timeout_seconds * 0.9
        helper_input = {
            "request": request,
            "output_dir": recovery_cli_path,
            "max_source_bytes": _SDK_SOURCE_MAX_BYTES,
            "max_metadata_bytes": _SDK_METADATA_MAX_BYTES,
            "helper_deadline_seconds": helper_deadline,
        }
        try:
            result = self.runner(
                argv,
                input=canonical_json(helper_input),
                capture_output=True,
                text=True,
                timeout=self.read_timeout_seconds,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RehearsalError("exact-version pull did not produce evidence") from exc
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
        returncode = getattr(result, "returncode", None)
        if (
            isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or returncode != 0
            or not isinstance(stdout, str)
            or not isinstance(stderr, str)
            or stderr.strip()
        ):
            raise RehearsalError("exact-version SDK source response is ambiguous")
        if len(stdout.encode("utf-8")) > _SDK_METADATA_MAX_BYTES:
            raise RehearsalError("exact-version SDK source response exceeds its byte limit")

        def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
            parsed: dict[str, object] = {}
            for key, value in pairs:
                if key in parsed:
                    raise RehearsalError(f"duplicate SDK response key: {key}")
                parsed[key] = value
            return parsed

        def reject_constant(value: str) -> None:
            raise RehearsalError(f"nonfinite SDK response value: {value}")

        try:
            helper_result = json.loads(
                stdout,
                object_pairs_hook=pairs_hook,
                parse_constant=reject_constant,
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise RehearsalError("exact-version SDK source response is not JSON") from exc
        expected_helper_keys = {
            "schema_version",
            "status",
            "kaggle_package_version",
            "request",
            "metadata",
            "source_sha256",
            "source_bytes",
            "helper_deadline_seconds",
            "deadline_enforcement",
        }
        if (
            not isinstance(helper_result, dict)
            or set(helper_result) != expected_helper_keys
            or helper_result.get("schema_version") != 1
            or helper_result.get("status") != "PASS"
            or helper_result.get("kaggle_package_version") != "2.2.4"
            or helper_result.get("request") != request
            or helper_result.get("helper_deadline_seconds") != helper_deadline
            or helper_result.get("deadline_enforcement") != "POSIX_SETITIMER_OS_EXIT"
        ):
            raise RehearsalError("exact-version SDK source response violates its contract")

        source = root / "source.ipynb"
        entries = list(root.iterdir())
        if {entry.name for entry in entries} != {source.name} or any(
            entry.is_symlink() or not entry.is_file() for entry in entries
        ):
            raise RehearsalError("exact-version SDK source read returned an unexpected file set")
        _strict_object(source)
        raw_source_sha256 = _sha256(source)
        source_bytes = source.stat().st_size
        if (
            helper_result.get("source_sha256") != raw_source_sha256
            or helper_result.get("source_bytes") != source_bytes
            or source_bytes <= 0
            or source_bytes > _SDK_SOURCE_MAX_BYTES
        ):
            raise RehearsalError("exact-version SDK source bytes violate their helper receipt")
        normalized_source_sha256 = hashlib.sha256(
            _cli_normalized_notebook_source(source).encode("utf-8")
        ).hexdigest()
        metadata = helper_result.get("metadata")
        if not isinstance(metadata, Mapping):
            raise RehearsalError("exact-version SDK response metadata is unavailable")
        metadata_bytes = canonical_json(dict(metadata)).encode("utf-8")
        if len(metadata_bytes) > _SDK_METADATA_MAX_BYTES:
            raise RehearsalError("exact-version SDK response metadata exceeds its byte limit")
        observed_at = datetime.now(UTC).isoformat()
        artifact = {
            "schema_version": 1,
            "evidence_scope": "kaggle_sdk_get_kernel_exact_version_source",
            "exact_kernel_ref": f"{notebook_slug}/{notebook_version}",
            "notebook_version": notebook_version,
            "version_label": f"v{notebook_version}",
            "observed_at": observed_at,
            "provider_creation_time": None,
            "intent_association": "unresolved",
            "sdk_python_prefix": list(self.sdk_python_prefix or []),
            "helper_sha256": hashlib.sha256(_SDK_EXACT_SOURCE_HELPER.encode()).hexdigest(),
            "request": request,
            "recovery_cli_path": recovery_cli_path,
            "returncode": returncode,
            "stdout_sha256": self._safe_hash(stdout),
            "stderr_sha256": self._safe_hash(stderr),
            "limits": {
                "max_source_bytes": _SDK_SOURCE_MAX_BYTES,
                "max_metadata_bytes": _SDK_METADATA_MAX_BYTES,
                "parent_timeout_seconds": self.read_timeout_seconds,
                "helper_deadline_seconds": helper_deadline,
            },
            "source": {
                "path": source.name,
                "bytes": source_bytes,
                "sha256": raw_source_sha256,
                "cli_normalized_sha256": normalized_source_sha256,
            },
            "metadata": dict(metadata),
            "dataset_versions_status": "UNAVAILABLE_FROM_PROVIDER_METADATA",
        }
        artifact_path = root / EXACT_VERSION_RECEIPT_FILE
        try:
            with artifact_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical_json(artifact) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise RehearsalError("cannot exclusively persist exact-version SDK receipt") from exc
        # Re-read actual persisted bytes.  The artifact hash is evidence about
        # those bytes, rather than a caller-authored hash assertion.
        if _strict_object(artifact_path) != artifact:
            raise RehearsalError("persisted exact-version receipt changed during write")
        if _sha256(source) != raw_source_sha256:
            raise RehearsalError("exact-version SDK source changed while persisting its receipt")
        artifact_sha256 = _sha256(artifact_path)
        if _sha256(artifact_path) != artifact_sha256:
            raise RehearsalError("exact-version SDK receipt changed during final rehash")
        return ExactVersionPullReceipt(
            artifact_path=str(artifact_path),
            artifact_sha256=artifact_sha256,
            exact_kernel_ref=f"{notebook_slug}/{notebook_version}",
            notebook_version=notebook_version,
            source_path=str(source),
            source_sha256=raw_source_sha256,
            cli_normalized_source_sha256=normalized_source_sha256,
            observed_at=observed_at,
        )


class WSLPathVerifier:
    """Resolve a Windows package path through the exact WSL distribution used to push."""

    def __init__(
        self,
        distro: str,
        *,
        runner: Callable[..., object] = subprocess.run,
        timeout_seconds: float = 15,
    ) -> None:
        if not distro or "\x00" in distro:
            raise RehearsalError("WSL distribution must be resolved")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise RehearsalError("WSL path timeout must be finite and positive")
        self.distro = distro
        self.runner = runner
        self.timeout_seconds = float(timeout_seconds)

    def __call__(self, local_path: Path) -> str:
        try:
            result = self.runner(
                [
                    "wsl",
                    "-d",
                    self.distro,
                    "--",
                    "wslpath",
                    "-a",
                    "-u",
                    local_path.resolve(strict=True).as_posix(),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RehearsalError("cannot verify the WSL package mapping") from exc
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
        returncode = getattr(result, "returncode", None)
        lines = stdout.splitlines() if isinstance(stdout, str) else []
        if returncode != 0 or not isinstance(stderr, str) or stderr.strip() or len(lines) != 1:
            raise RehearsalError("WSL package mapping did not return one exact path")
        resolved = lines[0].strip()
        if not resolved.startswith("/") or "\x00" in resolved:
            raise RehearsalError("WSL package mapping is not an absolute path")
        return resolved


def _verify_exact_version_pull(
    *,
    receipt: ExactVersionPullReceipt,
    client: KaggleRehearsalCLI,
    binding: Mapping[str, object],
    notebook_slug: str,
    notebook_version: int,
    recovery_cli_path: str,
    intent_created_at: str,
) -> dict[str, object]:
    """Verify persisted SDK GetKernel evidence without inferring launch association."""

    raw_artifact_path = Path(receipt.artifact_path)
    raw_source_path = Path(receipt.source_path)
    if raw_artifact_path.is_symlink() or raw_source_path.is_symlink():
        raise RehearsalError("exact-version SDK receipt files cannot be symlinks")
    try:
        artifact_path = raw_artifact_path.resolve(strict=True)
        source_path = raw_source_path.resolve(strict=True)
    except OSError as exc:
        raise RehearsalError("exact-version SDK receipt files are unavailable") from exc
    if (
        not artifact_path.is_file()
        or not source_path.is_file()
        or artifact_path.parent != source_path.parent
        or {entry.name for entry in artifact_path.parent.iterdir()}
        != {EXACT_VERSION_RECEIPT_FILE, "source.ipynb"}
        or any(
            entry.is_symlink() or not entry.is_file() for entry in artifact_path.parent.iterdir()
        )
    ):
        raise RehearsalError("exact-version SDK receipt files are not one exclusive file set")
    if _sha256(artifact_path) != receipt.artifact_sha256:
        raise RehearsalError("exact-version SDK receipt artifact bytes changed")
    artifact = _strict_object(artifact_path)
    expected_ref = f"{notebook_slug}/{notebook_version}"
    expected_request = {
        "user_name": notebook_slug.split("/", 1)[0],
        "kernel_slug": notebook_slug.split("/", 1)[1],
        "version_label": f"v{notebook_version}",
    }
    expected_keys = {
        "schema_version",
        "evidence_scope",
        "exact_kernel_ref",
        "notebook_version",
        "version_label",
        "observed_at",
        "provider_creation_time",
        "intent_association",
        "sdk_python_prefix",
        "helper_sha256",
        "request",
        "recovery_cli_path",
        "returncode",
        "stdout_sha256",
        "stderr_sha256",
        "limits",
        "source",
        "metadata",
        "dataset_versions_status",
    }
    helper_deadline = client.read_timeout_seconds * 0.9
    expected_limits = {
        "max_source_bytes": _SDK_SOURCE_MAX_BYTES,
        "max_metadata_bytes": _SDK_METADATA_MAX_BYTES,
        "parent_timeout_seconds": client.read_timeout_seconds,
        "helper_deadline_seconds": helper_deadline,
    }
    if (
        set(artifact) != expected_keys
        or artifact.get("schema_version") != 1
        or artifact.get("evidence_scope") != "kaggle_sdk_get_kernel_exact_version_source"
        or receipt.exact_kernel_ref != expected_ref
        or receipt.notebook_version != notebook_version
        or receipt.observed_at != artifact.get("observed_at")
        or artifact.get("exact_kernel_ref") != expected_ref
        or artifact.get("notebook_version") != notebook_version
        or artifact.get("version_label") != f"v{notebook_version}"
        or artifact.get("provider_creation_time") is not None
        or artifact.get("intent_association") != "unresolved"
        or artifact.get("sdk_python_prefix") != client.sdk_python_prefix
        or artifact.get("helper_sha256")
        != hashlib.sha256(_SDK_EXACT_SOURCE_HELPER.encode()).hexdigest()
        or artifact.get("request") != expected_request
        or artifact.get("recovery_cli_path") != recovery_cli_path
        or artifact.get("returncode") != 0
        or artifact.get("limits") != expected_limits
        or artifact.get("dataset_versions_status") != "UNAVAILABLE_FROM_PROVIDER_METADATA"
        or artifact_path.name != EXACT_VERSION_RECEIPT_FILE
        or source_path.name != "source.ipynb"
    ):
        raise RehearsalError("exact-version SDK receipt violates its evidence scope")
    if artifact.get("stderr_sha256") != hashlib.sha256(b"").hexdigest():
        raise RehearsalError("exact-version SDK receipt has nonempty stderr")
    stdout_sha256 = artifact.get("stdout_sha256")
    if not isinstance(stdout_sha256, str) or not _SHA256.fullmatch(stdout_sha256):
        raise RehearsalError("exact-version SDK receipt has an invalid stdout hash")
    try:
        observed = datetime.fromisoformat(str(artifact["observed_at"]))
        intent_created = datetime.fromisoformat(intent_created_at)
    except (KeyError, TypeError, ValueError) as exc:
        raise RehearsalError("exact-version SDK receipt needs valid UTC timestamps") from exc
    if (
        any(
            value.tzinfo is None
            or value.utcoffset() is None
            or value.utcoffset().total_seconds() != 0
            for value in (observed, intent_created)
        )
        or observed < intent_created
    ):
        raise RehearsalError("exact-version SDK source was not observed after the intent")

    source = artifact.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "path",
        "bytes",
        "sha256",
        "cli_normalized_sha256",
    }:
        raise RehearsalError("exact-version SDK source binding is malformed")
    raw_sha256 = _sha256(source_path)
    source_bytes = source_path.stat().st_size
    normalized_sha256 = hashlib.sha256(
        _cli_normalized_notebook_source(source_path).encode("utf-8")
    ).hexdigest()
    if (
        source.get("path") != "source.ipynb"
        or source.get("bytes") != source_bytes
        or source_bytes <= 0
        or source_bytes > _SDK_SOURCE_MAX_BYTES
        or source.get("sha256") != raw_sha256
        or receipt.source_sha256 != raw_sha256
        or source.get("cli_normalized_sha256") != normalized_sha256
        or receipt.cli_normalized_source_sha256 != normalized_sha256
        or _sha256(source_path) != raw_sha256
    ):
        raise RehearsalError("exact-version SDK source hashes or byte bounds are inconsistent")
    expected_normalized = binding.get("cli_normalized_notebook_sha256")
    if not isinstance(expected_normalized, str) or not _SHA256.fullmatch(expected_normalized):
        raise RehearsalError("launch binding has no CLI-normalized notebook identity")
    if normalized_sha256 != expected_normalized:
        raise RehearsalError("SDK source differs from the exact CLI-normalized notebook")

    dataset_versions = binding.get("dataset_versions")
    if not isinstance(dataset_versions, Mapping) or not dataset_versions:
        raise RehearsalError("launch binding has no dataset source versions")
    expected_dataset_refs: list[str] = []
    for ref, version in dataset_versions.items():
        if (
            not isinstance(ref, str)
            or not _SLUG.fullmatch(ref)
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version <= 0
        ):
            raise RehearsalError("launch binding has an invalid dataset source version")
        expected_dataset_refs.append(ref)
    metadata = artifact.get("metadata")
    metadata_keys = {
        "id",
        "ref",
        "title",
        "slug",
        "language",
        "kernel_type",
        "is_private",
        "enable_gpu",
        "enable_internet",
        "dataset_data_sources",
        "kernel_data_sources",
        "competition_data_sources",
        "model_data_sources",
        "enable_tpu",
        "current_version_number",
        "docker_image",
        "machine_shape",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != metadata_keys:
        raise RehearsalError("exact-version SDK metadata surface is malformed")
    provider_id = metadata.get("id")
    if isinstance(provider_id, bool) or not isinstance(provider_id, int) or provider_id <= 0:
        raise RehearsalError("exact-version SDK metadata has no positive provider id")
    for field in ("is_private", "enable_gpu", "enable_internet", "enable_tpu"):
        if type(metadata.get(field)) is not bool:
            raise RehearsalError(f"exact-version SDK metadata {field} must be an exact boolean")
    current_version = metadata.get("current_version_number")
    if (
        isinstance(current_version, bool)
        or not isinstance(current_version, int)
        or current_version <= 0
    ):
        raise RehearsalError(
            "exact-version SDK metadata current_version_number must be a positive integer"
        )
    dataset_refs = metadata.get("dataset_data_sources")
    if (
        not isinstance(dataset_refs, list)
        or not all(isinstance(ref, str) for ref in dataset_refs)
        or len(dataset_refs) != len(set(dataset_refs))
        or set(dataset_refs) != set(expected_dataset_refs)
    ):
        raise RehearsalError("exact-version SDK metadata violates dataset_data_sources")
    expected_metadata = {
        "ref": notebook_slug,
        "title": binding.get("notebook_title"),
        "slug": notebook_slug.split("/", 1)[1],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "kernel_data_sources": [],
        "competition_data_sources": [binding.get("competition")],
        "model_data_sources": [],
        "enable_tpu": False,
        "current_version_number": notebook_version,
        "machine_shape": binding.get("machine_shape"),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RehearsalError(f"exact-version SDK metadata violates {key}")
    docker_image = metadata.get("docker_image")
    if not isinstance(docker_image, str):
        raise RehearsalError("exact-version SDK metadata has invalid docker_image")
    return artifact


def _positive_decimal(value: object, field: str) -> Decimal:
    try:
        parsed = decimal_nonnegative(value, field)
    except AdmissionError as exc:
        raise RehearsalError(str(exc)) from exc
    if parsed <= 0:
        raise RehearsalError(f"{field} must be positive")
    return parsed


def _quota_admission(
    *,
    store: CampaignStore,
    contract: QuotaContract,
    observation: Mapping[str, object],
    now: datetime,
) -> dict[str, object]:
    if observation.get("resource") != "GPU" or observation.get("ledger_id") != contract.ledger_id:
        raise RehearsalError("quota observation is not applicable to this Kaggle GPU ledger")
    measured_rate = contract.measured_debit_rate_per_wall_hour
    documented_bound = contract.documented_debit_rate_upper_bound_per_wall_hour
    if measured_rate is not None and documented_bound is not None:
        raise RehearsalError("measured and documented quota debit rates are mutually exclusive")
    if measured_rate is None:
        if documented_bound is None:
            raise RehearsalError("a measured or documented quota debit rate is required")
        rate = _positive_decimal(documented_bound, "documented quota debit-rate upper bound")
        source = contract.quota_rate_source
        if not isinstance(source, str) or not source.strip():
            raise RehearsalError("documented quota rate requires a nonempty source")
        rate_evidence: dict[str, object] = {
            "quota_rate_mode": "documented_upper_bound",
            "documented_debit_rate_upper_bound_per_wall_hour": str(rate),
            "quota_rate_source": source.strip(),
        }
    else:
        if contract.quota_rate_source is not None:
            raise RehearsalError("quota_rate_source is only valid for a documented rate")
        rate = _positive_decimal(measured_rate, "measured quota debit rate")
        rate_evidence = {"measured_debit_rate_per_wall_hour": str(rate)}
    final_runtime = _positive_decimal(
        contract.final_attempt_runtime_bound_hours, "final-attempt runtime bound"
    )
    platform_limit = _positive_decimal(
        contract.verified_platform_runtime_limit_hours, "verified platform runtime limit"
    )
    if (
        isinstance(contract.notebook_timeout_seconds, bool)
        or not isinstance(contract.notebook_timeout_seconds, int)
        or contract.notebook_timeout_seconds <= 0
    ):
        raise RehearsalError("notebook timeout must be a positive integer")
    rehearsal_runtime = Decimal(contract.notebook_timeout_seconds) / Decimal(3600)
    if rehearsal_runtime > platform_limit:
        raise RehearsalError("rehearsal timeout exceeds the verified platform limit")
    try:
        protected = release_quota_reserve(
            full_runtime_hours=final_runtime,
            quota_hours_per_wall_hour=rate,
            verified_runtime_limit_hours=platform_limit,
            attempts=2,
        )
    except AdmissionError as exc:
        raise RehearsalError(str(exc)) from exc
    reservation = rehearsal_runtime * rate

    snapshot = store.budget_snapshot(contract.ledger_id)
    if (
        snapshot.unit != "quota_hours"
        or snapshot.authorized_total is None
        or "kaggle" not in snapshot.resource_scope.lower()
        or "gpu" not in snapshot.resource_scope.lower()
    ):
        raise RehearsalError("Kaggle quota ledger must be finite and use quota_hours")
    if snapshot.observation_time != observation.get("observed_at"):
        raise RehearsalError("quota ledger and supplied live observation have different timestamps")
    observed_remaining = _positive_decimal(observation.get("remaining_hours"), "remaining quota")
    try:
        observed_used = decimal_nonnegative(observation.get("used_hours"), "used quota")
    except AdmissionError as exc:
        raise RehearsalError(str(exc)) from exc
    observed_total = _positive_decimal(observation.get("total_hours"), "total quota")
    if observed_used + observed_remaining != observed_total:
        raise RehearsalError("supplied Kaggle quota components do not reconcile")
    if snapshot.authorized_total != observed_total:
        raise RehearsalError("Kaggle quota ledger total differs from the live observation")
    gross_remaining = (
        snapshot.authorized_total
        - snapshot.confirmed_spend
        - snapshot.unreconciled_spend
        - snapshot.outstanding_reservations
    )
    if gross_remaining != observed_remaining:
        raise RehearsalError("quota ledger does not reconcile to the supplied remaining quota")
    if snapshot.protected_reserve < protected:
        raise RehearsalError("quota ledger does not protect two bounded final attempts")
    active = observation.get("active_gpu_jobs")
    if isinstance(active, bool) or not isinstance(active, int) or active != 0:
        raise RehearsalError("verified Kaggle active GPU job count must be zero")
    try:
        admitted = admit_resource_action(
            observation={
                "observed_at": observation.get("observed_at"),
                "remaining": str(observed_remaining),
            },
            now=now,
            reservation=reservation,
            protected_reserve=snapshot.protected_reserve,
            active_gpu_jobs=active,
        )
    except AdmissionError as exc:
        raise RehearsalError(str(exc)) from exc
    return {
        **admitted,
        "ledger_id": contract.ledger_id,
        **rate_evidence,
        "notebook_timeout_seconds": contract.notebook_timeout_seconds,
        "final_attempt_runtime_bound_hours": str(final_runtime),
        "verified_platform_runtime_limit_hours": str(platform_limit),
        "required_two_attempt_reserve": str(protected),
    }


def _require_rehearsal_permission(store: CampaignStore) -> tuple[str, bool]:
    campaign = store.get_campaign()
    authorization = campaign.get("authorization")
    if not isinstance(authorization, Mapping):
        raise RehearsalError("campaign authorization is unavailable")
    scoped = authorization.get("private_kaggle_rehearsals") is True
    legacy_routine = authorization.get("routine_runs_and_submissions") is True
    if not (scoped or legacy_routine):
        raise RehearsalError(
            "explicit private Kaggle rehearsal or routine run permission is required"
        )
    source = authorization.get("source")
    if not isinstance(source, str) or not source.strip():
        raise RehearsalError("Kaggle rehearsal permission must retain its explicit source")
    return source, scoped


def _validate_authorization_source_binding(
    registered: Mapping[str, object],
    *,
    authorization_source: str,
    required: bool,
) -> None:
    spec = registered.get("spec")
    if not isinstance(spec, Mapping):
        raise RehearsalError("registered run has no immutable specification")
    expected = spec.get("authorization_source_sha256")
    if expected is None:
        if required:
            raise RehearsalError(
                "private Kaggle rehearsal spec does not bind its authorization source"
            )
        return
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise RehearsalError("authorization_source_sha256 is not a valid SHA-256")
    actual = hashlib.sha256(authorization_source.encode("utf-8")).hexdigest()
    if actual != expected:
        raise RehearsalError("campaign authorization source changed after run review")


def _require_no_internal_kaggle_launch(store: CampaignStore, run_id: str) -> None:
    active_states = {
        JobState.APPROVED.value,
        JobState.LAUNCH_INTENT.value,
        JobState.LAUNCH_UNKNOWN.value,
        JobState.RUNNING.value,
    }
    for run in store.list_runs():
        if run["run_id"] == run_id or run["state"] not in active_states:
            continue
        execution = run["spec"].get("execution", {})
        if execution.get("host") == "kaggle":
            raise RehearsalError("another Kaggle launch or GPU run is unresolved")


def _expected_rehearsal_binding(
    *,
    identity: PackageIdentity,
    package_cli_path: str,
    quota: Mapping[str, object],
) -> Mapping[str, object]:
    binding = {
        "schema_version": 1,
        "notebook_slug": identity.notebook_slug,
        "notebook_title": identity.notebook_title,
        "competition": identity.competition,
        "dataset_versions": dict(identity.dataset_versions),
        "release_digest": identity.release_digest,
        "package_manifest_sha256": identity.package_manifest_sha256,
        "kernel_metadata_sha256": identity.kernel_metadata_sha256,
        "artifact_lock_sha256": identity.artifact_lock_sha256,
        "notebook_sha256": identity.notebook_sha256,
        "cli_normalized_notebook_sha256": identity.cli_normalized_notebook_sha256,
        "package_cli_path": package_cli_path,
        "machine_shape": identity.machine_shape,
        "notebook_timeout_seconds": quota["notebook_timeout_seconds"],
        "final_attempt_runtime_bound_hours": quota["final_attempt_runtime_bound_hours"],
        "verified_platform_runtime_limit_hours": quota["verified_platform_runtime_limit_hours"],
        "required_two_attempt_reserve": quota["required_two_attempt_reserve"],
    }
    for key in (
        "measured_debit_rate_per_wall_hour",
        "quota_rate_mode",
        "documented_debit_rate_upper_bound_per_wall_hour",
        "quota_rate_source",
    ):
        if key in quota:
            binding[key] = quota[key]
    return MappingProxyType(binding)


def _validate_run_binding(
    *,
    registered: Mapping[str, object],
    identity: PackageIdentity,
    package_cli_path: str,
    quota: Mapping[str, object],
    client: KaggleRehearsalCLI,
) -> None:
    spec = registered.get("spec")
    if not isinstance(spec, Mapping):
        raise RehearsalError("registered run has no immutable specification")
    execution = spec.get("execution")
    if not isinstance(execution, Mapping) or execution.get("host") != "kaggle":
        raise RehearsalError("rehearsal run host must be kaggle, not a worker or SSH host")
    expected_argv = client.push_argv(
        package_cli_path=package_cli_path,
        accelerator=identity.machine_shape,
        notebook_timeout_seconds=int(quota["notebook_timeout_seconds"]),
    )
    if execution.get("argv") != expected_argv:
        raise RehearsalError("registered run does not bind the exact installed CLI push command")
    if execution.get("working_directory") != package_cli_path:
        raise RehearsalError(
            "registered Kaggle working directory must be the uploaded package path"
        )
    if execution.get("max_steps_or_clips") != 1:
        raise RehearsalError("Kaggle rehearsal must be exactly one push attempt")
    try:
        bound_quota = decimal_nonnegative(
            execution.get("max_quota_hours"), "execution.max_quota_hours"
        )
    except AdmissionError as exc:
        raise RehearsalError(str(exc)) from exc
    if bound_quota != Decimal(str(quota["reservation"])):
        raise RehearsalError("registered run quota bound differs from the reserved worst case")
    expected_binding = dict(
        _expected_rehearsal_binding(
            identity=identity,
            package_cli_path=package_cli_path,
            quota=quota,
        )
    )
    if spec.get("kaggle_rehearsal") != expected_binding:
        raise RehearsalError("registered run does not bind the exact package and quota contract")
    source = spec.get("source")
    if not isinstance(source, Mapping) or (
        source.get("source_bundle_sha256") != identity.package_manifest_sha256
        or source.get("dependency_manifest_sha256") != identity.artifact_lock_sha256
        or source.get("effective_config_sha256") != identity.kernel_metadata_sha256
    ):
        raise RehearsalError("registered run source identity differs from the package")


def _intent_is_dispatched(
    store: CampaignStore,
    *,
    intent_id: str,
    run_id: str,
    run_spec_sha256: str,
    fencing_token: int,
) -> None:
    intent = store.get_intent(intent_id)
    expected = (
        "launch",
        run_id,
        run_spec_sha256,
        fencing_token,
        IntentState.DISPATCHED.value,
    )
    actual = (
        intent["intent_kind"],
        intent["subject_id"],
        intent["subject_digest"],
        intent["fencing_token"],
        intent["state"],
    )
    if actual != expected:
        raise RehearsalError("persisted launch intent is not the exact dispatched mutation")


def _require_live_review_lease(store: CampaignStore, review_lease: ReviewLease) -> tuple[str, int]:
    if review_lease.store is not store:
        raise RehearsalError("review lease belongs to a different campaign store")
    token = review_lease.fencing_token
    stream = review_lease._stream
    if (
        isinstance(token, bool)
        or not isinstance(token, int)
        or token <= 0
        or stream is None
        or stream.closed
    ):
        raise RehearsalError("caller must hold the campaign OS review lock")
    return review_lease.owner, token


def _unknown(
    store: CampaignStore,
    *,
    intent_id: str,
    phase: str,
    push: PushAttempt | None = None,
    error: BaseException | None = None,
) -> dict[str, object]:
    receipt: dict[str, object] = {"phase": phase, "status": "UNKNOWN"}
    if push is not None:
        receipt["push"] = push.as_dict()
    if error is not None:
        receipt["error_type"] = type(error).__name__
    intent = store.mark_intent_unknown(intent_id, receipt=receipt)
    return {
        "status": "UNKNOWN",
        "intent_id": intent_id,
        "reservation_ids": intent["reservation_ids"],
        "reservations_retained": True,
        "receipt": receipt,
    }


def authorize_and_launch_rehearsal(
    *,
    store: CampaignStore,
    run_id: str,
    package_dir: Path,
    package_cli_path: str,
    identity: PackageIdentity,
    quota_contract: QuotaContract,
    quota_observation: Mapping[str, object],
    client: KaggleRehearsalCLI,
    decision_id: str,
    intent_id: str,
    request_id: str,
    reservation_id: str,
    review_lease: ReviewLease,
    verify_cli_package_path: WSLPathVerifier,
    now: datetime,
) -> dict[str, object]:
    """Preflight, reserve, persist, dispatch, and invoke exactly one kernel push."""

    package_receipt = preflight_package(package_dir, identity)
    reviewer, fencing_token = _require_live_review_lease(store, review_lease)
    if not isinstance(verify_cli_package_path, WSLPathVerifier):
        raise RehearsalError("launch requires the concrete read-only WSL path verifier")
    if verify_cli_package_path(package_dir) != package_cli_path:
        raise RehearsalError("CLI package path does not map to the reviewed local package")
    authorization_source, scoped_permission = _require_rehearsal_permission(store)
    _require_no_internal_kaggle_launch(store, run_id)
    registered = store.get_run(run_id)
    _validate_authorization_source_binding(
        registered,
        authorization_source=authorization_source,
        required=scoped_permission,
    )
    quota = _quota_admission(
        store=store,
        contract=quota_contract,
        observation=quota_observation,
        now=now,
    )
    _validate_run_binding(
        registered=registered,
        identity=identity,
        package_cli_path=package_cli_path,
        quota=quota,
        client=client,
    )
    description_tag = f"biohub-{registered['run_spec_sha256'][:24]}"
    store.authorize_launch(
        run_id,
        decision_id=decision_id,
        reviewer=reviewer,
        reservations=[
            {
                "ledger_id": quota_contract.ledger_id,
                "reservation_id": reservation_id,
                "amount": quota["reservation"],
                "note": "one bounded Kaggle notebook push",
            }
        ],
        reason="exact private offline GPU rehearsal fits quota outside two final attempts",
        quality_class="operational_rehearsal",
        evidence_paths=[str(package_dir / "package-manifest.json")],
        intent_id=intent_id,
        request_id=request_id,
        description_tag=description_tag,
        fencing_token=fencing_token,
        now=now,
    )
    store.validate_dispatch(
        intent_id,
        run_id,
        str(registered["run_spec_sha256"]),
        fencing_token,
        now=now,
    )

    def verify_intent() -> None:
        _require_live_review_lease(store, review_lease)
        # Re-read every package byte and the controller-to-WSL mapping immediately
        # before the only external mutation.
        preflight_package(package_dir, identity)
        if verify_cli_package_path(package_dir) != package_cli_path:
            raise RehearsalError("CLI package mapping changed before dispatch")
        _intent_is_dispatched(
            store,
            intent_id=intent_id,
            run_id=run_id,
            run_spec_sha256=str(registered["run_spec_sha256"]),
            fencing_token=fencing_token,
        )

    try:
        push = client.push_once(
            package_cli_path=package_cli_path,
            accelerator=identity.machine_shape,
            notebook_timeout_seconds=quota_contract.notebook_timeout_seconds,
            expected_slug=identity.notebook_slug,
            verify_persisted_intent=verify_intent,
        )
    except RehearsalError as exc:
        return _unknown(store, intent_id=intent_id, phase="push", error=exc)
    if not push.exact_receipt or push.notebook_version is None:
        return _unknown(store, intent_id=intent_id, phase="push_receipt", push=push)
    launch_receipt = {
        "push": push.as_dict(),
        "exact_kernel_ref": f"{identity.notebook_slug}/{push.notebook_version}",
        "evidence_scope": "exact_push_acknowledgment",
        "kernel_status": None,
        "kernel_status_scope": "not_read_slug_current_only",
        "package": package_receipt,
        "quota": quota,
    }
    confirmed = store.confirm_launch(
        intent_id,
        provider_job_id=launch_receipt["exact_kernel_ref"],
        receipt=launch_receipt,
    )
    return {
        "status": "CONFIRMED",
        "intent_id": intent_id,
        "exact_kernel_ref": launch_receipt["exact_kernel_ref"],
        "kernel_status": None,
        "reservation_ids": confirmed["reservation_ids"],
    }


def reconcile_rehearsal(
    *,
    store: CampaignStore,
    intent_id: str,
    client: KaggleRehearsalCLI,
    exact_version: int | None = None,
    recovery_dir: Path | None = None,
    recovery_cli_path: str | None = None,
    verify_cli_recovery_path: WSLPathVerifier | None = None,
) -> dict[str, object]:
    """Reconcile only an exact persisted push acknowledgment.

    For an ambiguous push, SDK GetKernel with an explicit ``version_label`` can
    prove candidate source and operational metadata.  That provider surface has
    no reliable creation time tying the candidate to this intent, while status
    remains slug-current.  Partial source evidence therefore remains UNKNOWN.
    """

    intent = store.get_intent(intent_id)
    if intent["intent_kind"] != "launch" or intent["state"] != IntentState.UNKNOWN.value:
        raise RehearsalError("recovery requires an UNKNOWN launch intent")
    run = store.get_run(intent["subject_id"])
    binding = run["spec"].get("kaggle_rehearsal")
    if not isinstance(binding, Mapping):
        raise RehearsalError("UNKNOWN run has no Kaggle rehearsal binding")
    slug = binding.get("notebook_slug")
    if not isinstance(slug, str) or not _SLUG.fullmatch(slug):
        raise RehearsalError("UNKNOWN run has no exact notebook slug")

    stored_push = intent.get("receipt", {}).get("push")
    stored_version = (
        stored_push.get("notebook_version") if isinstance(stored_push, Mapping) else None
    )
    stored_exact = isinstance(stored_push, Mapping) and stored_push.get("exact_receipt") is True
    if stored_exact:
        if (
            isinstance(stored_version, bool)
            or not isinstance(stored_version, int)
            or stored_version <= 0
            or stored_push.get("notebook_slug") != slug
        ):
            raise RehearsalError("persisted exact push acknowledgment is malformed")
        if exact_version is not None and exact_version != stored_version:
            raise RehearsalError("supplied version conflicts with the exact push acknowledgment")
        exact_ref = f"{slug}/{stored_version}"
        receipt = {
            "exact_kernel_ref": exact_ref,
            "evidence_scope": "persisted_exact_push_acknowledgment",
            "push": dict(stored_push),
            "kernel_status": None,
            "kernel_status_scope": "not_read_slug_current_only",
        }
        reconciled = store.reconcile_launch(
            intent_id,
            outcome="RUNNING",
            provider_job_id=exact_ref,
            receipt=receipt,
        )
        return {
            "status": "CONFIRMED",
            "intent_id": intent_id,
            "exact_kernel_ref": exact_ref,
            "kernel_status": None,
            "reservation_ids": reconciled["reservation_ids"],
        }

    # A version parsed from a non-exact push response is only a candidate.  It
    # never bypasses the exact SDK source read and semantic identity proof below.
    if stored_version is not None:
        if (
            isinstance(stored_version, bool)
            or not isinstance(stored_version, int)
            or stored_version <= 0
        ):
            raise RehearsalError("persisted candidate version is malformed")
        if exact_version is not None and exact_version != stored_version:
            raise RehearsalError("supplied version conflicts with the persisted candidate version")
        exact_version = stored_version
    if (
        exact_version is None
        or isinstance(exact_version, bool)
        or not isinstance(exact_version, int)
        or exact_version <= 0
    ):
        return {
            "status": "UNKNOWN",
            "intent_id": intent_id,
            "reservation_ids": intent["reservation_ids"],
            "reservations_retained": True,
            "reason": "no exact candidate version; absence was not inferred",
        }
    exact_ref = f"{slug}/{exact_version}"
    if recovery_dir is None or recovery_cli_path is None or verify_cli_recovery_path is None:
        return {
            "status": "UNKNOWN",
            "intent_id": intent_id,
            "exact_kernel_ref": exact_ref,
            "reservation_ids": intent["reservation_ids"],
            "reservations_retained": True,
            "reason": "candidate version lacks a controlled exact-version SDK source read",
        }
    try:
        pull_receipt = client.pull_exact_version(
            notebook_slug=slug,
            notebook_version=exact_version,
            recovery_dir=recovery_dir,
            recovery_cli_path=recovery_cli_path,
            verify_cli_recovery_path=verify_cli_recovery_path,
        )
        artifact = _verify_exact_version_pull(
            receipt=pull_receipt,
            client=client,
            binding=binding,
            notebook_slug=slug,
            notebook_version=exact_version,
            recovery_cli_path=recovery_cli_path,
            intent_created_at=str(intent["created_at"]),
        )
    except RehearsalError as exc:
        return {
            "status": "UNKNOWN",
            "intent_id": intent_id,
            "exact_kernel_ref": exact_ref,
            "reservation_ids": intent["reservation_ids"],
            "reservations_retained": True,
            "reason": f"candidate exact-version SDK source proof failed: {exc}",
        }
    return {
        "status": "UNKNOWN",
        "intent_id": intent_id,
        "exact_kernel_ref": exact_ref,
        "reservation_ids": intent["reservation_ids"],
        "reservations_retained": True,
        "reason": (
            "exact SDK source read verified candidate bytes and operational metadata, but the provider "
            "surface supplies no creation time or intent association; slug-current status "
            "cannot resolve this launch"
        ),
        "partial_evidence": {
            "receipt": pull_receipt.as_dict(),
            "artifact": artifact,
            "identity_verified": True,
            "intent_association": "unresolved",
        },
    }
