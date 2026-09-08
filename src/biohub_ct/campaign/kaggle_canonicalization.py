"""Narrow proof for the E0 R3 Kaggle title-canonicalization event.

The frozen package requested one notebook slug, while Kaggle CLI 2.2.4 warned
that the title canonicalized to another slug and reported the created version
there.  This module verifies that one event from immutable local evidence.  It
contains no provider mutation operation and never changes the requested package
identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from biohub_ct.campaign.contracts import canonical_json
from biohub_ct.campaign.kaggle_rehearsal import E0_R3_PACKAGE_IDENTITY, PackageIdentity


class CanonicalizationError(RuntimeError):
    """The canonicalization evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class CanonicalizationEvidence:
    """Paths to the four independently persisted inputs for the mapping proof."""

    launch_receipt_path: Path
    recovered_push_stdout_path: Path
    sdk_receipt_path: Path
    sdk_source_path: Path


@dataclass(frozen=True)
class CanonicalizationResult:
    """Validated actual provider identity without replacing the requested identity."""

    actual_notebook_slug: str
    notebook_version: int
    receipt_path: Path
    receipt_sha256: str
    source_sha256: str
    cli_normalized_source_sha256: str


SCHEMA_VERSION: Final = 1
RECEIPT_KIND: Final = "e0_r3_kaggle_title_canonicalization_proof"
R3_RUN_ID: Final = "e0-r3-kaggle-full-20260908-01"
R3_INTENT_ID: Final = f"{R3_RUN_ID}-intent"
R3_REQUEST_ID: Final = f"{R3_RUN_ID}-launch"
R3_LAUNCH_RECEIPT_SHA256: Final = "8a33fe730b1c6f360bb27bd6efb09986c60fa713108032fa9dade1b8c7577423"
R3_PUSH_STDOUT_SHA256: Final = "49bdb98e7730915acfa463891372761b9d98b06e9c848cd17502908d055a919b"
R3_SDK_V1_RECEIPT_SHA256: Final = "4e931ee6c2fed2b1ea45e55a09423a0d05a660f62cb0f9f3e3d2d43ce9d0b9f9"
EMPTY_SHA256: Final = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
KAGGLE_PACKAGE_VERSION: Final = "2.2.4"
KAGGLE_API_EXTENDED_SHA256: Final = (
    "210e589ab1f7359ffb560aacfe1c003ac1bc844c366f4ff709bf3d13de4d5458"
)
KAGGLE_SLUGIFY_SHA256: Final = "3103ecc34bb68362d4fbc0414fb2b40ce946b40848c68287403e1f5a5b4a9656"
_WARNING: Final = (
    "Your kernel title does not resolve to the specified id. This may result in surprising "
    "behavior. We suggest making your title something that resolves to the specified id. See "
    "https://en.wikipedia.org/wiki/Clean_URL#Slug for more information on how slugs are "
    "determined."
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SLUG = re.compile(r"[a-z0-9_-]+/[a-z0-9_-]+\Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CanonicalizationError(f"cannot read bound artifact: {path.name}") from exc
    return digest.hexdigest()


def _strict_object(path: Path) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalizationError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CanonicalizationError(f"nonfinite JSON value in {path.name}: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalizationError(f"cannot read strict JSON from {path.name}") from exc
    if not isinstance(value, dict):
        raise CanonicalizationError(f"{path.name} must contain a JSON object")
    return value


def _require_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CanonicalizationError(f"{field} must be an object")
    return value


def _identity_json(identity: PackageIdentity) -> dict[str, object]:
    return json.loads(json.dumps(asdict(identity), allow_nan=False))


def _canonicalized_title_slug(title: str) -> str:
    """The relevant Kaggle 2.2.4 slugify behavior for this ASCII R3 title."""

    if not title or not title.isascii():
        raise CanonicalizationError("this proof supports only the bound ASCII notebook title")
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", title.strip()).strip("-").lower()
    if not value or not re.fullmatch(r"[a-z0-9_-]+", value):
        raise CanonicalizationError("bound title does not produce a canonical Kaggle slug")
    return value


def _normalized_notebook(path: Path) -> tuple[str, dict[str, object]]:
    notebook = _strict_object(path)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise CanonicalizationError("SDK notebook cells must be a list")
    for cell in cells:
        if not isinstance(cell, dict):
            raise CanonicalizationError("SDK notebook cells must be objects")
        if cell.get("cell_type") == "code" and "outputs" in cell:
            cell["outputs"] = []
        source = cell.get("source")
        if isinstance(source, list):
            if not all(isinstance(part, str) for part in source):
                raise CanonicalizationError("SDK notebook source lists must contain strings")
            cell["source"] = "".join(source)
    encoded = json.dumps(notebook)
    return _sha256_bytes(encoded.encode("utf-8")), notebook


def _resolved_file(path: Path, label: str) -> Path:
    # Resolve only after checking the lexical path.  Otherwise ``resolve``
    # erases the evidence that a caller redirected a bound artifact through a
    # symlink or junction after receipt creation.
    absolute = Path(os.path.abspath(path))
    try:
        for component in (absolute, *absolute.parents):
            is_junction = getattr(component, "is_junction", lambda: False)
            if component.is_symlink() or is_junction():
                raise CanonicalizationError(f"{label} path cannot contain links or junctions")
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise CanonicalizationError(f"{label} is unavailable") from exc
    if not resolved.is_file():
        raise CanonicalizationError(f"{label} must be a regular file")
    return resolved


def _validate_launch_receipt(path: Path, identity: PackageIdentity) -> dict[str, object]:
    if identity != E0_R3_PACKAGE_IDENTITY:
        raise CanonicalizationError("canonicalization proof is restricted to frozen E0 R3")
    if _sha256_file(path) != R3_LAUNCH_RECEIPT_SHA256:
        raise CanonicalizationError("launch receipt does not match the persisted R3 mutation")
    receipt = _strict_object(path)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "e0_r3_kaggle_rehearsal_operator_receipt"
        or receipt.get("status") != "UNKNOWN"
        or receipt.get("launch_requested") is not True
        or receipt.get("launch_readiness") != "CONSUMED"
        or receipt.get("retry_forbidden_without_reconciliation") is not True
        or receipt.get("package_identity") != _identity_json(identity)
    ):
        raise CanonicalizationError("launch receipt does not bind the frozen UNKNOWN intent")
    launch_input = _require_object(receipt.get("launch_input"), "launch_input")
    launch_result = _require_object(receipt.get("launch_result"), "launch_result")
    inner_receipt = _require_object(launch_result.get("receipt"), "launch_result.receipt")
    push = _require_object(inner_receipt.get("push"), "launch_result.receipt.push")
    if (
        launch_input.get("run_id") != R3_RUN_ID
        or launch_input.get("intent_id") != R3_INTENT_ID
        or launch_input.get("request_id") != R3_REQUEST_ID
        or launch_result.get("intent_id") != R3_INTENT_ID
        or launch_result.get("status") != "UNKNOWN"
        or inner_receipt.get("status") != "UNKNOWN"
        or inner_receipt.get("phase") != "push_receipt"
        or push.get("exact_receipt") is not False
        or push.get("notebook_slug") != identity.notebook_slug
        or push.get("notebook_version") is not None
        or push.get("result_url") is not None
        or push.get("returncode") != 0
        or push.get("stdout_sha256") != R3_PUSH_STDOUT_SHA256
        or push.get("stderr_sha256") != EMPTY_SHA256
    ):
        raise CanonicalizationError("launch receipt push binding is inconsistent")
    package = _require_object(receipt.get("package"), "package")
    if (
        package.get("status") != "PASS"
        or package.get("notebook_slug") != identity.notebook_slug
        or package.get("release_digest") != identity.release_digest
        or package.get("cli_normalized_notebook_sha256") != identity.cli_normalized_notebook_sha256
        or package.get("private") is not True
        or package.get("internet") is not False
        or package.get("gpu") is not True
        or package.get("machine_shape") != identity.machine_shape
    ):
        raise CanonicalizationError("launch receipt package flags or identity are inconsistent")
    return receipt


def _validate_transcript(path: Path, identity: PackageIdentity) -> tuple[str, int, str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise CanonicalizationError("cannot read recovered push stdout") from exc
    if _sha256_bytes(raw) != R3_PUSH_STDOUT_SHA256:
        raise CanonicalizationError("recovered transcript does not match original push stdout")
    owner = identity.notebook_slug.split("/", 1)[0]
    actual_slug = f"{owner}/{_canonicalized_title_slug(identity.notebook_title)}"
    result_url = f"https://www.kaggle.com/code/{actual_slug}"
    expected = f"{_WARNING}\nKernel version 1 successfully pushed.  Please check progress at {result_url}\n"
    if text != expected:
        raise CanonicalizationError("push transcript is not the exact warning and success response")
    if not _SLUG.fullmatch(actual_slug):
        raise CanonicalizationError("canonicalized provider slug is invalid")
    return actual_slug, 1, result_url


def _validate_sdk_read(
    receipt_path: Path,
    source_path: Path,
    identity: PackageIdentity,
    actual_slug: str,
    version: int,
) -> tuple[dict[str, object], str, str]:
    if _sha256_file(receipt_path) != R3_SDK_V1_RECEIPT_SHA256:
        raise CanonicalizationError("SDK receipt does not match the reviewed exact-version read")
    receipt = _strict_object(receipt_path)
    request = _require_object(receipt.get("request"), "SDK request")
    metadata = _require_object(receipt.get("metadata"), "SDK metadata")
    owner, slug = actual_slug.split("/", 1)
    if request != {"user_name": owner, "kernel_slug": slug, "version_label": f"v{version}"}:
        raise CanonicalizationError("SDK read was not pinned to the transcript's exact version")
    source_sha256 = _sha256_file(source_path)
    if receipt.get("source_sha256") != source_sha256:
        raise CanonicalizationError("SDK source bytes do not match the SDK receipt")
    normalized_sha256, notebook = _normalized_notebook(source_path)
    if normalized_sha256 != identity.cli_normalized_notebook_sha256:
        raise CanonicalizationError("SDK exact-version source differs from the frozen notebook")
    if len(notebook["cells"]) != 14:
        raise CanonicalizationError("SDK notebook does not have the frozen 14-cell structure")
    metadata_core = {
        "ref": actual_slug,
        "title": identity.notebook_title,
        "slug": slug,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "current_version_number": version,
        "machine_shape": identity.machine_shape,
    }
    if any(metadata.get(key) != value for key, value in metadata_core.items()):
        raise CanonicalizationError("SDK metadata identity or execution flags are inconsistent")
    datasets = metadata.get("dataset_data_sources")
    expected_datasets = sorted(ref for ref, _ in identity.dataset_versions)
    if (
        not isinstance(datasets, list)
        or not all(isinstance(ref, str) for ref in datasets)
        or sorted(datasets) != expected_datasets
    ):
        raise CanonicalizationError("SDK metadata dataset sources differ from the frozen package")
    if metadata.get("competition_data_sources") != [identity.competition]:
        raise CanonicalizationError(
            "SDK metadata competition source differs from the frozen package"
        )
    if metadata.get("kernel_data_sources") != [] or metadata.get("model_data_sources") != []:
        raise CanonicalizationError("SDK metadata contains unexpected kernel or model sources")
    return receipt, source_sha256, normalized_sha256


def _binding(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256_file(path)}


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise CanonicalizationError("generated_at must be an ISO-8601 timestamp")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CanonicalizationError("generated_at must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != UTC.utcoffset(timestamp):
        raise CanonicalizationError("generated_at must use a UTC offset")
    return value


def _expected_proof(
    evidence: CanonicalizationEvidence,
    *,
    generated_at: str,
    identity: PackageIdentity,
) -> tuple[dict[str, object], str, int, str, str]:
    launch_path = _resolved_file(evidence.launch_receipt_path, "launch receipt")
    stdout_path = _resolved_file(evidence.recovered_push_stdout_path, "push transcript")
    sdk_receipt_path = _resolved_file(evidence.sdk_receipt_path, "SDK receipt")
    sdk_source_path = _resolved_file(evidence.sdk_source_path, "SDK source")
    _validate_launch_receipt(launch_path, identity)
    actual_slug, version, result_url = _validate_transcript(stdout_path, identity)
    sdk_receipt, source_sha256, normalized_sha256 = _validate_sdk_read(
        sdk_receipt_path, sdk_source_path, identity, actual_slug, version
    )
    proof = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": "PASS",
        "generated_at": generated_at,
        "requested_package_identity": _identity_json(identity),
        "original_intent": {
            "run_id": R3_RUN_ID,
            "intent_id": R3_INTENT_ID,
            "request_id": R3_REQUEST_ID,
            "launch_receipt": _binding(launch_path),
        },
        "push_transcript": {
            **_binding(stdout_path),
            "stdout_sha256": R3_PUSH_STDOUT_SHA256,
            "stderr_sha256": EMPTY_SHA256,
            "returncode": 0,
            "warning": "TITLE_DOES_NOT_RESOLVE_TO_SPECIFIED_ID",
        },
        "client_provenance": {
            "kaggle_package_version": KAGGLE_PACKAGE_VERSION,
            "kaggle_api_extended_sha256": KAGGLE_API_EXTENDED_SHA256,
            "kaggle_slugify_sha256": KAGGLE_SLUGIFY_SHA256,
        },
        "mapping": {
            "requested_notebook_slug": identity.notebook_slug,
            "actual_notebook_slug": actual_slug,
            "notebook_version": version,
            "result_url": result_url,
            "cause": "KAGGLE_TITLE_CANONICALIZATION",
        },
        "sdk_exact_source_read": {
            "receipt": _binding(sdk_receipt_path),
            "source": {
                **_binding(sdk_source_path),
                "cli_normalized_sha256": normalized_sha256,
            },
            "request": sdk_receipt["request"],
            "metadata": sdk_receipt["metadata"],
        },
        "constraints": {
            "read_only": True,
            "provider_mutations": [],
            "requested_identity_preserved": True,
        },
    }
    return proof, actual_slug, version, source_sha256, normalized_sha256


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise CanonicalizationError("canonicalization receipt already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Linking a complete temporary file is an atomic no-replace publish on
        # the local filesystems supported by this campaign.  It cannot silently
        # replace a proof that an operator has already reviewed.
        os.link(temporary, path)
        temporary.unlink()
    except FileExistsError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise CanonicalizationError("canonicalization receipt already exists") from exc
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def create_canonicalization_receipt(
    evidence: CanonicalizationEvidence,
    receipt_path: Path,
    *,
    identity: PackageIdentity = E0_R3_PACKAGE_IDENTITY,
) -> CanonicalizationResult:
    """Validate persisted evidence and atomically write the narrow mapping proof."""

    generated_at = datetime.now(UTC).isoformat()
    proof, _, _, _, _ = _expected_proof(evidence, generated_at=generated_at, identity=identity)
    destination = receipt_path.resolve()
    bound_paths = {
        Path(binding["path"])
        for binding in (
            proof["original_intent"]["launch_receipt"],
            proof["push_transcript"],
            proof["sdk_exact_source_read"]["receipt"],
            proof["sdk_exact_source_read"]["source"],
        )
    }
    if destination in bound_paths:
        raise CanonicalizationError("mapping receipt cannot overwrite a bound evidence artifact")
    _atomic_write(destination, proof)
    return verify_canonicalization_receipt(destination, identity=identity)


def verify_canonicalization_receipt(
    path: Path,
    *,
    identity: PackageIdentity = E0_R3_PACKAGE_IDENTITY,
) -> CanonicalizationResult:
    """Re-read every bound artifact and validate a mapping receipt without network access."""

    receipt_path = _resolved_file(path, "canonicalization receipt")
    receipt = _strict_object(receipt_path)
    generated_at = _validate_timestamp(receipt.get("generated_at"))
    original_intent = _require_object(receipt.get("original_intent"), "original_intent")
    transcript = _require_object(receipt.get("push_transcript"), "push_transcript")
    sdk = _require_object(receipt.get("sdk_exact_source_read"), "sdk_exact_source_read")
    launch_binding = _require_object(original_intent.get("launch_receipt"), "launch_receipt")
    sdk_receipt_binding = _require_object(sdk.get("receipt"), "SDK receipt binding")
    sdk_source_binding = _require_object(sdk.get("source"), "SDK source binding")
    for binding, label in (
        (launch_binding, "launch receipt"),
        (transcript, "push transcript"),
        (sdk_receipt_binding, "SDK receipt"),
        (sdk_source_binding, "SDK source"),
    ):
        if not isinstance(binding.get("path"), str) or not _SHA256.fullmatch(
            str(binding.get("sha256"))
        ):
            raise CanonicalizationError(f"{label} binding is malformed")
    evidence = CanonicalizationEvidence(
        launch_receipt_path=Path(str(launch_binding["path"])),
        recovered_push_stdout_path=Path(str(transcript["path"])),
        sdk_receipt_path=Path(str(sdk_receipt_binding["path"])),
        sdk_source_path=Path(str(sdk_source_binding["path"])),
    )
    expected, actual_slug, version, source_sha256, normalized_sha256 = _expected_proof(
        evidence, generated_at=generated_at, identity=identity
    )
    if canonical_json(receipt) != canonical_json(expected):
        raise CanonicalizationError("canonicalization receipt fields do not match bound evidence")
    return CanonicalizationResult(
        actual_notebook_slug=actual_slug,
        notebook_version=version,
        receipt_path=receipt_path,
        receipt_sha256=_sha256_file(receipt_path),
        source_sha256=source_sha256,
        cli_normalized_source_sha256=normalized_sha256,
    )
