"""Bounded, read-only retrieval of one exact Kaggle notebook version's outputs.

Kaggle CLI 2.2.4 parses a version in ``kernels output`` but does not put it on
the output-list request.  This adapter calls the same installed SDK explicitly
with ``version_label='vN'`` and treats the downloaded bytes as transport
evidence only.  It never persists signed output URLs or raw subprocess output.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from biohub_ct.campaign.contracts import canonical_json


class KaggleOutputError(RuntimeError):
    """Exact-version output retrieval or proof construction failed."""


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SLUG = re.compile(r"[a-z0-9_-]+/[a-z0-9_-]+\Z")
TRANSPORT_RECEIPT_FILENAME = "kaggle-output-transport-receipt.json"
EXACT_VERSION_PROOF_FILENAME = "kaggle-exact-version-output-proof.json"
DOWNLOAD_DIRECTORY = "downloaded"


@dataclass(frozen=True)
class ExactOutputSpec:
    notebook_slug: str
    notebook_version: int
    release_digest: str
    expected_reference: Mapping[str, object]
    max_files: int
    max_file_bytes: int
    max_total_bytes: int
    max_runtime_seconds: float
    run_manifest_filename: str = "public_reference_run_manifest.json"
    submission_filename: str = "submission.csv"
    page_size: int = 20
    expected_input_shapes_tzyx: Mapping[str, Sequence[int]] | None = None
    expected_input_shapes_source: str | None = None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _strict_json_object(path: Path) -> dict[str, object]:
    def pairs(pairs_: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs_:
            if key in result:
                raise KaggleOutputError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise KaggleOutputError(f"nonfinite JSON value in {path.name}: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KaggleOutputError(f"cannot read strict JSON from {path.name}") from exc
    if not isinstance(value, dict):
        raise KaggleOutputError(f"{path.name} must contain a JSON object")
    return value


def _plain_json(value: object, field: str) -> object:
    """Normalize a caller contract and reject non-JSON/nonfinite values."""

    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise KaggleOutputError(f"{field} must be finite JSON") from exc


def _shape_contract(
    value: Mapping[str, Sequence[int]] | None,
) -> dict[str, list[int]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise KaggleOutputError("expected input shapes must be a nonempty mapping")
    result: dict[str, list[int]] = {}
    for dataset, shape in value.items():
        if not isinstance(dataset, str) or not dataset:
            raise KaggleOutputError("expected input shapes contain an invalid dataset ID")
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence) or len(shape) != 4:
            raise KaggleOutputError(f"expected shape for {dataset} must be [t,z,y,x]")
        dimensions = list(shape)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in dimensions
        ):
            raise KaggleOutputError(f"expected shape for {dataset} has invalid dimensions")
        result[dataset] = dimensions
    return dict(sorted(result.items()))


def _validate_spec(spec: ExactOutputSpec) -> tuple[dict[str, object], dict[str, list[int]] | None]:
    if not isinstance(spec.notebook_slug, str) or not _SLUG.fullmatch(spec.notebook_slug):
        raise KaggleOutputError("notebook slug must be exact owner/notebook")
    if isinstance(spec.notebook_version, bool) or not isinstance(spec.notebook_version, int):
        raise KaggleOutputError("notebook version must be a positive integer")
    if (
        spec.notebook_version <= 0
        or not isinstance(spec.release_digest, str)
        or not _SHA256.fullmatch(spec.release_digest)
    ):
        raise KaggleOutputError("notebook version/release digest is invalid")
    for filename, field in (
        (spec.run_manifest_filename, "run manifest filename"),
        (spec.submission_filename, "submission filename"),
    ):
        if (
            not isinstance(filename, str)
            or not filename
            or filename in {".", ".."}
            or any(character in filename for character in ("\\", ":", "\x00"))
            or PurePosixPath(filename).name != filename
        ):
            raise KaggleOutputError(f"{field} must be a basename")
    limits = (spec.max_files, spec.max_file_bytes, spec.max_total_bytes, spec.page_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in limits):
        raise KaggleOutputError("file, byte, and page limits must be positive integers")
    if spec.max_total_bytes < spec.max_file_bytes:
        raise KaggleOutputError("total byte limit cannot be smaller than the per-file limit")
    if (
        isinstance(spec.max_runtime_seconds, bool)
        or not isinstance(spec.max_runtime_seconds, (int, float))
        or not math.isfinite(spec.max_runtime_seconds)
        or spec.max_runtime_seconds <= 0
    ):
        raise KaggleOutputError("runtime limit must be finite and positive")
    reference = _plain_json(spec.expected_reference, "expected reference")
    if not isinstance(reference, dict) or not reference:
        raise KaggleOutputError("expected reference must be a nonempty JSON object")
    shapes = _shape_contract(spec.expected_input_shapes_tzyx)
    if shapes is None and spec.expected_input_shapes_source is not None:
        raise KaggleOutputError("shape source cannot be supplied without expected shapes")
    if shapes is not None and not spec.expected_input_shapes_source:
        raise KaggleOutputError("independently supplied shapes require a source label")
    return reference, shapes


# The helper executes inside the Python environment that owns the Kaggle SDK.
# It prints one sanitized JSON object.  Signed URLs exist only in its memory.
_SDK_HELPER = r"""
import hashlib, importlib.metadata, json, os, pathlib, sys, urllib.request
from urllib.parse import urlparse

def emit(value, code=0):
    sys.stdout.write(json.dumps(value, sort_keys=True, allow_nan=False))
    raise SystemExit(code)

try:
    cfg = json.load(sys.stdin)
    helper_deadline = cfg["helper_deadline_seconds"]
    if os.name == "nt":
        import threading, time
        def hard_deadline():
            time.sleep(helper_deadline)
            os._exit(124)
        threading.Thread(target=hard_deadline, daemon=True).start()
        deadline_enforcement = "WINDOWS_OS_EXIT_WATCHDOG"
    else:
        import signal
        def alarm_deadline(_signum, _frame):
            # A provider retry handler must not swallow the deadline exception.
            os._exit(124)
        signal.signal(signal.SIGALRM, alarm_deadline)
        signal.setitimer(signal.ITIMER_REAL, helper_deadline)
        deadline_enforcement = "POSIX_SETITIMER_OS_EXIT"
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    root = pathlib.Path(cfg["output_dir"])
    marker = root / ".path-binding"
    marker_bytes = marker.read_bytes()
    if hashlib.sha256(marker_bytes).hexdigest() != cfg["marker_sha256"]:
        raise RuntimeError("output path binding failed")
    owner, slug = cfg["notebook_slug"].split("/", 1)
    version_label = cfg["version_label"]
    client = api.build_kaggle_client()
    inventory, seen_names, seen_tokens, logs = [], set(), set(), []
    token = None
    pages = 0
    while True:
        pages += 1
        if pages > cfg["max_files"] + 1:
            raise RuntimeError("output pagination exceeded the bounded page count")
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.version_label = version_label
        request.page_size = cfg["page_size"]
        if token:
            request.page_token = token
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        log = response.log or ""
        log_bytes = log.encode("utf-8")
        if len(log_bytes) > 8 * 1024 * 1024:
            raise RuntimeError("kernel log exceeded the fixed 8 MiB evidence limit")
        if log_bytes:
            logs.append({"page": pages, "bytes": len(log_bytes),
                         "sha256": hashlib.sha256(log_bytes).hexdigest()})
        for item in response.files or []:
            name, url = item.file_name, item.url
            if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
                raise RuntimeError("unsafe output filename")
            pure = pathlib.PurePosixPath(name)
            parts = pure.parts
            if pure.is_absolute() or pure.as_posix() != name or not parts or any(p in ("", ".", "..") for p in parts):
                raise RuntimeError("unsafe output path")
            if name in seen_names:
                raise RuntimeError("duplicate output path")
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise RuntimeError("unsafe signed output URL")
            seen_names.add(name)
            inventory.append((name, url))
            if len(inventory) > cfg["max_files"]:
                raise RuntimeError("output file count exceeded limit before download")
        next_token = response.next_page_token or None
        if not next_token:
            break
        if not isinstance(next_token, str) or next_token in seen_tokens:
            raise RuntimeError("invalid or repeated output page token")
        seen_tokens.add(next_token)
        token = next_token
    if not inventory:
        raise RuntimeError("exact notebook version returned no output files")

    records, total = [], 0
    for name, url in inventory:
        target = root.joinpath(*pathlib.PurePosixPath(name).parts)
        if not target.resolve(strict=False).is_relative_to(root.resolve(strict=True)):
            raise RuntimeError("output path escaped destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        probe = target.parent
        while True:
            if probe.is_symlink():
                raise RuntimeError("symlink encountered in output path")
            if probe == root:
                break
            if not probe.is_relative_to(root):
                raise RuntimeError("output parent escaped destination")
            probe = probe.parent
        digest, size = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(url, timeout=min(60.0, cfg["max_runtime_seconds"])) as response:
                final = urlparse(response.geturl())
                if final.scheme != "https" or not final.netloc:
                    raise RuntimeError("signed output URL redirected outside HTTPS")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > cfg["max_file_bytes"]:
                    raise RuntimeError("output Content-Length exceeded per-file limit")
                with target.open("xb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        size += len(block)
                        total += len(block)
                        if size > cfg["max_file_bytes"] or total > cfg["max_total_bytes"]:
                            raise RuntimeError("download exceeded configured byte limit")
                        stream.write(block)
                        digest.update(block)
        except BaseException:
            try:
                target.unlink()
            except OSError:
                pass
            raise
        records.append({"path": name, "bytes": size, "sha256": digest.hexdigest()})
    emit({
        "schema_version": 1,
        "status": "PASS",
        "kaggle_package_version": importlib.metadata.version("kaggle"),
        "notebook_slug": cfg["notebook_slug"],
        "notebook_version": cfg["notebook_version"],
        "version_label": version_label,
        "provider_request_fields": {
            "user_name": owner, "kernel_slug": slug,
            "version_label": version_label, "page_size": cfg["page_size"],
        },
        "pages": pages,
        "files": records,
        "total_bytes": total,
        "log_observations": logs,
        "marker_sha256": cfg["marker_sha256"],
        "helper_deadline_seconds": helper_deadline,
        "deadline_enforcement": deadline_enforcement,
    })
except Exception as exc:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        status = None
    emit({"schema_version": 1, "status": "ERROR", "error_type": type(exc).__name__,
          "error_sha256": hashlib.sha256(repr(exc).encode()).hexdigest(),
          "http_status": status}, 2)
"""


def _sanitized_helper_result(raw: str) -> dict[str, object] | None:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return None
    if value.get("status") == "ERROR":
        result = {
            key: value.get(key)
            for key in ("schema_version", "status", "error_type", "error_sha256", "http_status")
        }
        if (
            not isinstance(result["error_type"], str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", result["error_type"])
            or not _SHA256.fullmatch(str(result["error_sha256"]))
        ):
            return None
        if result["http_status"] is not None and (
            isinstance(result["http_status"], bool) or not isinstance(result["http_status"], int)
        ):
            return None
        return result
    if value.get("status") != "PASS":
        return None
    # Project onto known fields before persistence.  A modified SDK/helper must
    # not smuggle a signed URL into an unknown response field.
    scalar_fields = (
        "schema_version",
        "status",
        "kaggle_package_version",
        "notebook_slug",
        "notebook_version",
        "version_label",
        "pages",
        "total_bytes",
        "marker_sha256",
        "helper_deadline_seconds",
        "deadline_enforcement",
    )
    projected = {key: value.get(key) for key in scalar_fields}
    request = value.get("provider_request_fields")
    files = value.get("files")
    logs = value.get("log_observations")
    if not isinstance(request, dict) or not isinstance(files, list) or not isinstance(logs, list):
        return None
    projected["provider_request_fields"] = {
        key: request.get(key) for key in ("user_name", "kernel_slug", "version_label", "page_size")
    }
    projected["files"] = [
        {key: row.get(key) for key in ("path", "bytes", "sha256")}
        for row in files
        if isinstance(row, dict)
    ]
    projected["log_observations"] = [
        {key: row.get(key) for key in ("page", "bytes", "sha256")}
        for row in logs
        if isinstance(row, dict)
    ]
    if len(projected["files"]) != len(files) or len(projected["log_observations"]) != len(logs):
        return None
    package_version = projected["kaggle_package_version"]
    if not isinstance(package_version, str) or not re.fullmatch(
        r"[A-Za-z0-9_.+!-]{1,64}", package_version
    ):
        return None
    integer_fields = ("notebook_version", "pages", "total_bytes")
    if any(
        isinstance(projected[field], bool)
        or not isinstance(projected[field], int)
        or projected[field] < (0 if field == "total_bytes" else 1)
        for field in integer_fields
    ):
        return None
    if not isinstance(projected["notebook_slug"], str) or not _SLUG.fullmatch(
        projected["notebook_slug"]
    ):
        return None
    if not isinstance(projected["version_label"], str) or not re.fullmatch(
        r"v[1-9][0-9]*", projected["version_label"]
    ):
        return None
    if not isinstance(projected["marker_sha256"], str) or not _SHA256.fullmatch(
        projected["marker_sha256"]
    ):
        return None
    if (
        isinstance(projected["helper_deadline_seconds"], bool)
        or not isinstance(projected["helper_deadline_seconds"], (int, float))
        or not math.isfinite(projected["helper_deadline_seconds"])
        or projected["helper_deadline_seconds"] <= 0
        or projected["deadline_enforcement"]
        not in {"POSIX_SETITIMER_OS_EXIT", "WINDOWS_OS_EXIT_WATCHDOG"}
    ):
        return None
    request_projected = projected["provider_request_fields"]
    if (
        not isinstance(request_projected["user_name"], str)
        or not re.fullmatch(r"[a-z0-9_-]+", request_projected["user_name"])
        or not isinstance(request_projected["kernel_slug"], str)
        or not re.fullmatch(r"[a-z0-9_-]+", request_projected["kernel_slug"])
        or not isinstance(request_projected["version_label"], str)
        or not re.fullmatch(r"v[1-9][0-9]*", request_projected["version_label"])
        or isinstance(request_projected["page_size"], bool)
        or not isinstance(request_projected["page_size"], int)
        or request_projected["page_size"] <= 0
    ):
        return None
    for row in projected["files"]:
        relative = row["path"]
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or not relative
            or "\\" in relative
            or ":" in relative
            or "\x00" in relative
            or pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in ("", ".", "..") for part in pure.parts)
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] < 0
            or not isinstance(row["sha256"], str)
            or not _SHA256.fullmatch(row["sha256"])
        ):
            return None
    for row in projected["log_observations"]:
        if (
            isinstance(row["page"], bool)
            or not isinstance(row["page"], int)
            or row["page"] <= 0
            or isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or row["bytes"] <= 0
            or not isinstance(row["sha256"], str)
            or not _SHA256.fullmatch(row["sha256"])
        ):
            return None
    return projected


def _verified_inventory(
    output_dir: Path,
    helper: Mapping[str, object],
    spec: ExactOutputSpec,
    marker_sha256: str,
) -> list[dict[str, object]]:
    expected_request = {
        "user_name": spec.notebook_slug.split("/", 1)[0],
        "kernel_slug": spec.notebook_slug.split("/", 1)[1],
        "version_label": f"v{spec.notebook_version}",
        "page_size": spec.page_size,
    }
    if any(
        (
            helper.get("notebook_slug") != spec.notebook_slug,
            helper.get("notebook_version") != spec.notebook_version,
            helper.get("version_label") != f"v{spec.notebook_version}",
            helper.get("provider_request_fields") != expected_request,
            helper.get("marker_sha256") != marker_sha256,
            helper.get("helper_deadline_seconds") != spec.max_runtime_seconds * 0.9,
        )
    ):
        raise KaggleOutputError("helper result is not bound to the requested notebook version/path")
    rows = helper.get("files")
    if not isinstance(rows, list) or not rows or len(rows) > spec.max_files:
        raise KaggleOutputError("helper returned an invalid file inventory")
    verified: list[dict[str, object]] = []
    seen: set[str] = set()
    total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise KaggleOutputError("helper file inventory has an invalid schema")
        relative, size, digest = row["path"], row["bytes"], row["sha256"]
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            not isinstance(relative, str)
            or "\\" in relative
            or ":" in relative
            or pure is None
            or pure.is_absolute()
            or pure.as_posix() != relative
        ):
            raise KaggleOutputError("helper returned an unsafe output path")
        parts = PurePosixPath(relative).parts
        if not parts or any(part in ("", ".", "..") for part in parts) or relative in seen:
            raise KaggleOutputError("helper returned a duplicate/unsafe output path")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise KaggleOutputError("helper returned an invalid output size")
        if (
            size > spec.max_file_bytes
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise KaggleOutputError("helper returned invalid output bounds/hash")
        path = output_dir.joinpath(*parts)
        try:
            inside = path.resolve(strict=True).is_relative_to(output_dir.resolve(strict=True))
        except OSError:
            inside = False
        if not inside:
            raise KaggleOutputError(f"downloaded output escaped destination: {relative}")
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise KaggleOutputError(f"downloaded output is missing or changed: {relative}")
        if _sha256_file(path) != digest:
            raise KaggleOutputError(f"downloaded output hash mismatch: {relative}")
        seen.add(relative)
        total += size
        verified.append(
            {
                "path": relative,
                "basename": PurePosixPath(relative).name,
                "bytes": size,
                "sha256": digest,
            }
        )
    if total > spec.max_total_bytes or helper.get("total_bytes") != total:
        raise KaggleOutputError("helper total byte count is invalid")
    actual = {
        path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()
    }
    if any(path.is_symlink() for path in output_dir.rglob("*")) or actual != seen:
        raise KaggleOutputError("download directory contains unverified entries")
    return sorted(verified, key=lambda row: str(row["path"]))


def _one_basename(
    inventory: Sequence[Mapping[str, object]], output_dir: Path, basename: str
) -> tuple[Path, Mapping[str, object]]:
    matches = [row for row in inventory if row.get("basename") == basename]
    if len(matches) != 1:
        raise KaggleOutputError(f"exactly one {basename} output is required")
    relative = str(matches[0]["path"])
    return output_dir.joinpath(*PurePosixPath(relative).parts), matches[0]


def _build_proof(
    spec: ExactOutputSpec,
    reference: Mapping[str, object],
    expected_shapes: dict[str, list[int]] | None,
    inventory: list[dict[str, object]],
    output_dir: Path,
    receipt_path: Path,
) -> dict[str, object]:
    manifest_path, manifest_row = _one_basename(inventory, output_dir, spec.run_manifest_filename)
    submission_path, submission_row = _one_basename(inventory, output_dir, spec.submission_filename)
    manifest = _strict_json_object(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "COMPLETE":
        raise KaggleOutputError("run manifest is not a complete schema-1 artifact")
    if manifest.get("release_digest") != spec.release_digest:
        raise KaggleOutputError("run manifest release digest does not match the requested release")
    if manifest.get("reference") != reference:
        raise KaggleOutputError("run manifest reference differs from the reviewed reference")
    submission = manifest.get("submission")
    if not isinstance(submission, dict):
        raise KaggleOutputError("run manifest submission identity is missing")
    submission_hash = submission_row["sha256"]
    if (
        manifest.get("csv_sha256") != submission_hash
        or submission.get("sha256") != submission_hash
        or PurePosixPath(str(submission.get("path", "")).replace("\\", "/")).name
        != spec.submission_filename
    ):
        raise KaggleOutputError("run manifest does not bind the downloaded submission bytes")
    actual_shapes = manifest.get("actual_input_shapes_tzyx")
    dataset_shapes = manifest.get("dataset_shapes")
    if expected_shapes is None:
        shape_status = "UNAVAILABLE"
    else:
        if actual_shapes != expected_shapes or dataset_shapes != expected_shapes:
            raise KaggleOutputError(
                "run manifest shapes differ from the independent shape contract"
            )
        shape_status = "VERIFIED"
    return {
        "schema_version": 1,
        "kind": "KAGGLE_EXACT_VERSION_OUTPUT_PROOF",
        "status": "PASS",
        "transport_status": "VERIFIED",
        "notebook_slug": spec.notebook_slug,
        "notebook_version": spec.notebook_version,
        "version_label": f"v{spec.notebook_version}",
        "release_digest": spec.release_digest,
        "run_manifest_sha256": manifest_row["sha256"],
        "submission_sha256": submission_hash,
        "embedded_release_identity_verified": True,
        "embedded_reference_verified": True,
        "embedded_submission_hash_verified": True,
        "output_files": inventory,
        "download_directory": str(output_dir),
        "run_manifest_path": str(manifest_path),
        "submission_path": str(submission_path),
        "transport_receipt_path": str(receipt_path),
        "transport_receipt_sha256": _sha256_file(receipt_path),
        "expected_input_shapes_tzyx": expected_shapes,
        "expected_input_shapes_source": spec.expected_input_shapes_source,
        "expected_input_shapes_sha256": (
            _sha256_bytes(canonical_json(expected_shapes).encode()) if expected_shapes else None
        ),
        "shape_identity_status": shape_status,
        "release_validation_status": "UNRESOLVED_PENDING_INDEPENDENT_VALIDATOR",
        "verification_scope": (
            "Exact Kaggle version request, downloaded bytes, embedded release/reference identity, "
            "and submission hash binding only. This is not operational or quality approval."
        ),
    }


class KaggleExactOutputClient:
    """Run the exact-version SDK helper through a literal Python argv prefix."""

    def __init__(
        self,
        argv_prefix: Sequence[str],
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        path_adapter: Callable[[Path], str] = os.fspath,
    ) -> None:
        if not argv_prefix or any(
            not isinstance(part, str) or not part or "\x00" in part for part in argv_prefix
        ):
            raise KaggleOutputError("a verified literal Kaggle-Python argv prefix is required")
        self.argv_prefix = list(argv_prefix)
        self.runner = runner
        self.path_adapter = path_adapter

    def retrieve_and_verify_outputs(
        self, spec: ExactOutputSpec, destination: Path
    ) -> dict[str, object]:
        reference, expected_shapes = _validate_spec(spec)
        destination = Path(destination)
        try:
            original_parent = destination.parent
            if original_parent.is_symlink():
                raise KaggleOutputError("destination parent cannot be a symlink")
            parent = original_parent.resolve(strict=True)
        except OSError as exc:
            raise KaggleOutputError("destination parent is unavailable") from exc
        if parent.is_symlink() or not parent.is_dir() or destination.exists():
            raise KaggleOutputError("destination must be a new directory under a regular parent")
        destination.mkdir(mode=0o700)
        output_dir = destination / DOWNLOAD_DIRECTORY
        output_dir.mkdir(mode=0o700)
        marker = output_dir / ".path-binding"
        marker_bytes = secrets.token_bytes(32)
        marker.write_bytes(marker_bytes)
        marker_sha256 = _sha256_bytes(marker_bytes)
        receipt_path = destination / TRANSPORT_RECEIPT_FILENAME
        started_at = datetime.now(UTC).isoformat()
        config = {
            "notebook_slug": spec.notebook_slug,
            "notebook_version": spec.notebook_version,
            "version_label": f"v{spec.notebook_version}",
            "output_dir": self.path_adapter(output_dir.resolve()),
            "marker_sha256": marker_sha256,
            "max_files": spec.max_files,
            "max_file_bytes": spec.max_file_bytes,
            "max_total_bytes": spec.max_total_bytes,
            "max_runtime_seconds": spec.max_runtime_seconds,
            # Finish before the parent timeout. If the WSL launcher is killed,
            # an already-started Linux child still owns this local deadline.
            "helper_deadline_seconds": spec.max_runtime_seconds * 0.9,
            "page_size": spec.page_size,
        }
        result: subprocess.CompletedProcess[str] | None = None
        process_error: BaseException | None = None
        try:
            result = self.runner(
                [*self.argv_prefix, "-c", _SDK_HELPER],
                input=json.dumps(config, sort_keys=True),
                capture_output=True,
                text=True,
                timeout=spec.max_runtime_seconds,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            process_error = exc
        stdout = result.stdout if result is not None and isinstance(result.stdout, str) else ""
        stderr = result.stderr if result is not None and isinstance(result.stderr, str) else ""
        helper = _sanitized_helper_result(stdout)
        marker_ok = marker.is_file() and _sha256_file(marker) == marker_sha256
        try:
            marker.unlink()
        except OSError:
            marker_ok = False
        receipt: dict[str, object] = {
            "schema_version": 1,
            "kind": "KAGGLE_EXACT_VERSION_TRANSPORT_RECEIPT",
            "status": "PASS"
            if (
                result is not None
                and result.returncode == 0
                and helper is not None
                and helper.get("status") == "PASS"
                and marker_ok
            )
            else "ERROR",
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "notebook_slug": spec.notebook_slug,
            "notebook_version": spec.notebook_version,
            "version_label": f"v{spec.notebook_version}",
            "provider_request_fields": {
                "user_name": spec.notebook_slug.split("/", 1)[0],
                "kernel_slug": spec.notebook_slug.split("/", 1)[1],
                "version_label": f"v{spec.notebook_version}",
                "page_size": spec.page_size,
            },
            "parent_timeout_seconds": spec.max_runtime_seconds,
            "helper_deadline_seconds": config["helper_deadline_seconds"],
            "deadline_scope": (
                "The helper deadline starts immediately after its stdin configuration is parsed."
            ),
            "sdk_helper_sha256": _sha256_bytes(_SDK_HELPER.encode()),
            "argv_prefix_sha256": _sha256_bytes(canonical_json(self.argv_prefix).encode()),
            "stdout_sha256": _sha256_bytes(stdout.encode()),
            "stderr_sha256": _sha256_bytes(stderr.encode()),
            "returncode": result.returncode if result is not None else None,
            "process_error_type": type(process_error).__name__ if process_error else None,
            "path_binding_verified": marker_ok,
            "helper_result": helper,
            "raw_subprocess_output_persisted": False,
            "signed_urls_persisted": False,
        }
        _write_json(receipt_path, receipt)
        if receipt["status"] != "PASS" or helper is None:
            raise KaggleOutputError(f"exact-version transport failed; receipt: {receipt_path}")
        try:
            inventory = _verified_inventory(output_dir, helper, spec, marker_sha256)
            proof = _build_proof(
                spec, reference, expected_shapes, inventory, output_dir, receipt_path
            )
            proof_path = destination / EXACT_VERSION_PROOF_FILENAME
            _write_json(proof_path, proof)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise KaggleOutputError(
                f"downloaded bytes failed exact-version proof; receipt: {receipt_path}"
            ) from exc
        return {
            "proof": proof,
            "proof_path": str(proof_path),
            "proof_sha256": _sha256_file(proof_path),
            "transport_receipt_path": str(receipt_path),
            "download_directory": str(output_dir),
        }


def retrieve_and_verify_outputs(
    client: KaggleExactOutputClient, spec: ExactOutputSpec, destination: Path
) -> dict[str, object]:
    """Functional entry point for exact-version transport and proof."""

    return client.retrieve_and_verify_outputs(spec, destination)


__all__ = [
    "DOWNLOAD_DIRECTORY",
    "EXACT_VERSION_PROOF_FILENAME",
    "TRANSPORT_RECEIPT_FILENAME",
    "ExactOutputSpec",
    "KaggleExactOutputClient",
    "KaggleOutputError",
    "retrieve_and_verify_outputs",
]
