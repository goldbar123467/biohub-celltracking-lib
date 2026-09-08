#!/usr/bin/env python3
"""Read-only operator for exact-version E0 R3 output and release validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.campaign.kaggle_outputs import (
    ExactOutputSpec,
    KaggleExactOutputClient,
)
from biohub_ct.campaign.kaggle_rehearsal import (
    PackageIdentity,
    WSLPathVerifier,
    preflight_package,
)
from biohub_ct.campaign.rehearsal_packages import (
    DEFAULT_PACKAGE_DIRS,
    REHEARSAL_PACKAGES,
    reviewed_package,
)
from biohub_ct.campaign.release_validation import (
    validate_downloaded_e0_release,
)


class OperatorInputError(ValueError):
    """The requested operator action is ambiguous, unresolved, or unsafe."""


RESULT_FILENAME = "validation-result.json"
_TEMP_FILENAME = ".validation-result.json.partial"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UNRESOLVED = re.compile(
    r"(?:^|[^a-z])(todo|tbd|unknown|unresolved|placeholder)(?:[^a-z]|$)", re.IGNORECASE
)
_DOWNLOAD_OPTIONS = {
    "notebook_version": "--notebook-version",
    "input_shapes_json": "--input-shapes-json",
    "input_shapes_source_receipt": "--input-shapes-source-receipt",
    "max_files": "--max-files",
    "max_file_bytes": "--max-file-bytes",
    "max_total_bytes": "--max-total-bytes",
    "max_runtime_seconds": "--max-runtime-seconds",
    "page_size": "--page-size",
    "wsl_distro": "--wsl-distro",
    "kaggle_sdk_python": "--kaggle-sdk-python",
    "destination_sdk_path": "--destination-sdk-path",
}
_LOCAL_OPTIONS = {
    "proof_json": "--proof-json",
    "submission_csv": "--submission-csv",
    "run_manifest_json": "--run-manifest-json",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _strict_json_object(path: Path, label: str) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OperatorInputError(f"cannot read {label}") from exc

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OperatorInputError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise OperatorInputError(f"{label} contains nonfinite JSON value {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorInputError(f"cannot parse strict JSON from {label}") from exc
    if not isinstance(value, dict):
        raise OperatorInputError(f"{label} must contain a JSON object")
    return value, _sha256_bytes(raw)


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise OperatorInputError(f"{label} must be an existing regular file")
    return path.resolve(strict=True)


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise OperatorInputError(f"{label} must be an existing regular directory")
    return path.resolve(strict=True)


def _require_within(path: Path, root: Path, label: str) -> Path:
    resolved = _regular_file(path, label)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise OperatorInputError(f"{label} escaped the operator transport directory")
    return resolved


def _prepare_destination(path: Path, package_dir: Path) -> Path:
    if path.is_symlink() or path.exists():
        raise OperatorInputError("--destination must be a new path")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise OperatorInputError("--destination parent must be an existing regular directory")
    resolved = path.resolve(strict=False)
    if resolved == package_dir or resolved.is_relative_to(package_dir):
        raise OperatorInputError("--destination cannot modify the immutable reviewed package")
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise OperatorInputError("cannot create exclusive --destination") from exc
    return path.resolve(strict=True)


def _atomic_result(destination: Path, value: Mapping[str, object]) -> Path:
    """Install complete JSON without replacing an existing result or partial."""

    result = destination / RESULT_FILENAME
    temporary = destination / _TEMP_FILENAME
    if result.exists() or result.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise OperatorInputError("result or deterministic temporary path already exists")
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # link() is an atomic no-replace install on the supported local filesystems.
        os.link(temporary, result)
        temporary.unlink()
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return result


def _validate_mode_options(args: argparse.Namespace) -> None:
    if args.download:
        conflicts = [
            flag for name, flag in _LOCAL_OPTIONS.items() if getattr(args, name) is not None
        ]
        missing = [flag for name, flag in _DOWNLOAD_OPTIONS.items() if getattr(args, name) is None]
        if conflicts:
            raise OperatorInputError(
                "--download cannot be combined with local evidence options: " + ", ".join(conflicts)
            )
        if missing:
            raise OperatorInputError("--download requires explicit options: " + ", ".join(missing))
        if args.max_total_bytes < args.max_file_bytes:
            raise OperatorInputError("--max-total-bytes cannot be smaller than --max-file-bytes")
        if args.page_size > args.max_files:
            raise OperatorInputError("--page-size cannot exceed --max-files")
    else:
        supplied = [
            flag for name, flag in _DOWNLOAD_OPTIONS.items() if getattr(args, name) is not None
        ]
        missing = [flag for name, flag in _LOCAL_OPTIONS.items() if getattr(args, name) is None]
        if supplied:
            raise OperatorInputError(
                "download options require explicit --download: " + ", ".join(supplied)
            )
        if missing:
            raise OperatorInputError("local validation requires: " + ", ".join(missing))


def _validate_wsl_values(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.wsl_distro):
        raise OperatorInputError("--wsl-distro is invalid")
    for value, label in (
        (args.kaggle_sdk_python, "--kaggle-sdk-python"),
        (args.destination_sdk_path, "--destination-sdk-path"),
    ):
        if not isinstance(value, str) or "\x00" in value or _UNRESOLVED.search(value):
            raise OperatorInputError(f"{label} is unresolved")
        pure = PurePosixPath(value)
        if (
            not pure.is_absolute()
            or pure.as_posix() != value
            or any(part in ("", ".", "..") for part in pure.parts)
        ):
            raise OperatorInputError(f"{label} must be one canonical absolute POSIX path")


def _load_shape_contract(
    path: Path, source_receipt_path: Path
) -> tuple[dict[str, list[int]], str, str]:
    value, digest = _strict_json_object(path, "independent input shapes JSON")
    required = {
        "schema_version",
        "kind",
        "status",
        "source",
        "source_receipt_sha256",
        "shapes_tzyx",
    }
    if set(value) != required:
        raise OperatorInputError("independent input shapes JSON has the wrong exact schema")
    if (
        value["schema_version"] != 1
        or value["kind"] != "BIOHUB_INDEPENDENT_INPUT_SHAPES"
        or value["status"] != "VERIFIED"
    ):
        raise OperatorInputError("independent input shapes JSON is not verified schema version 1")
    source = value["source"]
    source_digest = value["source_receipt_sha256"]
    _, actual_source_digest = _strict_json_object(
        source_receipt_path, "independent shape source receipt"
    )
    if (
        not isinstance(source, str)
        or not source.strip()
        or _UNRESOLVED.search(source)
        or not isinstance(source_digest, str)
        or not _SHA256.fullmatch(source_digest)
        or source_digest != actual_source_digest
    ):
        raise OperatorInputError("independent shape provenance is unresolved")
    raw_shapes = value["shapes_tzyx"]
    if not isinstance(raw_shapes, Mapping) or not raw_shapes:
        raise OperatorInputError("shapes_tzyx must be a nonempty object")
    shapes: dict[str, list[int]] = {}
    for dataset, raw_shape in raw_shapes.items():
        if (
            not isinstance(dataset, str)
            or not dataset
            or not isinstance(raw_shape, list)
            or len(raw_shape) != 4
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
                for dimension in raw_shape
            )
        ):
            raise OperatorInputError(f"invalid positive TZYX shape for dataset {dataset!r}")
        shapes[dataset] = list(raw_shape)
    provenance = (
        f"{source.strip()}; source_receipt_sha256={source_digest}; shapes_json_sha256={digest}"
    )
    return dict(sorted(shapes.items())), provenance, digest


def _result_dict(value: object) -> dict[str, object]:
    method = getattr(value, "to_dict", None)
    result = method() if callable(method) else value
    if not isinstance(result, Mapping):
        raise TypeError("release validator returned a non-mapping result")
    # Round-trip now so no late serialization surprise can leave a partial result.
    normalized = json.loads(json.dumps(dict(result), allow_nan=False))
    if not isinstance(normalized, dict):
        raise TypeError("release validator result is not a JSON object")
    return normalized


def _base_result(
    *,
    mode: str,
    package_dir: Path,
    generated_at: datetime,
    generation: str,
    package_identity: PackageIdentity,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "KAGGLE_REFERENCE_OUTPUT_OPERATOR_RESULT",
        "mode": mode,
        "generated_at": generated_at.isoformat(),
        "reviewed_package_generation": f"E0_{generation.upper()}",
        "package_dir": str(package_dir),
        "package_identity": json.loads(json.dumps(asdict(package_identity), allow_nan=False)),
        "read_only_provider_operation": True,
        "mutation_performed": False,
        "submission_performed": False,
        "approval_or_campaign_store_written": False,
    }


def execute(
    args: argparse.Namespace,
    *,
    transport_factory: Callable[..., Any] = KaggleExactOutputClient,
    path_verifier_factory: Callable[[str], Callable[[Path], str]] = WSLPathVerifier,
    release_validator: Callable[..., Any] = validate_downloaded_e0_release,
    package_preflight: Callable[..., Mapping[str, object]] = preflight_package,
    now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Execute local validation or an explicitly configured exact-version download."""

    package_identity = reviewed_package(args.generation)
    _validate_mode_options(args)
    package_dir = _regular_directory(args.package_dir, "--package-dir")
    canonicalization_path = None
    if args.canonicalization_receipt is not None:
        canonicalization_path = _regular_file(
            args.canonicalization_receipt, "--canonicalization-receipt"
        )
    if args.download:
        _validate_wsl_values(args)
    else:
        _regular_file(args.proof_json, "--proof-json")
        _regular_file(args.submission_csv, "--submission-csv")
        _regular_file(args.run_manifest_json, "--run-manifest-json")
    destination = _prepare_destination(args.destination, package_dir)
    mode = "EXACT_VERSION_DOWNLOAD_AND_VALIDATE" if args.download else "LOCAL_VALIDATE"
    base = _base_result(
        mode=mode,
        package_dir=package_dir,
        generated_at=now_factory(),
        generation=args.generation,
        package_identity=package_identity,
    )
    stage = "INPUT_VALIDATION"
    try:
        notebook_slug = package_identity.notebook_slug
        canonicalization_hash = None
        canonicalized_version = None
        if canonicalization_path is not None:
            from biohub_ct.campaign.kaggle_canonicalization import verify_canonicalization_receipt

            stage = "PROVIDER_CANONICALIZATION_PROOF"
            mapping = verify_canonicalization_receipt(
                canonicalization_path, identity=package_identity
            )
            notebook_slug = mapping.actual_notebook_slug
            canonicalized_version = mapping.notebook_version
            canonicalization_hash = mapping.receipt_sha256
            if args.download and args.notebook_version != canonicalized_version:
                raise OperatorInputError("requested version differs from canonicalization proof")
            base["provider_canonicalization"] = {
                "receipt_path": str(canonicalization_path),
                "receipt_sha256": canonicalization_hash,
                "requested_notebook_slug": package_identity.notebook_slug,
                "actual_notebook_slug": notebook_slug,
                "notebook_version": canonicalized_version,
            }
        transport_summary: dict[str, object] | None = None
        if args.download:
            stage = "REVIEWED_PACKAGE_PREFLIGHT"
            package_preflight(package_dir, package_identity)
            lock, lock_hash = _strict_json_object(
                package_dir / "artifact-lock.json", "reviewed artifact lock"
            )
            if (
                lock_hash != package_identity.artifact_lock_sha256
                or lock.get("release_digest") != package_identity.release_digest
            ):
                raise OperatorInputError("reviewed artifact lock changed after preflight")
            reference = lock.get("reference")
            if not isinstance(reference, Mapping) or not reference:
                raise OperatorInputError("reviewed artifact lock has no reference object")
            stage = "INDEPENDENT_SHAPE_EVIDENCE"
            shapes_path = _regular_file(args.input_shapes_json, "--input-shapes-json")
            shape_source_receipt = _regular_file(
                args.input_shapes_source_receipt, "--input-shapes-source-receipt"
            )
            shapes, shape_source, shapes_hash = _load_shape_contract(
                shapes_path, shape_source_receipt
            )

            stage = "WSL_PATH_BINDING"
            path_verifier = path_verifier_factory(args.wsl_distro)
            mapped_destination = path_verifier(destination)
            if mapped_destination != args.destination_sdk_path:
                raise OperatorInputError(
                    "verified WSL destination mapping differs from --destination-sdk-path"
                )

            def checked_path_adapter(local_path: Path) -> str:
                mapped = path_verifier(local_path)
                relative = local_path.resolve(strict=True).relative_to(destination)
                expected = (
                    PurePosixPath(args.destination_sdk_path).joinpath(*relative.parts).as_posix()
                )
                if mapped != expected:
                    raise OperatorInputError(
                        "download child path differs from explicit WSL mapping"
                    )
                return mapped

            stage = "EXACT_VERSION_DOWNLOAD"
            client = transport_factory(
                [
                    "wsl",
                    "-d",
                    args.wsl_distro,
                    "--",
                    args.kaggle_sdk_python,
                ],
                path_adapter=checked_path_adapter,
            )
            spec = ExactOutputSpec(
                notebook_slug=notebook_slug,
                notebook_version=args.notebook_version,
                release_digest=package_identity.release_digest,
                expected_reference=reference,
                max_files=args.max_files,
                max_file_bytes=args.max_file_bytes,
                max_total_bytes=args.max_total_bytes,
                max_runtime_seconds=args.max_runtime_seconds,
                page_size=args.page_size,
                expected_input_shapes_tzyx=shapes,
                expected_input_shapes_source=shape_source,
            )
            transport = client.retrieve_and_verify_outputs(spec, destination / "transport")
            proof = transport.get("proof")
            if not isinstance(proof, Mapping):
                raise RuntimeError("exact-version transport returned no proof mapping")
            transport_root = (destination / "transport").resolve(strict=True)
            download_root = (transport_root / "downloaded").resolve(strict=True)
            proof_path = _require_within(
                Path(str(transport.get("proof_path"))), transport_root, "transport proof"
            )
            persisted_proof, persisted_proof_hash = _strict_json_object(
                proof_path, "transport proof"
            )
            if persisted_proof != proof or transport.get("proof_sha256") != persisted_proof_hash:
                raise RuntimeError("transport proof mapping differs from its persisted bytes")
            proof = persisted_proof
            submission_csv = _require_within(
                Path(str(proof.get("submission_path"))), download_root, "downloaded submission"
            )
            run_manifest_json = _require_within(
                Path(str(proof.get("run_manifest_path"))),
                download_root,
                "downloaded run manifest",
            )
            receipt_path = _require_within(
                Path(str(transport.get("transport_receipt_path"))),
                transport_root,
                "transport receipt",
            )
            transport_summary = {
                "notebook_slug": notebook_slug,
                "notebook_version": args.notebook_version,
                "proof_path": str(proof_path),
                "proof_sha256": persisted_proof_hash,
                "transport_receipt_path": str(receipt_path),
                "transport_receipt_sha256": _sha256_file(receipt_path),
                "input_shapes_json": str(shapes_path),
                "input_shapes_json_sha256": shapes_hash,
                "input_shapes_source_receipt": str(shape_source_receipt),
                "input_shapes_source_receipt_sha256": _sha256_file(shape_source_receipt),
                "artifact_lock_sha256": lock_hash,
                "explicit_bounds": {
                    "max_files": args.max_files,
                    "max_file_bytes": args.max_file_bytes,
                    "max_total_bytes": args.max_total_bytes,
                    "max_runtime_seconds": args.max_runtime_seconds,
                    "page_size": args.page_size,
                },
                "wsl_distro": args.wsl_distro,
                "kaggle_sdk_python": args.kaggle_sdk_python,
                "destination_sdk_path": args.destination_sdk_path,
            }
        else:
            stage = "LOCAL_EVIDENCE_READ"
            proof_path = _regular_file(args.proof_json, "--proof-json")
            submission_csv = _regular_file(args.submission_csv, "--submission-csv")
            run_manifest_json = _regular_file(args.run_manifest_json, "--run-manifest-json")
            proof, proof_hash = _strict_json_object(proof_path, "exact-version proof")

        stage = "INDEPENDENT_RELEASE_VALIDATION"
        if (
            canonicalized_version is not None
            and proof.get("notebook_version") != canonicalized_version
        ):
            raise OperatorInputError("output version differs from canonicalization proof")
        proof_hash_before = persisted_proof_hash if args.download else proof_hash
        submission_hash_before = _sha256_file(submission_csv)
        manifest_hash_before = _sha256_file(run_manifest_json)
        additional_validation_inputs = (
            {"canonicalization_receipt": canonicalization_path}
            if canonicalization_path is not None
            else {}
        )
        release = _result_dict(
            release_validator(
                submission_csv,
                run_manifest_json,
                package_dir,
                exact_version_proof=proof,
                package_identity=package_identity,
                **additional_validation_inputs,
            )
        )
        stage = "FINAL_IDENTITY_RECHECK"
        package_preflight(package_dir, package_identity)
        if canonicalization_path is not None:
            mapping_after = verify_canonicalization_receipt(
                canonicalization_path, identity=package_identity
            )
            if mapping_after.receipt_sha256 != canonicalization_hash:
                raise RuntimeError("canonicalization evidence changed during validation")
        proof_hash_after = _sha256_file(proof_path)
        submission_hash_after = _sha256_file(submission_csv)
        manifest_hash_after = _sha256_file(run_manifest_json)
        expected_release_fields = {
            "status": "STRUCTURAL_AND_IDENTITY_PASS_ADMISSION_BLOCKED",
            "identity_status": "PASS",
            "release_digest": package_identity.release_digest,
            "notebook_slug": notebook_slug,
            "notebook_version": proof.get("notebook_version"),
            "submission_sha256": submission_hash_after,
            "run_manifest_sha256": manifest_hash_after,
            "package_manifest_sha256": package_identity.package_manifest_sha256,
            "artifact_lock_sha256": package_identity.artifact_lock_sha256,
            "official_format_status": "PASS",
            "scorer_compatibility_status": "PASS",
            "e0_lineage_contract_status": "PASS",
        }
        if any(release.get(key) != expected for key, expected in expected_release_fields.items()):
            raise RuntimeError(
                "release validator result is not bound to the final input identities"
            )
        if (
            proof_hash_after != proof_hash_before
            or submission_hash_after != submission_hash_before
            or manifest_hash_after != manifest_hash_before
        ):
            raise RuntimeError("proof, submission, or run manifest changed during validation")
        result = {
            **base,
            "status": "VALIDATION_COMPLETE",
            "inputs": {
                "proof_json": str(proof_path),
                "proof_sha256": proof_hash_after,
                "submission_csv": str(submission_csv),
                "submission_sha256": submission_hash_after,
                "run_manifest_json": str(run_manifest_json),
                "run_manifest_sha256": manifest_hash_after,
            },
            "transport": transport_summary,
            "release_validation": release,
            "operator_conclusion": (
                "Validation completed. Quality/resource admission remains exactly as reported by "
                "the independent validator; this result is not submission approval."
            ),
        }
    except Exception as exc:  # noqa: BLE001 - sanitize every operator-boundary failure.
        result = {
            **base,
            "status": "ERROR",
            "failure_stage": stage,
            "error_type": type(exc).__name__,
            "error_sha256": _sha256_bytes(repr(exc).encode()),
            "error": "Operator failed closed; inspect the named input or nested transport receipt.",
        }
    result_path = _atomic_result(destination, result)
    return {
        "result": result,
        "result_path": str(result_path),
        "result_sha256": _sha256_file(result_path),
    }


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0 or str(result) != value:
        raise argparse.ArgumentTypeError("must be a canonical positive integer")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--generation", choices=tuple(REHEARSAL_PACKAGES), default="r3")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--proof-json", type=Path)
    parser.add_argument("--submission-csv", type=Path)
    parser.add_argument("--run-manifest-json", type=Path)
    parser.add_argument("--canonicalization-receipt", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--notebook-version", type=_positive_int)
    parser.add_argument("--input-shapes-json", type=Path)
    parser.add_argument("--input-shapes-source-receipt", type=Path)
    parser.add_argument("--max-files", type=_positive_int)
    parser.add_argument("--max-file-bytes", type=_positive_int)
    parser.add_argument("--max-total-bytes", type=_positive_int)
    parser.add_argument("--max-runtime-seconds", type=_positive_float)
    parser.add_argument("--page-size", type=_positive_int)
    parser.add_argument("--wsl-distro")
    parser.add_argument("--kaggle-sdk-python")
    parser.add_argument("--destination-sdk-path")
    args = parser.parse_args(argv)
    if args.package_dir is None:
        args.package_dir = ROOT / DEFAULT_PACKAGE_DIRS[args.generation]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output = execute(parse_args(argv))
    except (OSError, OperatorInputError) as exc:
        print(
            json.dumps(
                {"status": "INPUT_REJECTED", "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    except Exception as exc:  # noqa: BLE001 - do not expose arbitrary subprocess/provider text.
        print(
            json.dumps(
                {
                    "status": "OPERATOR_FAILED",
                    "error_type": type(exc).__name__,
                    "error_sha256": _sha256_bytes(repr(exc).encode()),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    summary = {
        "status": output["result"]["status"],
        "result_path": output["result_path"],
        "result_sha256": output["result_sha256"],
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if output["result"]["status"] == "VALIDATION_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
