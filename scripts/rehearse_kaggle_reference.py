"""Preflight E0 R3 by default; launch exactly once only from explicit live inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.campaign.kaggle_rehearsal import (
    KaggleRehearsalCLI,
    QuotaContract,
    RehearsalError,
    WSLPathVerifier,
    authorize_and_launch_rehearsal,
    preflight_package,
)
from biohub_ct.campaign.rehearsal_packages import (
    DEFAULT_PACKAGE_DIRS,
    REHEARSAL_PACKAGES,
    reviewed_package,
)
from biohub_ct.campaign.state import CampaignStore
from biohub_ct.campaign.watchdog import atomic_json

LAUNCH_REQUIRED_OPTIONS = {
    "store": "--store",
    "run_id": "--run-id",
    "decision_id": "--decision-id",
    "intent_id": "--intent-id",
    "request_id": "--request-id",
    "reservation_id": "--reservation-id",
    "reviewer": "--reviewer",
    "quota_observation_json": "--quota-observation-json",
    "quota_ledger_id": "--quota-ledger-id",
    "measured_quota_rate": "--measured-quota-rate",
    "notebook_timeout_seconds": "--notebook-timeout-seconds",
    "final_runtime_hours": "--final-runtime-hours",
    "platform_runtime_limit_hours": "--platform-runtime-limit-hours",
    "package_cli_path": "--package-cli-path",
    "wsl_distro": "--wsl-distro",
    "kaggle_cli_path": "--kaggle-cli-path",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _strict_json_object(path: Path) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON value in {path.name}: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return value


def _missing_launch_options(args: argparse.Namespace) -> list[str]:
    missing = [
        option
        for attribute, option in LAUNCH_REQUIRED_OPTIONS.items()
        if getattr(args, attribute) is None
    ]
    if args.documented_quota_rate_upper_bound is not None:
        if "--measured-quota-rate" in missing:
            missing.remove("--measured-quota-rate")
        if not args.quota_rate_source or not args.quota_rate_source.strip():
            missing.append("--quota-rate-source")
    return missing


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, receipt)


def _safe_receipt_path(args: argparse.Namespace) -> Path:
    """Reject output aliases that could corrupt immutable or admission-bearing inputs."""
    receipt = args.receipt_json.resolve()
    package_root = args.package_dir.resolve()
    if receipt == package_root or receipt.is_relative_to(package_root):
        raise ValueError("--receipt-json must be outside the reviewed package directory")
    protected = {
        label: value.resolve()
        for label, value in (
            ("campaign store", args.store),
            ("quota observation", args.quota_observation_json),
        )
        if value is not None
    }
    write_targets = {receipt, receipt.with_name(receipt.name + ".partial")}
    for label, path in protected.items():
        if path in write_targets:
            raise ValueError(f"--receipt-json cannot overwrite the canonical {label}")
    return receipt


def execute(
    args: argparse.Namespace,
    *,
    store_factory: Callable[[Path], Any] = CampaignStore,
    client_factory: Callable[[Sequence[str]], Any] = KaggleRehearsalCLI,
    verifier_factory: Callable[[str], Any] = WSLPathVerifier,
    launcher: Callable[..., Mapping[str, object]] = authorize_and_launch_rehearsal,
    package_preflight: Callable[[Path, Any], Mapping[str, object]] = preflight_package,
    now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Execute the operator flow without inventing live admission inputs."""

    package_identity = reviewed_package(args.generation)
    receipt_path = _safe_receipt_path(args)
    generated_at = now_factory()
    identity = json.loads(json.dumps(asdict(package_identity), allow_nan=False))
    base: dict[str, object] = {
        "schema_version": 1,
        "kind": f"e0_{args.generation}_kaggle_rehearsal_operator_receipt",
        "generated_at": generated_at.isoformat(),
        "launch_requested": bool(args.launch),
        "package_identity": identity,
        "admission_explanation": {
            "package_preflight_is_not_launch_admission": True,
            "actual_launch_requires": list(LAUNCH_REQUIRED_OPTIONS.values()),
            "stored_private_rehearsal_authorization_checked_by_library": True,
            "quota_rate_alternative": {
                "documented_bound": "--documented-quota-rate-upper-bound",
                "requires_source": "--quota-rate-source",
                "documented_bound_is_not_measured": True,
            },
            "live_quota_and_zero_active_gpu_jobs_checked_by_rehearsal_library": True,
            "single_push_requires_persisted_and_dispatched_intent": True,
            "unknown_outcome_requires_reconciliation_before_retry": True,
        },
    }
    try:
        package_receipt = package_preflight(args.package_dir, package_identity)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        failed = {
            **base,
            "status": "PREFLIGHT_FAILED",
            "launch_readiness": "NOT_READY",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_receipt(receipt_path, failed)
        return failed

    wsl_mapping: dict[str, object]
    if args.wsl_distro is not None and args.package_cli_path is not None:
        try:
            mapped_path = verifier_factory(args.wsl_distro)(args.package_dir)
        except (OSError, RehearsalError) as exc:
            failed = {
                **base,
                "status": "WSL_MAPPING_FAILED",
                "launch_readiness": "NOT_READY",
                "package": package_receipt,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_receipt(receipt_path, failed)
            return failed
        wsl_mapping = {
            "status": "PASS" if mapped_path == args.package_cli_path else "MISMATCH",
            "distro": args.wsl_distro,
            "mapped_package_path": mapped_path,
            "expected_package_cli_path": args.package_cli_path,
        }
        if mapped_path != args.package_cli_path:
            failed = {
                **base,
                "status": "WSL_MAPPING_FAILED",
                "launch_readiness": "NOT_READY",
                "package": package_receipt,
                "wsl_package_mapping": wsl_mapping,
                "error": "Windows package path does not map to --package-cli-path",
            }
            _write_receipt(receipt_path, failed)
            return failed
    else:
        wsl_mapping = {
            "status": "NOT_CHECKED",
            "reason": "supply both --wsl-distro and --package-cli-path for read-only mapping",
        }

    missing = _missing_launch_options(args)
    dry = {
        **base,
        "status": "PREFLIGHT_PASS_NO_LAUNCH",
        "launch_readiness": "NOT_READY_MISSING_LIVE_INPUTS" if missing else "NOT_EVALUATED",
        "missing_launch_options": missing,
        "package": package_receipt,
        "wsl_package_mapping": wsl_mapping,
    }
    if not args.launch:
        _write_receipt(receipt_path, dry)
        return dry
    if missing:
        not_ready = {**dry, "status": "NOT_READY", "launch_readiness": "NOT_READY"}
        _write_receipt(receipt_path, not_ready)
        return not_ready

    store_path = args.store.resolve()
    quota_path = args.quota_observation_json.resolve()
    if not store_path.is_file() or not quota_path.is_file():
        not_ready = {
            **dry,
            "status": "NOT_READY",
            "launch_readiness": "NOT_READY",
            "error": "campaign store and quota observation must already exist",
        }
        _write_receipt(receipt_path, not_ready)
        return not_ready
    try:
        quota_observation = _strict_json_object(quota_path)
        quota_contract = QuotaContract(
            ledger_id=args.quota_ledger_id,
            measured_debit_rate_per_wall_hour=args.measured_quota_rate,
            notebook_timeout_seconds=args.notebook_timeout_seconds,
            final_attempt_runtime_bound_hours=args.final_runtime_hours,
            verified_platform_runtime_limit_hours=args.platform_runtime_limit_hours,
            documented_debit_rate_upper_bound_per_wall_hour=args.documented_quota_rate_upper_bound,
            quota_rate_source=args.quota_rate_source,
        )
        client = client_factory(["wsl", "-d", args.wsl_distro, "--", args.kaggle_cli_path])
        verifier = verifier_factory(args.wsl_distro)
        store = store_factory(store_path)
        launch_input = {
            "store": str(store_path),
            "run_id": args.run_id,
            "decision_id": args.decision_id,
            "intent_id": args.intent_id,
            "request_id": args.request_id,
            "reservation_id": args.reservation_id,
            "reviewer": args.reviewer,
            "quota_observation_path": str(quota_path),
            "quota_observation_sha256": _sha256(quota_path),
            "quota_contract": asdict(quota_contract),
            "package_cli_path": args.package_cli_path,
            "wsl_distro": args.wsl_distro,
            "kaggle_cli_path": args.kaggle_cli_path,
        }
        ready = {
            **dry,
            "status": "READY_FOR_EXPLICIT_LAUNCH",
            "launch_readiness": "LIBRARY_GATES_PENDING",
            "launch_input": launch_input,
        }
        _write_receipt(receipt_path, ready)
        with store.review_lock(args.reviewer) as lease:
            result = launcher(
                store=store,
                run_id=args.run_id,
                package_dir=args.package_dir,
                package_cli_path=args.package_cli_path,
                identity=package_identity,
                quota_contract=quota_contract,
                quota_observation=quota_observation,
                client=client,
                decision_id=args.decision_id,
                intent_id=args.intent_id,
                request_id=args.request_id,
                reservation_id=args.reservation_id,
                review_lease=lease,
                verify_cli_package_path=verifier,
                now=generated_at,
            )
        final = {
            **ready,
            "status": str(result.get("status", "UNKNOWN")),
            "launch_readiness": "CONSUMED",
            "launch_result": dict(result),
            "retry_forbidden_without_reconciliation": result.get("status") == "UNKNOWN",
        }
        _write_receipt(receipt_path, final)
        return final
    except Exception as exc:  # noqa: BLE001 - persist any operator-boundary failure.
        rejected = {
            **dry,
            "status": "LAUNCH_REJECTED_OR_UNRESOLVED",
            "launch_readiness": "NOT_READY",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "reconcile_before_retry": True,
        }
        _write_receipt(receipt_path, rejected)
        return rejected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", choices=tuple(REHEARSAL_PACKAGES), default="r3")
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument(
        "--receipt-json",
        type=Path,
        default=None,
    )
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--store", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--decision-id")
    parser.add_argument("--intent-id")
    parser.add_argument("--request-id")
    parser.add_argument("--reservation-id")
    parser.add_argument("--reviewer")
    parser.add_argument("--quota-observation-json", type=Path)
    parser.add_argument("--quota-ledger-id")
    rate_group = parser.add_mutually_exclusive_group()
    rate_group.add_argument("--measured-quota-rate")
    rate_group.add_argument("--documented-quota-rate-upper-bound")
    parser.add_argument("--quota-rate-source")
    parser.add_argument("--notebook-timeout-seconds", type=int)
    parser.add_argument("--final-runtime-hours")
    parser.add_argument("--platform-runtime-limit-hours")
    parser.add_argument("--package-cli-path")
    parser.add_argument("--wsl-distro")
    parser.add_argument("--kaggle-cli-path")
    args = parser.parse_args(argv)
    if args.package_dir is None:
        args.package_dir = ROOT / DEFAULT_PACKAGE_DIRS[args.generation]
    if args.receipt_json is None:
        args.receipt_json = (
            ROOT / f"reports/campaigns/e0-{args.generation}-kaggle-rehearsal-operator.json"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = execute(args)
    except ValueError as exc:
        print(
            json.dumps(
                {"status": "INPUT_REJECTED", "error": str(exc)},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2
    receipt_path = args.receipt_json.resolve()
    summary = {
        "status": receipt["status"],
        "launch_requested": receipt["launch_requested"],
        "receipt_json": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    if receipt["status"] in {"PREFLIGHT_PASS_NO_LAUNCH", "CONFIRMED"}:
        return 0
    if receipt["status"] == "UNKNOWN":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
