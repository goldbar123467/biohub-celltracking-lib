from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from biohub_ct.campaign import CampaignStore
from biohub_ct.campaign.kaggle_rehearsal import (
    E0_R3_PACKAGE_IDENTITY,
    KaggleRehearsalCLI,
    PackageIdentity,
    QuotaContract,
    RehearsalError,
    WSLPathVerifier,
    authorize_and_launch_rehearsal,
    preflight_package,
    reconcile_rehearsal,
)

NOW = datetime(2026, 9, 8, 3, 0, tzinfo=UTC)
CLI_PATH = "/mnt/c/reviewed-package"
RECOVERY_CLI_PATH = "/mnt/c/recovery"
SDK_PYTHON_PREFIX = ["wsl", "-d", "Ubuntu-24.04", "--", "/sdk/python"]


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_package(tmp_path: Path) -> tuple[Path, PackageIdentity]:
    root = tmp_path / "package"
    root.mkdir()
    release = "a" * 64
    datasets = (
        ("owner/dataset-one", 2),
        ("owner/dataset-two", 7),
    )
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [{"name": "stdout", "output_type": "stream", "text": ["1\n"]}],
                "source": ["x = 1\n", "x"],
            }
        ],
        "metadata": {"release_digest": release},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_hash = _write_json(root / "submission.ipynb", notebook)
    normalized = json.loads(json.dumps(notebook))
    normalized["cells"][0]["outputs"] = []
    normalized["cells"][0]["source"] = "x = 1\nx"
    normalized_hash = hashlib.sha256(json.dumps(normalized).encode()).hexdigest()
    metadata = {
        "code_file": "submission.ipynb",
        "competition_sources": ["biohub-test"],
        "dataset_sources": [f"{ref}/{version}" for ref, version in datasets],
        "enable_gpu": True,
        "enable_internet": False,
        "enable_tpu": False,
        "id": "owner/rehearsal",
        "is_private": True,
        "kernel_sources": [],
        "kernel_type": "notebook",
        "language": "python",
        "machine_shape": "NvidiaTeslaT4",
        "model_sources": [],
        "title": "Exact Rehearsal",
    }
    metadata_hash = _write_json(root / "kernel-metadata.json", metadata)
    artifact_hash = _write_json(
        root / "artifact-lock.json",
        {
            "schema_version": 1,
            "competition": "biohub-test",
            "release_digest": release,
            "datasets": {ref: {"version": version} for ref, version in datasets},
        },
    )
    manifest_hash = _write_json(
        root / "package-manifest.json",
        {
            "schema_version": 1,
            "status": "ready_for_root_review_not_launched",
            "release_digest": release,
            "kernel_metadata_sha256": metadata_hash,
            "artifact_lock_sha256": artifact_hash,
            "packaged_notebook": {
                "path": "submission.ipynb",
                "sha256": notebook_hash,
            },
            "algorithm_changes": [],
        },
    )
    return root, PackageIdentity(
        notebook_slug="owner/rehearsal",
        notebook_title="Exact Rehearsal",
        competition="biohub-test",
        machine_shape="NvidiaTeslaT4",
        release_digest=release,
        package_manifest_sha256=manifest_hash,
        kernel_metadata_sha256=metadata_hash,
        artifact_lock_sha256=artifact_hash,
        notebook_sha256=notebook_hash,
        cli_normalized_notebook_sha256=normalized_hash,
        dataset_versions=datasets,
    )


class ScriptedRunner:
    def __init__(self, *steps: object) -> None:
        self.steps = list(steps)
        self.calls: list[list[str]] = []
        self.call_kwargs: list[dict[str, object]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> object:
        assert kwargs["shell"] is False
        self.calls.append(argv)
        self.call_kwargs.append(dict(kwargs))
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(argv, **kwargs)
        return step


def completed(stdout: str = "", *, stderr: str = "", code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)


def success(version: int = 7) -> str:
    return (
        f"Kernel version {version} successfully pushed.  Please check progress at "
        "https://www.kaggle.com/code/owner/rehearsal\n"
    )


def exact_pull_step(
    recovery_dir: Path,
    package_dir: Path,
    identity: PackageIdentity,
    *,
    source_override: str | None = None,
    metadata_override: dict[str, object] | None = None,
):
    def run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        del argv
        helper_input = json.loads(str(kwargs["input"]))
        notebook = json.loads((package_dir / "submission.ipynb").read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if "outputs" in cell and cell["cell_type"] == "code":
                cell["outputs"] = []
            if "source" in cell and isinstance(cell["source"], list):
                cell["source"] = "".join(cell["source"])
        source = source_override if source_override is not None else json.dumps(notebook, indent=2)
        source_bytes = source.encode()
        (recovery_dir / "source.ipynb").write_bytes(source_bytes)
        metadata: dict[str, object] = {
            "id": 4242,
            "ref": identity.notebook_slug,
            "title": identity.notebook_title,
            "slug": identity.notebook_slug.split("/", 1)[1],
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": False,
            "dataset_data_sources": [ref for ref, _ in identity.dataset_versions],
            "kernel_data_sources": [],
            "competition_data_sources": [identity.competition],
            "model_data_sources": [],
            "enable_tpu": False,
            "current_version_number": 3,
            "docker_image": "gcr.io/kaggle/python@sha256:fixture",
            "machine_shape": identity.machine_shape,
        }
        metadata.update(metadata_override or {})
        result = {
            "schema_version": 1,
            "status": "PASS",
            "kaggle_package_version": "2.2.4",
            "request": helper_input["request"],
            "metadata": metadata,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_bytes": len(source_bytes),
            "helper_deadline_seconds": helper_input["helper_deadline_seconds"],
            "deadline_enforcement": "POSIX_SETITIMER_OS_EXIT",
        }
        return completed(json.dumps(result, sort_keys=True))

    return run


def quota_contract() -> QuotaContract:
    return QuotaContract(
        ledger_id="kaggle-gpu-hours",
        measured_debit_rate_per_wall_hour="1",
        notebook_timeout_seconds=3600,
        final_attempt_runtime_bound_hours="2",
        verified_platform_runtime_limit_hours="12",
    )


def documented_quota_contract(
    *,
    rate: object = "1.1",
    source: str | None = "Kaggle staff policy plus local conservative bound",
) -> QuotaContract:
    return QuotaContract(
        ledger_id="kaggle-gpu-hours",
        measured_debit_rate_per_wall_hour=None,
        notebook_timeout_seconds=3600,
        final_attempt_runtime_bound_hours="2",
        verified_platform_runtime_limit_hours="12",
        documented_debit_rate_upper_bound_per_wall_hour=rate,
        quota_rate_source=source,
    )


def quota_observation() -> dict[str, object]:
    return {
        "resource": "GPU",
        "ledger_id": "kaggle-gpu-hours",
        "remaining_hours": "10",
        "used_hours": "0",
        "total_hours": "10",
        "active_gpu_jobs": 0,
        "observed_at": NOW.isoformat(),
    }


def path_verifier(*outputs: str) -> WSLPathVerifier:
    values = outputs or (CLI_PATH,)
    calls = 0

    def resolve(argv: list[str], **kwargs: object) -> object:
        nonlocal calls
        assert kwargs["shell"] is False
        value = values[min(calls, len(values) - 1)]
        calls += 1
        return completed(value + "\n")

    return WSLPathVerifier("Ubuntu-24.04", runner=resolve)


def ready_store(
    tmp_path: Path,
    package: PackageIdentity,
    client: KaggleRehearsalCLI,
    *,
    routine_permission: bool = True,
    private_rehearsal_permission: bool = False,
    protected_reserve: str = "4",
    quota: QuotaContract | None = None,
    binding_override: dict[str, object] | None = None,
    authorization_source_binding: object | None = None,
) -> CampaignStore:
    quota = quota or quota_contract()
    rate = (
        quota.measured_debit_rate_per_wall_hour
        if quota.measured_debit_rate_per_wall_hour is not None
        else quota.documented_debit_rate_upper_bound_per_wall_hour
    )
    reservation = Decimal(quota.notebook_timeout_seconds) / Decimal(3600) * Decimal(str(rate))
    authorization_source = (
        "explicit private Kaggle rehearsal authorization"
        if private_rehearsal_permission
        else ("explicit test authorization" if routine_permission else "existing allocation only")
    )
    store = CampaignStore(tmp_path / "campaign.sqlite3")
    store.initialize_campaign(
        "campaign-test",
        "biohub-test",
        status="ACTIVE",
        authorization={
            "routine_runs_and_submissions": routine_permission,
            "private_kaggle_rehearsals": private_rehearsal_permission,
            "existing_allocation_authorized": True,
            "source": authorization_source,
        },
    )
    store.create_ledger(
        "kaggle-gpu-hours",
        resource_scope="Kaggle account GPU quota",
        unit="quota_hours",
        authorized_total="10",
        protected_reserve=protected_reserve,
        observation_time=NOW.isoformat(),
    )
    binding: dict[str, object] = {
        "schema_version": 1,
        "notebook_slug": package.notebook_slug,
        "notebook_title": package.notebook_title,
        "competition": package.competition,
        "dataset_versions": dict(package.dataset_versions),
        "release_digest": package.release_digest,
        "package_manifest_sha256": package.package_manifest_sha256,
        "kernel_metadata_sha256": package.kernel_metadata_sha256,
        "artifact_lock_sha256": package.artifact_lock_sha256,
        "notebook_sha256": package.notebook_sha256,
        "cli_normalized_notebook_sha256": package.cli_normalized_notebook_sha256,
        "package_cli_path": CLI_PATH,
        "machine_shape": package.machine_shape,
        "notebook_timeout_seconds": quota.notebook_timeout_seconds,
        "final_attempt_runtime_bound_hours": str(quota.final_attempt_runtime_bound_hours),
        "verified_platform_runtime_limit_hours": str(quota.verified_platform_runtime_limit_hours),
        "required_two_attempt_reserve": protected_reserve,
    }
    if quota.measured_debit_rate_per_wall_hour is not None:
        binding["measured_debit_rate_per_wall_hour"] = str(quota.measured_debit_rate_per_wall_hour)
    else:
        binding.update(
            quota_rate_mode="documented_upper_bound",
            documented_debit_rate_upper_bound_per_wall_hour=str(
                quota.documented_debit_rate_upper_bound_per_wall_hour
            ),
            quota_rate_source=quota.quota_rate_source,
        )
    binding.update(binding_override or {})
    run_spec: dict[str, object] = {
        "schema_version": 1,
        "run_id": "rehearsal-1",
        "experiment_id": "E0-kaggle-rehearsal",
        "work_kind": "operational_verification",
        "hypothesis": "exact private offline package executes",
        "source": {
            "source_bundle_sha256": package.package_manifest_sha256,
            "dependency_manifest_sha256": package.artifact_lock_sha256,
            "effective_config_sha256": package.kernel_metadata_sha256,
        },
        "data": {
            "input_manifest_sha256": package.artifact_lock_sha256,
            "split_manifest_sha256": package.artifact_lock_sha256,
        },
        "execution": {
            "host": "kaggle",
            "working_directory": CLI_PATH,
            "argv": client.push_argv(
                package_cli_path=CLI_PATH,
                accelerator=package.machine_shape,
                notebook_timeout_seconds=quota.notebook_timeout_seconds,
            ),
            "max_wall_seconds": 120,
            "max_steps_or_clips": 1,
            "max_quota_hours": str(reservation),
            "deadline_utc": (NOW + timedelta(hours=1)).isoformat(),
        },
        "kaggle_rehearsal": binding,
        "stop_rules": ["one push only", "quota bound"],
        "success_rules": ["exact version and status receipt"],
        "expected_artifacts": ["submission.csv", "run-manifest.json"],
    }
    bind_authorization_source = (
        private_rehearsal_permission
        if authorization_source_binding is None
        else authorization_source_binding is not False
    )
    if bind_authorization_source:
        run_spec["authorization_source_sha256"] = (
            hashlib.sha256(authorization_source.encode()).hexdigest()
            if authorization_source_binding is None or authorization_source_binding is True
            else authorization_source_binding
        )
    store.register_run(run_spec)
    return store


def launch(
    store: CampaignStore,
    package_dir: Path,
    package: PackageIdentity,
    client: KaggleRehearsalCLI,
    *,
    quota: QuotaContract | None = None,
) -> dict[str, object]:
    with store.review_lock("reviewer") as lease:
        return authorize_and_launch_rehearsal(
            store=store,
            run_id="rehearsal-1",
            package_dir=package_dir,
            package_cli_path=CLI_PATH,
            identity=package,
            quota_contract=quota or quota_contract(),
            quota_observation=quota_observation(),
            client=client,
            decision_id="decision-1",
            intent_id="intent-1",
            request_id="request-1",
            reservation_id="reservation-1",
            review_lease=lease,
            verify_cli_package_path=path_verifier(),
            now=NOW,
        )


def test_actual_e0_r3_package_preflights() -> None:
    root = Path(__file__).parents[1] / "work/e0-reference/package-r3"
    if not root.is_dir():
        pytest.skip(
            "optional integration artifact work/e0-reference/package-r3 is not in a clean clone"
        )
    receipt = preflight_package(root, E0_R3_PACKAGE_IDENTITY)
    assert receipt["status"] == "PASS"
    assert receipt["private"] is True
    assert receipt["internet"] is False
    assert (
        receipt["cli_normalized_notebook_sha256"]
        == E0_R3_PACKAGE_IDENTITY.cli_normalized_notebook_sha256
    )


def test_package_rejects_byte_drift_and_unreviewed_files(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    assert preflight_package(root, identity)["release_digest"] == "a" * 64
    (root / "extra.txt").write_text("unexpected")
    with pytest.raises(RehearsalError, match="file set"):
        preflight_package(root, identity)
    (root / "extra.txt").unlink()
    (root / "submission.ipynb").write_text("{}")
    with pytest.raises(RehearsalError, match="bytes"):
        preflight_package(root, identity)


def test_requires_live_os_review_lock_and_wsl_mapping(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    client = KaggleRehearsalCLI(["kaggle"], runner=ScriptedRunner())
    store = ready_store(tmp_path, identity, client)
    lease = store.review_lock("reviewer")
    with pytest.raises(RehearsalError, match="OS review lock"):
        authorize_and_launch_rehearsal(
            store=store,
            run_id="rehearsal-1",
            package_dir=root,
            package_cli_path=CLI_PATH,
            identity=identity,
            quota_contract=quota_contract(),
            quota_observation=quota_observation(),
            client=client,
            decision_id="decision-1",
            intent_id="intent-1",
            request_id="request-1",
            reservation_id="reservation-1",
            review_lease=lease,
            verify_cli_package_path=path_verifier("/different/package"),
            now=NOW,
        )
    with (
        store.review_lock("reviewer") as live_lease,
        pytest.raises(RehearsalError, match="does not map"),
    ):
        authorize_and_launch_rehearsal(
            store=store,
            run_id="rehearsal-1",
            package_dir=root,
            package_cli_path=CLI_PATH,
            identity=identity,
            quota_contract=quota_contract(),
            quota_observation=quota_observation(),
            client=client,
            decision_id="decision-1",
            intent_id="intent-1",
            request_id="request-1",
            reservation_id="reservation-1",
            review_lease=live_lease,
            verify_cli_package_path=path_verifier("/different/package"),
            now=NOW,
        )
    assert store.list_intents() == []
    assert store.budget_snapshot("kaggle-gpu-hours").outstanding_reservations == 0


def test_mapping_drift_after_dispatch_is_unknown_without_push(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner()
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(tmp_path, identity, client)
    with store.review_lock("reviewer") as lease:
        result = authorize_and_launch_rehearsal(
            store=store,
            run_id="rehearsal-1",
            package_dir=root,
            package_cli_path=CLI_PATH,
            identity=identity,
            quota_contract=quota_contract(),
            quota_observation=quota_observation(),
            client=client,
            decision_id="decision-1",
            intent_id="intent-1",
            request_id="request-1",
            reservation_id="reservation-1",
            review_lease=lease,
            verify_cli_package_path=path_verifier(CLI_PATH, "/changed/path"),
            now=NOW,
        )
    assert result["status"] == "UNKNOWN"
    assert runner.calls == []
    assert store.get_intent("intent-1")["state"] == "UNKNOWN"
    assert store.budget_snapshot("kaggle-gpu-hours").outstanding_reservations == 1


def test_permission_and_two_attempt_reserve_fail_before_push(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner()
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    denied = ready_store(tmp_path / "denied", identity, client, routine_permission=False)
    with (
        denied.review_lock("reviewer") as lease,
        pytest.raises(RehearsalError, match="private Kaggle rehearsal or routine run permission"),
    ):
        authorize_and_launch_rehearsal(
            store=denied,
            run_id="rehearsal-1",
            package_dir=root,
            package_cli_path=CLI_PATH,
            identity=identity,
            quota_contract=quota_contract(),
            quota_observation=quota_observation(),
            client=client,
            decision_id="decision-1",
            intent_id="intent-1",
            request_id="request-1",
            reservation_id="reservation-1",
            review_lease=lease,
            verify_cli_package_path=path_verifier(),
            now=NOW,
        )
    under_reserved = ready_store(tmp_path / "reserve", identity, client, protected_reserve="3.999")
    with (
        under_reserved.review_lock("reviewer") as lease,
        pytest.raises(RehearsalError, match="two bounded final attempts"),
    ):
        authorize_and_launch_rehearsal(
            store=under_reserved,
            run_id="rehearsal-1",
            package_dir=root,
            package_cli_path=CLI_PATH,
            identity=identity,
            quota_contract=quota_contract(),
            quota_observation=quota_observation(),
            client=client,
            decision_id="decision-1",
            intent_id="intent-1",
            request_id="request-1",
            reservation_id="reservation-1",
            review_lease=lease,
            verify_cli_package_path=path_verifier(),
            now=NOW,
        )
    assert runner.calls == []


def test_private_scoped_permission_allows_only_reviewed_rehearsal_authority(
    tmp_path: Path,
) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner(completed(success(13)))
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(
        tmp_path,
        identity,
        client,
        routine_permission=False,
        private_rehearsal_permission=True,
    )

    result = launch(store, root, identity, client)

    assert result["status"] == "CONFIRMED"
    assert result["exact_kernel_ref"] == "owner/rehearsal/13"
    assert len(runner.calls) == 1
    authorization = store.get_campaign()["authorization"]
    assert authorization["private_kaggle_rehearsals"] is True
    assert authorization["routine_runs_and_submissions"] is False
    assert authorization["source"] == "explicit private Kaggle rehearsal authorization"


@pytest.mark.parametrize(
    ("source_binding", "reason"),
    [
        (False, "does not bind its authorization source"),
        ("not-a-sha256", "not a valid SHA-256"),
    ],
)
def test_private_scoped_permission_requires_source_hash_binding(
    tmp_path: Path,
    source_binding: object,
    reason: str,
) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner()
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(
        tmp_path,
        identity,
        client,
        routine_permission=False,
        private_rehearsal_permission=True,
        authorization_source_binding=source_binding,
    )

    with pytest.raises(RehearsalError, match=reason):
        launch(store, root, identity, client)

    assert runner.calls == []
    assert store.list_intents() == []


def test_authorization_source_drift_rejects_private_rehearsal(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner()
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(
        tmp_path,
        identity,
        client,
        routine_permission=False,
        private_rehearsal_permission=True,
    )
    with store.review_lock("authorization-reviewer") as lease:
        store.update_campaign(
            {"authorization": {"source": "later authorization source"}},
            reviewer="authorization-reviewer",
            reason="test source drift",
            fencing_token=lease.fencing_token,
        )

    with pytest.raises(RehearsalError, match="changed after run review"):
        launch(store, root, identity, client)

    assert runner.calls == []
    assert store.list_intents() == []


def test_legacy_routine_permission_still_allows_reviewed_rehearsal(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner(completed(success(14)))
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(
        tmp_path,
        identity,
        client,
        routine_permission=True,
        private_rehearsal_permission=False,
    )

    result = launch(store, root, identity, client)

    assert result["status"] == "CONFIRMED"
    assert result["exact_kernel_ref"] == "owner/rehearsal/14"
    assert len(runner.calls) == 1


def test_legacy_routine_spec_rejects_present_authorization_hash_drift(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner()
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(
        tmp_path,
        identity,
        client,
        routine_permission=True,
        private_rehearsal_permission=False,
        authorization_source_binding="0" * 64,
    )

    with pytest.raises(RehearsalError, match="changed after run review"):
        launch(store, root, identity, client)

    assert runner.calls == []
    assert store.list_intents() == []


def test_documented_rate_upper_bound_is_honest_and_immutably_bound(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    contract = documented_quota_contract()
    runner = ScriptedRunner(completed(success(15)))
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(
        tmp_path,
        identity,
        client,
        routine_permission=False,
        private_rehearsal_permission=True,
        protected_reserve="4.4",
        quota=contract,
    )

    result = launch(store, root, identity, client, quota=contract)

    assert result["status"] == "CONFIRMED"
    binding = store.get_run("rehearsal-1")["spec"]["kaggle_rehearsal"]
    assert binding["quota_rate_mode"] == "documented_upper_bound"
    assert binding["documented_debit_rate_upper_bound_per_wall_hour"] == "1.1"
    assert binding["quota_rate_source"] == ("Kaggle staff policy plus local conservative bound")
    assert "measured_debit_rate_per_wall_hour" not in binding
    quota_receipt = store.get_intent("intent-1")["receipt"]["quota"]
    assert quota_receipt["quota_rate_mode"] == "documented_upper_bound"
    assert "measured_debit_rate_per_wall_hour" not in quota_receipt
    assert store.budget_snapshot("kaggle-gpu-hours").outstanding_reservations == Decimal("1.1")


@pytest.mark.parametrize(
    ("contract", "reason"),
    [
        (
            QuotaContract(
                ledger_id="kaggle-gpu-hours",
                measured_debit_rate_per_wall_hour="1",
                notebook_timeout_seconds=3600,
                final_attempt_runtime_bound_hours="2",
                verified_platform_runtime_limit_hours="12",
                documented_debit_rate_upper_bound_per_wall_hour="1.1",
                quota_rate_source="documented policy",
            ),
            "mutually exclusive",
        ),
        (documented_quota_contract(source=None), "nonempty source"),
        (documented_quota_contract(rate="0"), "must be positive"),
        (documented_quota_contract(rate="NaN"), "finite and nonnegative"),
    ],
)
def test_invalid_documented_rate_contract_fails_before_push(
    tmp_path: Path,
    contract: QuotaContract,
    reason: str,
) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner()
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(
        tmp_path,
        identity,
        client,
        routine_permission=False,
        private_rehearsal_permission=True,
    )

    with pytest.raises(RehearsalError, match=reason):
        launch(store, root, identity, client, quota=contract)

    assert runner.calls == []
    assert store.list_intents() == []


def test_documented_rate_source_drift_rejects_registered_binding(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    contract = documented_quota_contract()
    runner = ScriptedRunner()
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(
        tmp_path,
        identity,
        client,
        routine_permission=False,
        private_rehearsal_permission=True,
        protected_reserve="4.4",
        quota=contract,
        binding_override={"quota_rate_source": "different source"},
    )

    with pytest.raises(RehearsalError, match="exact package and quota contract"):
        launch(store, root, identity, client, quota=contract)

    assert runner.calls == []
    assert store.list_intents() == []


def test_exact_push_acknowledgment_confirms_one_launch_without_status(
    tmp_path: Path,
) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner(completed(success()))
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(tmp_path, identity, client)
    result = launch(store, root, identity, client)
    assert result == {
        "status": "CONFIRMED",
        "intent_id": "intent-1",
        "exact_kernel_ref": "owner/rehearsal/7",
        "kernel_status": None,
        "reservation_ids": ["reservation-1"],
    }
    assert len(runner.calls) == 1
    assert runner.calls[0][-6:] == [
        "--path",
        CLI_PATH,
        "--timeout",
        "3600",
        "--accelerator",
        "NvidiaTeslaT4",
    ]
    persisted = store.get_intent("intent-1")["receipt"]
    assert persisted["evidence_scope"] == "exact_push_acknowledgment"
    assert persisted["kernel_status_scope"] == "not_read_slug_current_only"
    assert store.get_run("rehearsal-1")["state"] == "RUNNING"
    assert store.budget_snapshot("kaggle-gpu-hours").outstanding_reservations == 1


def test_only_persisted_literal_exact_push_receipt_can_bypass_pull(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner(completed(success(11)))
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(tmp_path, identity, client)
    original_confirm = store.confirm_launch

    def strand_after_receipt(
        intent_id: str, *, provider_job_id: str, receipt: dict[str, object]
    ) -> dict[str, object]:
        del provider_job_id
        store.mark_intent_unknown(intent_id, receipt=receipt)
        raise RuntimeError("simulated controller interruption")

    store.confirm_launch = strand_after_receipt  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="controller interruption"):
        launch(store, root, identity, client)
    store.confirm_launch = original_confirm  # type: ignore[method-assign]

    recovered = reconcile_rehearsal(store=store, intent_id="intent-1", client=client)
    assert recovered["status"] == "CONFIRMED"
    assert recovered["exact_kernel_ref"] == "owner/rehearsal/11"
    assert recovered["kernel_status"] is None
    assert len(runner.calls) == 1


def test_timeout_is_unknown_retains_quota_and_cannot_push_twice(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner(subprocess.TimeoutExpired(["kaggle"], 120))
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(tmp_path, identity, client)
    result = launch(store, root, identity, client)
    assert result["status"] == "UNKNOWN"
    assert result["reservations_retained"] is True
    assert store.get_run("rehearsal-1")["state"] == "LAUNCH_UNKNOWN"
    assert store.budget_snapshot("kaggle-gpu-hours").outstanding_reservations == 1
    with pytest.raises(RehearsalError):
        launch(store, root, identity, client)
    assert len(runner.calls) == 1


def test_nonexact_push_version_cannot_bypass_candidate_proof(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    stdout = success() + "owner/missing is not a valid dataset source\n"
    runner = ScriptedRunner(completed(stdout))
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(tmp_path, identity, client)
    launched = launch(store, root, identity, client)
    assert launched["status"] == "UNKNOWN"
    assert launched["receipt"]["push"]["notebook_version"] == 7
    assert launched["receipt"]["push"]["exact_receipt"] is False

    recovered = reconcile_rehearsal(store=store, intent_id="intent-1", client=client)
    assert recovered["status"] == "UNKNOWN"
    assert "controlled exact-version SDK source read" in recovered["reason"]
    assert len(runner.calls) == 1
    assert store.get_intent("intent-1")["state"] == "UNKNOWN"


def test_ambiguous_push_without_version_does_not_query_or_claim_absence(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner(completed("Kernel version successfully pushed.\n"))
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(tmp_path, identity, client)
    assert launch(store, root, identity, client)["status"] == "UNKNOWN"
    no_read_runner = ScriptedRunner()
    recovered = reconcile_rehearsal(
        store=store,
        intent_id="intent-1",
        client=KaggleRehearsalCLI(["kaggle"], runner=no_read_runner),
    )
    assert recovered["status"] == "UNKNOWN"
    assert "absence was not inferred" in recovered["reason"]
    assert no_read_runner.calls == []


def test_exact_source_recovery_requires_explicit_wsl_sdk_python_prefix(
    tmp_path: Path,
) -> None:
    root, identity = make_package(tmp_path)
    launch_client = KaggleRehearsalCLI(["kaggle"], runner=ScriptedRunner(completed("ambiguous")))
    store = ready_store(tmp_path, identity, launch_client)
    assert launch(store, root, identity, launch_client)["status"] == "UNKNOWN"
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    runner = ScriptedRunner()

    result = reconcile_rehearsal(
        store=store,
        intent_id="intent-1",
        client=KaggleRehearsalCLI(["kaggle"], runner=runner),
        exact_version=3,
        recovery_dir=recovery,
        recovery_cli_path=RECOVERY_CLI_PATH,
        verify_cli_recovery_path=path_verifier(RECOVERY_CLI_PATH),
    )

    assert result["status"] == "UNKNOWN"
    assert "explicit Kaggle SDK Python prefix" in result["reason"]
    assert "partial_evidence" not in result
    assert runner.calls == []
    assert store.get_intent("intent-1")["state"] == "UNKNOWN"


def test_exact_pull_persists_and_verifies_partial_evidence_but_stays_unknown(
    tmp_path: Path,
) -> None:
    root, identity = make_package(tmp_path)
    launch_client = KaggleRehearsalCLI(["kaggle"], runner=ScriptedRunner(completed("ambiguous")))
    store = ready_store(tmp_path, identity, launch_client)
    assert launch(store, root, identity, launch_client)["status"] == "UNKNOWN"

    recovery = tmp_path / "recovery"
    recovery.mkdir()
    runner = ScriptedRunner(exact_pull_step(recovery, root, identity))
    client = KaggleRehearsalCLI(["kaggle"], sdk_python_prefix=SDK_PYTHON_PREFIX, runner=runner)
    recovered = reconcile_rehearsal(
        store=store,
        intent_id="intent-1",
        client=client,
        exact_version=3,
        recovery_dir=recovery,
        recovery_cli_path=RECOVERY_CLI_PATH,
        verify_cli_recovery_path=path_verifier(RECOVERY_CLI_PATH),
    )
    assert recovered["status"] == "UNKNOWN"
    assert recovered["exact_kernel_ref"] == "owner/rehearsal/3"
    assert recovered["partial_evidence"]["identity_verified"] is True
    assert recovered["partial_evidence"]["intent_association"] == "unresolved"
    receipt = recovered["partial_evidence"]["receipt"]
    receipt_path = Path(receipt["artifact_path"])
    assert receipt_path.is_file()
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == receipt["artifact_sha256"]
    artifact = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert artifact["provider_creation_time"] is None
    assert artifact["intent_association"] == "unresolved"
    assert artifact["evidence_scope"] == "kaggle_sdk_get_kernel_exact_version_source"
    assert artifact["dataset_versions_status"] == "UNAVAILABLE_FROM_PROVIDER_METADATA"
    assert artifact["source"]["sha256"] == receipt["source_sha256"]
    assert artifact["source"]["cli_normalized_sha256"] == (identity.cli_normalized_notebook_sha256)
    assert artifact["source"]["sha256"] != artifact["source"]["cli_normalized_sha256"]
    assert runner.calls[0][:-2] == SDK_PYTHON_PREFIX
    assert runner.calls[0][-2] == "-c"
    assert 'request.version_label = fields["version_label"]' in runner.calls[0][-1]
    helper_input = json.loads(str(runner.call_kwargs[0]["input"]))
    assert helper_input["request"] == {
        "user_name": "owner",
        "kernel_slug": "rehearsal",
        "version_label": "v3",
    }
    assert helper_input["max_source_bytes"] == 64 * 1024 * 1024
    assert helper_input["max_metadata_bytes"] == 1024 * 1024
    assert 0 < helper_input["helper_deadline_seconds"] < runner.call_kwargs[0]["timeout"]
    assert store.get_intent("intent-1")["state"] == "UNKNOWN"
    assert store.budget_snapshot("kaggle-gpu-hours").outstanding_reservations == 1


@pytest.mark.parametrize(
    ("source_override", "metadata_override", "reason"),
    [
        ("{}", None, "normalized notebook"),
        (None, {"dataset_data_sources": ["owner/dataset-one"]}, "dataset_data_sources"),
        (None, {"current_version_number": 2}, "current_version_number"),
        (None, {"is_private": False}, "is_private"),
        (None, {"enable_gpu": False}, "enable_gpu"),
        (None, {"enable_internet": True}, "enable_internet"),
    ],
)
def test_candidate_pull_rejects_source_or_versioned_metadata_drift(
    tmp_path: Path,
    source_override: str | None,
    metadata_override: dict[str, object] | None,
    reason: str,
) -> None:
    root, identity = make_package(tmp_path)
    launch_client = KaggleRehearsalCLI(["kaggle"], runner=ScriptedRunner(completed("ambiguous")))
    store = ready_store(tmp_path, identity, launch_client)
    assert launch(store, root, identity, launch_client)["status"] == "UNKNOWN"
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    runner = ScriptedRunner(
        exact_pull_step(
            recovery,
            root,
            identity,
            source_override=source_override,
            metadata_override=metadata_override,
        )
    )
    result = reconcile_rehearsal(
        store=store,
        intent_id="intent-1",
        client=KaggleRehearsalCLI(["kaggle"], sdk_python_prefix=SDK_PYTHON_PREFIX, runner=runner),
        exact_version=3,
        recovery_dir=recovery,
        recovery_cli_path=RECOVERY_CLI_PATH,
        verify_cli_recovery_path=path_verifier(RECOVERY_CLI_PATH),
    )
    assert result["status"] == "UNKNOWN"
    assert reason in result["reason"]
    assert "partial_evidence" not in result


@pytest.mark.parametrize(
    ("field", "malformed", "reason"),
    [
        ("is_private", 1, "exact boolean"),
        ("enable_gpu", 1, "exact boolean"),
        ("enable_internet", 0, "exact boolean"),
        ("enable_tpu", 0, "exact boolean"),
        ("current_version_number", True, "positive integer"),
    ],
)
def test_candidate_sdk_metadata_rejects_bool_integer_aliases(
    tmp_path: Path, field: str, malformed: object, reason: str
) -> None:
    root, identity = make_package(tmp_path)
    launch_client = KaggleRehearsalCLI(["kaggle"], runner=ScriptedRunner(completed("ambiguous")))
    store = ready_store(tmp_path, identity, launch_client)
    assert launch(store, root, identity, launch_client)["status"] == "UNKNOWN"
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    runner = ScriptedRunner(
        exact_pull_step(recovery, root, identity, metadata_override={field: malformed})
    )

    result = reconcile_rehearsal(
        store=store,
        intent_id="intent-1",
        client=KaggleRehearsalCLI(["kaggle"], sdk_python_prefix=SDK_PYTHON_PREFIX, runner=runner),
        exact_version=3,
        recovery_dir=recovery,
        recovery_cli_path=RECOVERY_CLI_PATH,
        verify_cli_recovery_path=path_verifier(RECOVERY_CLI_PATH),
    )

    assert result["status"] == "UNKNOWN"
    assert reason in result["reason"]
    assert "partial_evidence" not in result


def test_slug_current_status_has_no_version_specificity(tmp_path: Path) -> None:
    del tmp_path
    runner = ScriptedRunner(
        completed("KernelWorkerStatus.RUNNING\n"),
        completed("KernelWorkerStatus.RUNNING\n"),
    )
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    assert client.current_status("owner/rehearsal") == "RUNNING"
    assert client.current_status("owner/rehearsal") == "RUNNING"
    assert runner.calls[0][-2:] == ["status", "owner/rehearsal"]
    assert runner.calls[1] == runner.calls[0]


def test_slug_current_error_cannot_mark_exact_ack_failed(tmp_path: Path) -> None:
    root, identity = make_package(tmp_path)
    runner = ScriptedRunner(
        completed(success(9)),
        completed("Kernel status: KernelWorkerStatus.ERROR\n"),
    )
    client = KaggleRehearsalCLI(["kaggle"], runner=runner)
    store = ready_store(tmp_path, identity, client)
    result = launch(store, root, identity, client)
    assert result["status"] == "CONFIRMED"
    assert result["exact_kernel_ref"] == "owner/rehearsal/9"
    assert result["kernel_status"] is None
    assert len(runner.calls) == 1
    assert store.get_run("rehearsal-1")["state"] == "RUNNING"


def test_wsl_path_verifier_uses_literal_read_only_command(tmp_path: Path) -> None:
    root, _ = make_package(tmp_path)
    runner = ScriptedRunner(completed("/mnt/c/reviewed-package\n"))
    verifier = WSLPathVerifier("Ubuntu-24.04", runner=runner)
    assert verifier(root) == CLI_PATH
    assert runner.calls[0][:5] == ["wsl", "-d", "Ubuntu-24.04", "--", "wslpath"]
