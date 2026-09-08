from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.rehearse_kaggle_reference import execute, parse_args

from biohub_ct.campaign.rehearsal_packages import reviewed_package

NOW = datetime(2026, 9, 8, 5, 0, tzinfo=UTC)


def temp_package(tmp_path: Path, generation: str = "r3"):
    package = tmp_path / "package-r3-fixture"
    package.mkdir()
    for name in (
        "artifact-lock.json",
        "kernel-metadata.json",
        "package-manifest.json",
        "submission.ipynb",
    ):
        (package / name).write_text(json.dumps({"fixture": name}) + "\n", encoding="utf-8")

    def preflight(path: Path, identity: object) -> dict[str, object]:
        assert path == package
        assert identity == reviewed_package(generation)
        files = {
            item.name: hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted(package.iterdir())
        }
        assert set(files) == {
            "artifact-lock.json",
            "kernel-metadata.json",
            "package-manifest.json",
            "submission.ipynb",
        }
        return {"status": "PASS", "package_root": str(package), "file_sha256": files}

    return package, preflight


def test_default_flow_is_read_only_and_persists_reviewable_identity(tmp_path: Path) -> None:
    package, preflight = temp_package(tmp_path)
    receipt_path = tmp_path / "preflight.json"
    args = parse_args(["--package-dir", str(package), "--receipt-json", str(receipt_path)])
    receipt = execute(args, package_preflight=preflight, now_factory=lambda: NOW)

    assert receipt["status"] == "PREFLIGHT_PASS_NO_LAUNCH"
    assert receipt["launch_requested"] is False
    assert receipt["launch_readiness"] == "NOT_READY_MISSING_LIVE_INPUTS"
    assert "--measured-quota-rate" in receipt["missing_launch_options"]
    assert "--final-runtime-hours" in receipt["missing_launch_options"]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert receipt["package"]["status"] == "PASS"


def test_launch_flag_without_live_inputs_is_not_ready_and_calls_nothing(tmp_path: Path) -> None:
    package, preflight = temp_package(tmp_path)
    receipt_path = tmp_path / "not-ready.json"
    args = parse_args(
        ["--launch", "--package-dir", str(package), "--receipt-json", str(receipt_path)]
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "missing launch inputs must not construct or call mutation dependencies"
        )

    receipt = execute(
        args,
        store_factory=forbidden,
        client_factory=forbidden,
        verifier_factory=forbidden,
        launcher=forbidden,
        package_preflight=preflight,
        now_factory=lambda: NOW,
    )

    assert receipt["status"] == "NOT_READY"
    assert receipt["launch_requested"] is True
    assert "--quota-observation-json" in receipt["missing_launch_options"]
    assert "--notebook-timeout-seconds" in receipt["missing_launch_options"]


@pytest.mark.parametrize("documented", [False, True])
@pytest.mark.parametrize("generation", ["r3", "r4"])
def test_explicit_launch_threads_exact_inputs_under_review_lease(
    tmp_path: Path, documented: bool, generation: str
) -> None:
    package, preflight = temp_package(tmp_path, generation)
    store_path = tmp_path / "campaign.sqlite3"
    store_path.write_bytes(b"present")
    quota_path = tmp_path / "quota.json"
    quota = {
        "resource": "GPU",
        "ledger_id": "kaggle-gpu-hours",
        "used_hours": "1.5",
        "remaining_hours": "28.5",
        "total_hours": "30",
        "active_gpu_jobs": 0,
        "observed_at": NOW.isoformat(),
    }
    quota_path.write_text(json.dumps(quota), encoding="utf-8")
    receipt_path = tmp_path / "launch.json"
    args = parse_args(
        [
            "--launch",
            "--package-dir",
            str(package),
            "--receipt-json",
            str(receipt_path),
            "--store",
            str(store_path),
            "--run-id",
            "e0-r3-rehearsal",
            "--decision-id",
            "decision-1",
            "--intent-id",
            "intent-1",
            "--request-id",
            "request-1",
            "--reservation-id",
            "reservation-1",
            "--reviewer",
            "root-reviewer",
            "--quota-observation-json",
            str(quota_path),
            "--quota-ledger-id",
            "kaggle-gpu-hours",
            *(
                [
                    "--documented-quota-rate-upper-bound",
                    "1.10",
                    "--quota-rate-source",
                    "platform-evidence",
                ]
                if documented
                else ["--measured-quota-rate", "1.25"]
            ),
            "--notebook-timeout-seconds",
            "3600",
            "--final-runtime-hours",
            "2.5",
            "--platform-runtime-limit-hours",
            "12",
            "--package-cli-path",
            "/mnt/c/e0-r3",
            "--wsl-distro",
            "Ubuntu-24.04",
            "--kaggle-cli-path",
            "/home/thecl/.local/bin/kaggle",
        ]
    )
    events: list[object] = []

    class FakeStore:
        @contextmanager
        def review_lock(self, reviewer: str):
            events.append(("lease_enter", reviewer))
            yield "held-lease"
            events.append(("lease_exit", reviewer))

    def fake_store(path: Path) -> FakeStore:
        events.append(("store", path))
        return FakeStore()

    def fake_client(prefix: list[str]) -> SimpleNamespace:
        events.append(("client", prefix))
        return SimpleNamespace(prefix=prefix)

    def fake_verifier(distro: str):
        events.append(("verifier", distro))

        def verify(path: Path) -> str:
            events.append(("mapping", path))
            return "/mnt/c/e0-r3"

        return verify

    args.generation = generation

    def fake_launcher(**kwargs: object) -> dict[str, object]:
        events.append(("launch", kwargs))
        assert kwargs["identity"] == reviewed_package(generation)
        assert kwargs["review_lease"] == "held-lease"
        assert kwargs["quota_observation"] == quota
        contract = kwargs["quota_contract"]
        assert contract.measured_debit_rate_per_wall_hour == (None if documented else "1.25")
        assert contract.documented_debit_rate_upper_bound_per_wall_hour == (
            "1.10" if documented else None
        )
        assert contract.quota_rate_source == ("platform-evidence" if documented else None)
        assert contract.final_attempt_runtime_bound_hours == "2.5"
        assert contract.verified_platform_runtime_limit_hours == "12"
        assert contract.notebook_timeout_seconds == 3600
        return {
            "status": "CONFIRMED",
            "intent_id": "intent-1",
            "exact_kernel_ref": "owner/notebook/7",
        }

    receipt = execute(
        args,
        store_factory=fake_store,
        client_factory=fake_client,
        verifier_factory=fake_verifier,
        launcher=fake_launcher,
        package_preflight=preflight,
        now_factory=lambda: NOW,
    )

    assert receipt["status"] == "CONFIRMED"
    assert receipt["launch_readiness"] == "CONSUMED"
    assert [event[0] for event in events].count("launch") == 1
    assert events.index(("lease_enter", "root-reviewer")) < next(
        index for index, event in enumerate(events) if event[0] == "launch"
    )
    assert receipt["launch_input"]["quota_contract"]["measured_debit_rate_per_wall_hour"] == (
        None if documented else "1.25"
    )


def test_documented_rate_requires_source_and_rejects_conflicting_modes(tmp_path: Path) -> None:
    package, preflight = temp_package(tmp_path)
    args = parse_args(
        [
            "--package-dir",
            str(package),
            "--receipt-json",
            str(tmp_path / "receipt.json"),
            "--documented-quota-rate-upper-bound",
            "1.10",
        ]
    )
    receipt = execute(args, package_preflight=preflight, now_factory=lambda: NOW)
    assert "--quota-rate-source" in receipt["missing_launch_options"]
    assert "--measured-quota-rate" not in receipt["missing_launch_options"]
    with pytest.raises(SystemExit):
        parse_args(["--measured-quota-rate", "1", "--documented-quota-rate-upper-bound", "1.10"])


def test_receipt_cannot_overwrite_package_store_or_quota_inputs(tmp_path: Path) -> None:
    package, preflight = temp_package(tmp_path)
    store = tmp_path / "campaign.sqlite3"
    quota = tmp_path / "quota.json"
    store.write_bytes(b"canonical-store")
    quota.write_text('{"canonical":"quota"}\n', encoding="utf-8")
    protected = [package / "submission.ipynb", store, quota]
    originals = {path: path.read_bytes() for path in protected}

    for target in protected:
        args = parse_args(
            [
                "--package-dir",
                str(package),
                "--receipt-json",
                str(target),
                "--store",
                str(store),
                "--quota-observation-json",
                str(quota),
            ]
        )
        try:
            execute(args, package_preflight=preflight, now_factory=lambda: NOW)
        except ValueError as exc:
            assert "receipt-json" in str(exc)
        else:
            raise AssertionError("protected receipt destination was accepted")

    assert {path: path.read_bytes() for path in protected} == originals

    partial_store = tmp_path / "operator.json.partial"
    partial_store.write_bytes(b"canonical-store-with-partial-suffix")
    args = parse_args(
        [
            "--package-dir",
            str(package),
            "--receipt-json",
            str(tmp_path / "operator.json"),
            "--store",
            str(partial_store),
        ]
    )
    try:
        execute(args, package_preflight=preflight, now_factory=lambda: NOW)
    except ValueError as exc:
        assert "campaign store" in str(exc)
    else:
        raise AssertionError("atomic partial receipt alias was accepted")
    assert partial_store.read_bytes() == b"canonical-store-with-partial-suffix"
