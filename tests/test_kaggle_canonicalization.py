from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from biohub_ct.campaign.kaggle_canonicalization import (
    CanonicalizationError,
    CanonicalizationEvidence,
    create_canonicalization_receipt,
    verify_canonicalization_receipt,
)
from biohub_ct.campaign.kaggle_rehearsal import E0_R3_PACKAGE_IDENTITY

ROOT = Path(__file__).resolve().parents[1]
LIVE_EVIDENCE = ROOT / "reports" / "campaigns" / "e0-r3-kaggle-full-20260908-01"
ACTUAL_SLUG = "clarkkitchen/biohub-e0-public-reference-reproduction"
SOURCE_SHA256 = "fa6dcdff02c1b08a16327ce42da6b953ccfdcc20a75184f178994b73f4b9f196"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _copy_live_evidence(tmp_path: Path) -> CanonicalizationEvidence:
    required = (
        LIVE_EVIDENCE / "launch-receipt.json",
        LIVE_EVIDENCE / "recovered-push-stdout.txt",
        LIVE_EVIDENCE / "sdk-v1" / "receipt.json",
        LIVE_EVIDENCE / "sdk-v1" / "source.ipynb",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("private persisted R3 canonicalization evidence is unavailable")
    launch = tmp_path / "launch-receipt.json"
    transcript = tmp_path / "recovered-push-stdout.txt"
    sdk_receipt = tmp_path / "sdk-receipt.json"
    sdk_source = tmp_path / "source.ipynb"
    shutil.copyfile(LIVE_EVIDENCE / "launch-receipt.json", launch)
    shutil.copyfile(LIVE_EVIDENCE / "recovered-push-stdout.txt", transcript)
    shutil.copyfile(LIVE_EVIDENCE / "sdk-v1" / "receipt.json", sdk_receipt)
    shutil.copyfile(LIVE_EVIDENCE / "sdk-v1" / "source.ipynb", sdk_source)
    return CanonicalizationEvidence(
        launch_receipt_path=launch,
        recovered_push_stdout_path=transcript,
        sdk_receipt_path=sdk_receipt,
        sdk_source_path=sdk_source,
    )


def test_portable_missing_bound_artifact_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    evidence = CanonicalizationEvidence(missing, missing, missing, missing)

    with pytest.raises(CanonicalizationError, match="launch receipt is unavailable"):
        create_canonicalization_receipt(evidence, tmp_path / "mapping.json")

    assert not (tmp_path / "mapping.json").exists()


def test_portable_symlinked_bound_artifact_fails_before_content(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "launch-link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    evidence = CanonicalizationEvidence(link, target, target, target)

    with pytest.raises(CanonicalizationError, match="links or junctions"):
        create_canonicalization_receipt(evidence, tmp_path / "mapping.json")


def test_portable_offline_verifier_rejects_nonobject_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "mapping.json"
    receipt.write_text("[]\n", encoding="utf-8")

    with pytest.raises(CanonicalizationError, match="must contain a JSON object"):
        verify_canonicalization_receipt(receipt)


def _set_nested(value: dict[str, object], path: tuple[str, ...], replacement: object) -> None:
    current: object = value
    for key in path[:-1]:
        assert isinstance(current, dict)
        current = current[key]
    assert isinstance(current, dict)
    current[path[-1]] = replacement


def test_live_evidence_creates_and_revalidates_exact_saved_mapping_receipt(
    tmp_path: Path,
) -> None:
    evidence = _copy_live_evidence(tmp_path)
    receipt_path = tmp_path / "canonicalization-receipt.json"

    created = create_canonicalization_receipt(evidence, receipt_path)

    assert created.actual_notebook_slug == ACTUAL_SLUG
    assert created.notebook_version == 1
    assert created.source_sha256 == SOURCE_SHA256
    assert (
        created.cli_normalized_source_sha256
        == E0_R3_PACKAGE_IDENTITY.cli_normalized_notebook_sha256
    )
    assert created.receipt_path == receipt_path.resolve()
    assert created.receipt_sha256 == _sha256(receipt_path)

    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["kind"] == "e0_r3_kaggle_title_canonicalization_proof"
    assert saved["status"] == "PASS"
    assert saved["mapping"] == {
        "requested_notebook_slug": E0_R3_PACKAGE_IDENTITY.notebook_slug,
        "actual_notebook_slug": ACTUAL_SLUG,
        "notebook_version": 1,
        "result_url": f"https://www.kaggle.com/code/{ACTUAL_SLUG}",
        "cause": "KAGGLE_TITLE_CANONICALIZATION",
    }
    assert saved["constraints"] == {
        "read_only": True,
        "provider_mutations": [],
        "requested_identity_preserved": True,
    }
    assert saved["original_intent"]["intent_id"] == ("e0-r3-kaggle-full-20260908-01-intent")
    assert saved["push_transcript"]["sha256"] == _sha256(evidence.recovered_push_stdout_path)
    assert saved["sdk_exact_source_read"]["receipt"]["sha256"] == _sha256(evidence.sdk_receipt_path)
    assert saved["sdk_exact_source_read"]["source"]["sha256"] == SOURCE_SHA256
    assert verify_canonicalization_receipt(receipt_path) == created


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("request", "kernel_slug"), "biohub-e0-public-reference"),
        (("request", "version_label"), "v2"),
        (("metadata", "is_private"), False),
        (("metadata", "enable_gpu"), False),
        (("metadata", "enable_internet"), True),
        (("metadata", "enable_tpu"), True),
    ],
)
def test_creation_rejects_wrong_sdk_slug_version_or_execution_flags(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
) -> None:
    evidence = _copy_live_evidence(tmp_path)
    sdk_receipt = json.loads(evidence.sdk_receipt_path.read_text(encoding="utf-8"))
    _set_nested(sdk_receipt, path, replacement)
    _write_json(evidence.sdk_receipt_path, sdk_receipt)

    with pytest.raises(CanonicalizationError):
        create_canonicalization_receipt(evidence, tmp_path / "mapping.json")

    assert not (tmp_path / "mapping.json").exists()


def test_creation_rejects_source_bytes_that_differ_from_reviewed_sdk_receipt(
    tmp_path: Path,
) -> None:
    evidence = _copy_live_evidence(tmp_path)
    notebook = json.loads(evidence.sdk_source_path.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] += "\n# source drift\n"
    _write_json(evidence.sdk_source_path, notebook)

    with pytest.raises(CanonicalizationError, match="do not match the SDK receipt"):
        create_canonicalization_receipt(evidence, tmp_path / "mapping.json")


def test_creation_rejects_changed_recovered_transcript(tmp_path: Path) -> None:
    evidence = _copy_live_evidence(tmp_path)
    evidence.recovered_push_stdout_path.write_bytes(
        evidence.recovered_push_stdout_path.read_bytes() + b"unexpected\n"
    )

    with pytest.raises(CanonicalizationError, match="original push stdout"):
        create_canonicalization_receipt(evidence, tmp_path / "mapping.json")


def test_creation_rejects_changed_original_intent_receipt(tmp_path: Path) -> None:
    evidence = _copy_live_evidence(tmp_path)
    launch = json.loads(evidence.launch_receipt_path.read_text(encoding="utf-8"))
    launch["launch_result"]["intent_id"] = "different-intent"
    _write_json(evidence.launch_receipt_path, launch)

    with pytest.raises(CanonicalizationError, match="persisted R3 mutation"):
        create_canonicalization_receipt(evidence, tmp_path / "mapping.json")


def test_creation_refuses_to_overwrite_existing_receipt(tmp_path: Path) -> None:
    evidence = _copy_live_evidence(tmp_path)
    receipt_path = tmp_path / "mapping.json"
    receipt_path.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(CanonicalizationError, match="already exists"):
        create_canonicalization_receipt(evidence, receipt_path)

    assert receipt_path.read_text(encoding="utf-8") == "preserve me\n"


def test_creation_rejects_evidence_reached_through_symlink(tmp_path: Path) -> None:
    evidence = _copy_live_evidence(tmp_path)
    transcript_target = evidence.recovered_push_stdout_path
    transcript_link = tmp_path / "transcript-link.txt"
    try:
        transcript_link.symlink_to(transcript_target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    linked = CanonicalizationEvidence(
        launch_receipt_path=evidence.launch_receipt_path,
        recovered_push_stdout_path=transcript_link,
        sdk_receipt_path=evidence.sdk_receipt_path,
        sdk_source_path=evidence.sdk_source_path,
    )

    with pytest.raises(CanonicalizationError, match="links or junctions"):
        create_canonicalization_receipt(linked, tmp_path / "mapping.json")


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("mapping", "actual_notebook_slug"), "clarkkitchen/wrong-slug"),
        (("mapping", "notebook_version"), 2),
        (("sdk_exact_source_read", "source", "cli_normalized_sha256"), "0" * 64),
        (("sdk_exact_source_read", "metadata", "enable_gpu"), False),
        (("push_transcript", "stdout_sha256"), "0" * 64),
        (("original_intent", "intent_id"), "different-intent"),
    ],
)
def test_offline_verifier_rejects_tampered_saved_receipt_fields(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: object,
) -> None:
    evidence = _copy_live_evidence(tmp_path)
    receipt_path = tmp_path / "canonicalization-receipt.json"
    create_canonicalization_receipt(evidence, receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _set_nested(receipt, path, replacement)
    _write_json(receipt_path, receipt)

    with pytest.raises(CanonicalizationError):
        verify_canonicalization_receipt(receipt_path)
