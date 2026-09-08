from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from biohub_ct.campaign.kaggle_rehearsal import PackageIdentity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _canonical_digest(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SyntheticE0Package:
    path: Path
    identity: PackageIdentity
    manifest: dict[str, object]
    artifact_lock: dict[str, object]


def build_e0_package(root: Path, *, generation: str) -> SyntheticE0Package:
    """Build a deterministic package that exercises the real package preflight."""

    package = root.resolve() / f"synthetic-e0-{generation}"
    package.mkdir(parents=True)
    notebook_slug = f"fixture-owner/synthetic-e0-{generation}"
    title = f"Synthetic E0 {generation.upper()} Package"
    competition = "biohub-cell-tracking-during-development"
    dataset_ref = "fixture-owner/synthetic-e0-dependency"
    dataset_version = 1

    public_sources = [f"# synthetic public cell {index}\n" for index in range(12)]
    public_hashes = [hashlib.sha256(source.encode()).hexdigest() for source in public_sources]
    auxiliary_sources = {
        "interceptor_install": "# synthetic interceptor install\n",
        "input_integrity": "# synthetic input integrity\n",
        "base_validation": "# synthetic base validation\n",
        "interceptor_close": "# synthetic interceptor close\n",
    }
    auxiliary_hashes = {
        name: hashlib.sha256(source.encode()).hexdigest()
        for name, source in auxiliary_sources.items()
    }
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [{"output_type": "stream", "name": "stdout", "text": "cleared"}],
                "source": [source],
            }
            for source in public_sources
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = package / "submission.ipynb"
    _write_json(notebook_path, notebook)

    metadata = {
        "id": notebook_slug,
        "title": title,
        "code_file": "submission.ipynb",
        "is_private": True,
        "enable_internet": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "kernel_type": "notebook",
        "language": "python",
        "competition_sources": [competition],
        "dataset_sources": [f"{dataset_ref}/{dataset_version}"],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path = package / "kernel-metadata.json"
    _write_json(metadata_path, metadata)

    reference_notebook = {
        "slug": "fixture-owner/synthetic-public-reference",
        "version": 1,
        "sha256": "1" * 64,
    }
    dataset_record = {
        "dataset_id": dataset_ref,
        "version": dataset_version,
        "last_updated_utc": "2026-01-01T00:00:00Z",
        "license": "synthetic-test-fixture",
        "archive_file": "synthetic.zip",
        "archive_bytes": 3,
        "archive_sha256": hashlib.sha256(b"zip").hexdigest(),
        "file_count": 1,
        "wheel_count": 1,
        "files": {
            "synthetic.whl": {
                "bytes": 3,
                "sha256": hashlib.sha256(b"whl").hexdigest(),
            }
        },
    }
    lock_preimage = {
        "schema_version": 1,
        "competition": competition,
        "reference": {
            "evidence_class": "synthetic_test_fixture",
            "notebook": reference_notebook,
        },
        "datasets": {dataset_ref: dataset_record},
        "instrumentation": {
            "config": {
                "callback_after_public_cell_sha256": hashlib.sha256(
                    b"synthetic callback"
                ).hexdigest()
            }
        },
    }
    release_digest = _canonical_digest(lock_preimage)
    artifact_lock = {**lock_preimage, "release_digest": release_digest}
    lock_path = package / "artifact-lock.json"
    _write_json(lock_path, artifact_lock)

    manifest = {
        "schema_version": 1,
        "status": "ready_for_root_review_not_launched",
        "release_digest": release_digest,
        "kernel_metadata_sha256": _sha256(metadata_path),
        "artifact_lock_sha256": _sha256(lock_path),
        "packaged_notebook": {
            "path": "submission.ipynb",
            "sha256": _sha256(notebook_path),
        },
        "algorithm_changes": [],
        "evidence_class": "synthetic_test_fixture",
        "source_notebook": {
            **reference_notebook,
            "cell_source_sha256": public_hashes,
        },
        "dataset_archives": {dataset_ref: dataset_record},
        "instrumentation": {
            "generated_cell_sha256": auxiliary_hashes,
            "allowed_auxiliary_cell_sha256": list(auxiliary_hashes.values()),
        },
    }
    manifest_path = package / "package-manifest.json"
    _write_json(manifest_path, manifest)

    normalized = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in normalized["cells"]:
        cell["outputs"] = []
        cell["source"] = "".join(cell["source"])
    normalized_hash = hashlib.sha256(json.dumps(normalized).encode()).hexdigest()
    identity = PackageIdentity(
        notebook_slug=notebook_slug,
        notebook_title=title,
        competition=competition,
        machine_shape="NvidiaTeslaT4",
        release_digest=release_digest,
        package_manifest_sha256=_sha256(manifest_path),
        kernel_metadata_sha256=_sha256(metadata_path),
        artifact_lock_sha256=_sha256(lock_path),
        notebook_sha256=_sha256(notebook_path),
        cli_normalized_notebook_sha256=normalized_hash,
        dataset_versions=((dataset_ref, dataset_version),),
    )
    return SyntheticE0Package(package, identity, manifest, artifact_lock)
