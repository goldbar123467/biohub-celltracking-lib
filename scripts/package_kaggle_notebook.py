#!/usr/bin/env python
"""Build a deterministic, self-contained Kaggle inference notebook from reviewed source."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL_MANIFEST_SHA256 = "2c951e4c2afbb76b11b8d9f8007d26f8e5ddaf5086eadefb55256c79a067a84c"


def source_archive(root=ROOT):
    buffer = io.BytesIO()
    files = sorted((root / "src" / "biohub_ct").rglob("*.py"))
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(root / "src").as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Stable ZIP headers on Windows and Linux.
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes().replace(b"\r\n", b"\n"))
    return buffer.getvalue()


def build(output_dir, kernel_id, *, config=None, deadline_seconds=32400):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = source_archive()
    sha = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode()
    bootstrap = f"""import base64, hashlib, io, json, sys, zipfile
from pathlib import Path
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {sha!r}, 'Source archive checksum mismatch'
source = Path('/kaggle/working/biohub-source')
source.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for item in archive.infolist():
        target = (source / item.filename).resolve()
        assert target.is_relative_to(source.resolve()) and item.filename.endswith('.py'), 'Unsafe archive member'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(item))
sys.path.insert(0, str(source))
print('Source archive SHA256:', {sha!r})
"""
    # Reuse the independently executed offline wheel verification procedure.
    readiness = (ROOT / "notebooks/kaggle-readiness/readiness.py").read_text()
    wheel_setup = readiness[
        readiness.index("manifests =") : readiness.index("\nimport numpy as np")
    ]
    wheel_setup = wheel_setup.replace(
        "manifest_path = manifests[0]",
        "manifest_path = manifests[0]\n"
        f'assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == {WHEEL_MANIFEST_SHA256!r}, "Unexpected dependency bundle"',
    )
    dependencies = "import subprocess, sys, hashlib, json\nfrom pathlib import Path\n" + wheel_setup
    inference = f"""import signal, time
from biohub_ct.pipelines.baseline_classical import ClassicalConfig
from biohub_ct.pipelines.submission_pipeline import run_submission_pipeline
from biohub_ct.submission.validator import validate_submission

def timeout_handler(signum, frame):
    raise TimeoutError('Kaggle inference runtime limit reached')
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({int(deadline_seconds)})
slug = 'biohub-cell-tracking-during-development'
roots = [p for p in (Path('/kaggle/input/competitions') / slug, Path('/kaggle/input') / slug) if (p / 'test').is_dir()]
assert len(roots) == 1, f'Expected one competition test directory: {{roots}}'
config = ClassicalConfig(**{config or {}!r})
result = run_submission_pipeline(data_dir=roots[0] / 'test', output_path='/kaggle/working/submission.csv',
                                 config=config, deadline_seconds={int(deadline_seconds)})
print(validate_submission(result))
signal.alarm(0)
"""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Biohub offline submission rehearsal\n",
                "Classical baseline. Public example tests are training copies, so this run measures execution, not validation quality.\n",
            ],
        }
    ]
    for code in (bootstrap, dependencies, inference):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
        )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    (output / "submission.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    metadata = {
        "id": kernel_id,
        "title": "Biohub Submission Rehearsal",
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": ["clarkkitchen/biohub-offline-dependencies"],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    manifest = {
        "dependency_manifest_sha256": WHEEL_MANIFEST_SHA256,
        "source_archive_sha256": sha,
        "source_bytes": len(payload),
        "config": config or {},
        "notebook_sha256": hashlib.sha256((output / "submission.ipynb").read_bytes()).hexdigest(),
    }
    (output / "package-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--config")
    parser.add_argument("--deadline-seconds", type=int, default=32400)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else None
    print(
        json.dumps(
            build(
                args.output_dir,
                args.kernel_id,
                config=config,
                deadline_seconds=args.deadline_seconds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
