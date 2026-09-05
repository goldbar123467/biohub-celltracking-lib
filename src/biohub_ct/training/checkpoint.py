from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device topology differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(directory: Path | str, state: dict, *, is_best: bool = False) -> Path:
    """Commit an immutable checkpoint, then atomically update recovery pointers.

    A crash before manifest replacement leaves the previous committed latest valid.
    Orphan files from a failed commit are harmless and are never auto-resumed.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {**state, "format_version": FORMAT_VERSION}
    filename = f"step-{int(state['step']):09d}-{uuid.uuid4().hex[:10]}.pt"
    path = directory / filename
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix="checkpoint-", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with temporary_path.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        checksum = _sha256(path)
        atomic_json(path.with_suffix(".sha256.json"), {"sha256": checksum, "file": filename})
        manifest_path = directory / "manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        pointer = {"file": filename, "sha256": checksum, "step": int(state["step"])}
        atomic_json(
            manifest_path,
            {
                "format_version": FORMAT_VERSION,
                "latest": pointer,
                "previous": previous.get("latest"),
                "best": pointer if is_best else previous.get("best"),
            },
        )
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_checkpoint(path: Path | str, *, expected_identity: dict | None = None) -> dict:
    """Load trusted local training state after hash and optional identity checks."""
    path = Path(path)
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        pointer = manifest["latest"]
        filename = pointer["file"]
        if Path(filename).name != filename:
            raise ValueError("Invalid checkpoint manifest path")
        path = path / filename
        expected_hash = pointer["sha256"]
    else:
        expected_hash = json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
    if _sha256(path) != expected_hash:
        raise ValueError("Checkpoint SHA256 mismatch")
    # Training checkpoints contain Python/NumPy RNG state and are trusted local artifacts.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    if expected_identity is not None and state.get("identity") != expected_identity:
        raise ValueError("Checkpoint identity mismatch")
    return state
