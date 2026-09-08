"""Stream frames through an explicit raw-logits adapter into strict E1 caches."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biohub_ct.campaign.detection_diagnostics import (
    cache_manifest_path,
    hash_file,
    hash_json,
    load_logit_cache,
    make_cache_identity,
    save_logit_cache,
    validate_adapter_output,
)
from biohub_ct.data.zarr_io import open_zarr_volume


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be readable JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def require_digest(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def hash_attempt_artifacts(
    root: Path, attempt_dir: Path, artifact_paths: list[Path]
) -> dict[str, str]:
    """Hash the exact unique files owned by one exclusive attempt."""
    root = root.resolve(strict=True)
    attempt_dir = attempt_dir.resolve(strict=True)
    if not attempt_dir.is_relative_to(root):
        raise ValueError("Attempt directory must live inside the artifact root")
    manifest: dict[str, str] = {}
    for candidate in artifact_paths:
        path = candidate.resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(attempt_dir):
            raise ValueError("Every declared artifact must be a file inside the attempt")
        relative = path.relative_to(root).as_posix()
        if relative in manifest:
            raise ValueError("Declared artifacts must be unique")
        manifest[relative] = hash_file(path)
    return manifest


def select_frame_indices(
    volume_frame_count: int, max_frames: int, requested: list[int] | None
) -> list[int]:
    """Select an explicit bounded frame panel or the historical leading prefix."""
    if volume_frame_count < 1 or max_frames < 1:
        raise ValueError("Volume and maximum frame counts must be positive")
    if requested is None:
        return list(range(min(max_frames, volume_frame_count)))
    if not requested:
        raise ValueError("Explicit frame panel cannot be empty")
    if len(requested) > max_frames:
        raise ValueError("Explicit frame panel exceeds max-frames")
    if len(set(requested)) != len(requested):
        raise ValueError("Explicit frame indices must be unique")
    if any(index < 0 or index >= volume_frame_count for index in requested):
        raise ValueError("Explicit frame index is outside the volume")
    return list(requested)


class Progress:
    def __init__(self, path: Path, run_id: str, run_spec_sha256: str) -> None:
        if not path.is_absolute():
            raise ValueError("BIOHUB_PROGRESS_PATH must be absolute")
        self.path = path
        self.run_id = run_id
        self.run_spec_sha256 = require_digest(run_spec_sha256, "run spec identity")
        self.completed_units = 0
        self._lock = threading.Lock()

    def write(self, *, error: str | None = None) -> None:
        with self._lock:
            atomic_json(
                self.path,
                {
                    "run_id": self.run_id,
                    "run_spec_sha256": self.run_spec_sha256,
                    "completed_units": self.completed_units,
                    "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "error": error,
                },
            )

    @contextmanager
    def heartbeat(self):
        stopped = threading.Event()

        def emit() -> None:
            while not stopped.wait(30):
                self.write()

        thread = threading.Thread(target=emit, name="e1-progress-heartbeat", daemon=True)
        thread.start()
        try:
            yield self.write
        finally:
            stopped.set()
            thread.join(timeout=2)


@contextmanager
def enforce_wall_deadline(deadline_at: float):
    """Interrupt on POSIX; adapters also receive the same cooperative deadline."""
    remaining = deadline_at - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("E1 cache wall-time cap reached")
    can_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
    previous_handler = None
    if can_alarm:
        previous_handler = signal.getsignal(signal.SIGALRM)

        def timeout_handler(signum, frame):
            del signum, frame
            raise TimeoutError("E1 cache wall-time cap reached")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        if can_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def load_adapter(specification: str, config: dict[str, Any], *, model_file: Path):
    """Load MODULE:FACTORY and validate either supported cache interface."""
    if specification.count(":") != 1:
        raise ValueError("Adapter must use MODULE:FACTORY form")
    module_name, factory_name = specification.split(":")
    if not module_name or not factory_name:
        raise ValueError("Adapter must use MODULE:FACTORY form")
    factory = getattr(importlib.import_module(module_name), factory_name)
    if "model_file" in inspect.signature(factory).parameters:
        adapter = factory(config, model_file=model_file)
    else:
        adapter = factory(config)
    if not isinstance(getattr(adapter, "name", None), str) or not adapter.name:
        raise TypeError("Adapter must expose a nonempty name")
    generic = callable(getattr(adapter, "raw_logits", None))
    structured = all(
        callable(getattr(adapter, method, None))
        for method in ("capture_payload", "save_cache", "load_cache")
    ) and all(
        isinstance(getattr(adapter, field, None), str) and getattr(adapter, field)
        for field in ("cache_suffix", "output_schema")
    )
    if not generic and not structured:
        raise TypeError("Adapter must implement either raw_logits or structured cache methods")
    return adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zarr", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, action="append", required=True)
    parser.add_argument("--adapter", required=True, help="Explicit MODULE:FACTORY adapter")
    parser.add_argument("--adapter-config-json", type=Path, required=True)
    parser.add_argument("--transform-json", type=Path, required=True)
    parser.add_argument("--precision-json", type=Path, required=True)
    parser.add_argument("--tta-json", type=Path, required=True)
    parser.add_argument("--run-id", default=os.environ.get("BIOHUB_RUN_ID"))
    parser.add_argument(
        "--run-spec-sha256", default=os.environ.get("BIOHUB_RUN_SPEC_SHA256")
    )
    parser.add_argument("--intent-id", default=os.environ.get("BIOHUB_INTENT_ID"))
    parser.add_argument("--fencing-token", default=os.environ.get("BIOHUB_FENCING_TOKEN"))
    parser.add_argument(
        "--attempt-dir", type=Path, default=os.environ.get("BIOHUB_ATTEMPT_DIR")
    )
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument(
        "--frame-index",
        type=int,
        action="append",
        help="Repeat for an explicit frame panel; count remains bounded by --max-frames",
    )
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_id or not args.run_spec_sha256 or not args.intent_id:
        raise ValueError("Run ID, run-spec SHA-256, and intent ID are required")
    try:
        fencing_token = int(args.fencing_token)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fencing token must be an integer") from exc
    if fencing_token < 0 or args.attempt_dir is None:
        raise ValueError("Nonnegative fencing token and attempt directory are required")
    if (
        isinstance(args.max_frames, bool)
        or args.max_frames < 1
        or not math.isfinite(args.max_wall_seconds)
        or args.max_wall_seconds <= 0
    ):
        raise ValueError("Positive max-frames and max-wall-seconds are required")
    if not args.dataset_id or any(character in args.dataset_id for character in "/\\"):
        raise ValueError("Dataset ID must be one safe path component")
    progress_raw = os.environ.get("BIOHUB_PROGRESS_PATH")
    if not progress_raw:
        raise ValueError("BIOHUB_PROGRESS_PATH is required")
    attempt_dir = args.attempt_dir.resolve(strict=True)
    expected_attempt_root = (ROOT / "reports" / "campaign-workers").resolve()
    if not attempt_dir.is_relative_to(expected_attempt_root) or attempt_dir.name != args.run_id:
        raise ValueError("Attempt directory must be inside reports/campaign-workers")
    progress_path = Path(progress_raw).resolve()
    if not progress_path.is_relative_to(attempt_dir):
        raise ValueError("Progress must live inside the exclusive attempt directory")
    progress = Progress(progress_path, args.run_id, args.run_spec_sha256)
    deadline_at = time.monotonic() + args.max_wall_seconds
    progress.write()

    try:
        config = read_mapping(args.adapter_config_json, "adapter config")
        transform = read_mapping(args.transform_json, "transform contract")
        precision = read_mapping(args.precision_json, "precision contract")
        tta = read_mapping(args.tta_json, "TTA contract")
        model_path = args.model_file.resolve(strict=True)
        model_sha256 = hash_file(model_path)
        source_files = [path.resolve(strict=True) for path in args.source_file]
        source_records = [
            {"path": str(path), "sha256": hash_file(path)} for path in source_files
        ]
        source_sha256 = hash_json(
            {"file_sha256": sorted(record["sha256"] for record in source_records)}
        )
        adapter = load_adapter(args.adapter, config, model_file=model_path)
        volume = open_zarr_volume(args.input_zarr, require_complete_chunks=True)
        frame_indices = select_frame_indices(
            int(volume.shape[0]), args.max_frames, args.frame_index
        )
        output = args.output_dir.resolve()
        if not output.is_relative_to(attempt_dir):
            raise ValueError("Output cache must live inside the exclusive attempt directory")
        output.mkdir(parents=True, exist_ok=True)
        index_path = output / "cache-index.json"
        index = {
            "schema_version": 1,
            "status": "running",
            "run_id": args.run_id,
            "run_spec_sha256": args.run_spec_sha256,
            "dataset_id": args.dataset_id,
            "input_zarr": str(args.input_zarr.resolve(strict=True)),
            "input_shape_tzyx": list(volume.shape),
            "scale_zyx_um": list(volume.scale),
            "adapter": adapter.name,
            "output_schema": getattr(
                adapter, "output_schema", "biohub.raw_logits.zyx.float32.v1"
            ),
            "model_file": {"path": str(model_path), "sha256": model_sha256},
            "model_sha256": model_sha256,
            "source_files": source_records,
            "source_sha256": source_sha256,
            "adapter_config": config,
            "config_sha256": hash_json(config),
            "transform": transform,
            "transform_sha256": hash_json(transform),
            "precision": precision,
            "precision_sha256": hash_json(precision),
            "tta": tta,
            "tta_sha256": hash_json(tta),
            "planned_frames": len(frame_indices),
            "planned_frame_indices": frame_indices,
            "frames": [],
        }
        atomic_json(index_path, index)

        with enforce_wall_deadline(deadline_at):
            for frame_index in frame_indices:
                if time.monotonic() >= deadline_at:
                    raise TimeoutError("E1 cache wall-time cap reached before next frame")
                progress.write()
                frame = volume.read_frame(frame_index)
                identity = make_cache_identity(
                    model_sha256=model_sha256,
                    source_sha256=source_sha256,
                    config=config,
                    input_frame=frame,
                    transform=transform,
                    precision=precision,
                    tta=tta,
                    output_schema=getattr(
                        adapter, "output_schema", "biohub.raw_logits.zyx.float32.v1"
                    ),
                )
                suffix = getattr(adapter, "cache_suffix", ".npy")
                logits_path = output / f"{args.dataset_id}-frame-{frame_index:05d}{suffix}"
                if logits_path.exists() or cache_manifest_path(logits_path).exists():
                    if callable(getattr(adapter, "load_cache", None)):
                        payload, manifest = adapter.load_cache(logits_path, identity)
                    else:
                        payload, manifest = load_logit_cache(
                            logits_path, expected_identity=identity
                        )
                else:
                    with progress.heartbeat() as heartbeat:
                        if callable(getattr(adapter, "capture_payload", None)):
                            payload = adapter.capture_payload(
                                frame, deadline_at=deadline_at, progress=heartbeat
                            )
                        else:
                            payload = validate_adapter_output(
                                adapter.raw_logits(
                                    frame, deadline_at=deadline_at, progress=heartbeat
                                )
                            )
                    if callable(getattr(adapter, "save_cache", None)):
                        manifest = adapter.save_cache(logits_path, payload, identity)
                    else:
                        manifest = save_logit_cache(
                            logits_path, payload, identity, adapter=adapter.name
                        )
                frame_record = {
                    "frame": frame_index,
                    "input_frame_sha256": identity.input_frame_sha256,
                    "identity_sha256": identity.sha256,
                    "logits_path": logits_path.name,
                    "manifest_path": cache_manifest_path(logits_path).name,
                    "logits_file_sha256": manifest["file_sha256"],
                    "manifest_file_sha256": hash_file(cache_manifest_path(logits_path)),
                }
                if isinstance(payload, np.ndarray):
                    frame_record["logits_shape_zyx"] = list(payload.shape)
                else:
                    frame_record["tile_logits_shape_nzyx"] = list(payload.logits.shape)
                    frame_record["normalized_shape_zyx"] = list(payload.normalized_shape_zyx)
                index["frames"].append(frame_record)
                atomic_json(index_path, index)
                progress.completed_units += 1
                progress.write()

        index["status"] = "complete"
        index["completed_frames"] = len(index["frames"])
        index["elapsed_seconds"] = args.max_wall_seconds - max(
            0.0, deadline_at - time.monotonic()
        )
        atomic_json(index_path, index)
        cache_reference_path = attempt_dir / "cache-reference.json"
        cache_reference = {
            "schema_version": 1,
            "run_id": args.run_id,
            "run_spec_sha256": args.run_spec_sha256,
            "intent_id": args.intent_id,
            "fencing_token": fencing_token,
            "cache_index": {
                "absolute_path": str(index_path),
                "sha256": hash_file(index_path),
            },
            "cache_root": str(output),
            "cache_identity_sha256": hash_json(
                [frame["identity_sha256"] for frame in index["frames"]]
            ),
        }
        atomic_json(cache_reference_path, cache_reference)
        artifact_paths = [index_path, cache_reference_path, progress_path]
        for frame in index["frames"]:
            artifact_paths.extend(
                [output / frame["logits_path"], output / frame["manifest_path"]]
            )
        result = {
            "run_id": args.run_id,
            "run_spec_sha256": args.run_spec_sha256,
            "intent_id": args.intent_id,
            "fencing_token": fencing_token,
            "status": "COMPLETE",
            "completed_units": progress.completed_units,
            "artifact_sha256": hash_attempt_artifacts(
                ROOT, attempt_dir, artifact_paths
            ),
        }
        atomic_json(attempt_dir / "result.json", result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    except BaseException as exc:
        progress.write(error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
