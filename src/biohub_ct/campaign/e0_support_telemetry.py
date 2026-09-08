"""Build exact, additive telemetry for the pinned E0 support predictor."""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass

PRE_PUBLIC_SUPPORT_SHA256 = "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9"
PUBLIC_PATCHED_SUPPORT_SHA256 = "49613ad0b50ac90c3e07e3f8a803f2adf0926203be3f4a97d577c60b8f756178"
RUNTIME_MODULE_NAME = "_biohub_e0_support_telemetry_runtime"
SUPPORT_TELEMETRY_MARKER = "BIOHUB_E0_SUPPORT_TELEMETRY_V1"


@dataclass(frozen=True)
class PatchedSupportSource:
    """A compile-checked support source and its provenance chain."""

    source: str
    pre_public_sha256: str
    public_patched_input_sha256: str
    telemetry_output_sha256: str
    runtime_module_name: str
    runtime_module_sha256: str
    anchor_counts: tuple[tuple[str, int], ...]

    def source_chain(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "BIOHUB_E0_SUPPORT_TELEMETRY_SOURCE_CHAIN",
            "pre_public_support_sha256": self.pre_public_sha256,
            "public_patched_support_sha256": self.public_patched_input_sha256,
            "telemetry_output_support_sha256": self.telemetry_output_sha256,
            "runtime_module_name": self.runtime_module_name,
            "runtime_module_sha256": self.runtime_module_sha256,
            "anchor_counts": dict(self.anchor_counts),
        }


def _sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def runtime_helper_source() -> str:
    """Return the dependency-free helper to place beside the support script."""

    return textwrap.dedent(
        r'''
        """Runtime telemetry helper materialized by the E0 instrumented wrapper."""

        import atexit
        import hashlib
        import json
        import math
        import os
        import re
        import sys
        import threading
        import time
        from pathlib import Path


        _STAGES = (
            "data_read",
            "encode_tta",
            "detector_extraction",
            "pair_score",
            "threshold",
            "graph_build",
            "ilp",
            "geff",
        )
        _CUDA_STAGES = {"encode_tta", "detector_extraction", "pair_score", "threshold"}


        def _safe_component(value):
            text = str(value).strip()
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
            return safe[:80] or "unavailable"


        class Telemetry:
            def __init__(self, directory=None, clock_ns=None):
                configured = (
                    str(directory)
                    if directory is not None
                    else os.environ.get("BIOHUB_E0_TELEMETRY_DIR", "").strip()
                )
                self.enabled = bool(configured)
                self.root = Path(configured) if configured else None
                self.pid = os.getpid()
                self.process_started_ns = time.time_ns()
                self.process_id = f"{self.pid}-{self.process_started_ns}"
                self.process_dir = (
                    self.root / f"invocation-{self.process_id}" if self.root else None
                )
                self.event_path = self.process_dir / "events.jsonl" if self.process_dir else None
                self.clock_ns = clock_ns or time.perf_counter_ns
                self.shard = os.environ.get("BIOHUB_GPU_SHARD", "unavailable").strip() or "unavailable"
                self.diagnostic_arm = (
                    os.environ.get("BIOHUB_DIAGNOSTIC_ARM", "").strip() or "production"
                )
                self._lock = threading.RLock()
                self._torch = None
                self._cuda_reset = False
                self._current = None
                self._invocation = None
                self._invocation_ids = []
                self._invocation_sequence = 0
                self._finalized = False
                if self.process_dir:
                    self.process_dir.mkdir(parents=True, exist_ok=False)

            def attach_torch(self, torch_module):
                self._torch = torch_module
                if not self.enabled:
                    return
                try:
                    if bool(torch_module.cuda.is_available()):
                        torch_module.cuda.reset_peak_memory_stats()
                        self._cuda_reset = True
                except Exception:
                    self._cuda_reset = False

            def _synchronize(self, stage):
                if (
                    self.enabled
                    and stage in _CUDA_STAGES
                    and self._torch is not None
                    and bool(self._torch.cuda.is_available())
                ):
                    self._torch.cuda.synchronize()

            def started(self, stage=None):
                self._synchronize(stage)
                return self.clock_ns()

            def add_duration(self, stage, started_ns):
                if not self.enabled:
                    return
                if stage not in _STAGES:
                    raise ValueError(f"Unknown telemetry stage: {stage}")
                self._synchronize(stage)
                elapsed_ns = self.clock_ns() - int(started_ns)
                if elapsed_ns < 0:
                    raise RuntimeError("Telemetry monotonic clock moved backwards")
                with self._lock:
                    current = self._require_dataset()
                    current["duration_ns"][stage] += elapsed_ns
                    current["measured_stages"].add(stage)

            def begin_invocation(self, *, data_root, output_dir, method, fold, test_names):
                if not self.enabled:
                    return
                with self._lock:
                    if self._current is not None or self._invocation is not None:
                        raise RuntimeError("Telemetry invocation overlap")
                    self._invocation_sequence += 1
                    invocation_id = f"{self.process_id}-{self._invocation_sequence}"
                    self._invocation = {
                        "invocation_id": invocation_id,
                        "data_root": str(Path(data_root).resolve()),
                        "output_dir": str(Path(output_dir).resolve()),
                        "method": str(method),
                        "fold": int(fold),
                        "test_names": [str(name) for name in test_names],
                    }
                    self._invocation_ids.append(invocation_id)
                    self._write("invocation_start", {"identity": dict(self._invocation)})

            def finish_invocation(self):
                if not self.enabled:
                    return
                with self._lock:
                    if self._current is not None:
                        raise RuntimeError("Cannot finish invocation with an active dataset")
                    invocation = self._require_invocation()
                    self._write(
                        "invocation_finish",
                        {"identity": dict(invocation), "outcome": "returned"},
                    )
                    self._invocation = None

            def begin_dataset(self, dataset, dataset_path):
                if not self.enabled:
                    return
                with self._lock:
                    if self._current is not None:
                        raise RuntimeError("Telemetry dataset overlap")
                    invocation = self._require_invocation()
                    self._current = {
                        "dataset": str(dataset),
                        "dataset_path": str(Path(dataset_path).resolve()),
                        "invocation_id": invocation["invocation_id"],
                        "duration_ns": {stage: 0 for stage in _STAGES},
                        "measured_stages": set(),
                        "pair_universe": 0,
                        "threshold_passing_edge_candidates": 0,
                        "ilp": {"status": "unavailable", "reason": "not_reached"},
                        "coordinate_artifact": {
                            "status": "unavailable",
                            "reason": "not_reached",
                        },
                    }

            def add_pair_counts(self, *, pair_universe, threshold_passing):
                if not self.enabled:
                    return
                universe = int(pair_universe)
                passing = int(threshold_passing)
                if universe < 0 or passing < 0 or passing > universe:
                    raise ValueError("Invalid edge-candidate telemetry counts")
                with self._lock:
                    current = self._require_dataset()
                    current["pair_universe"] += universe
                    current["threshold_passing_edge_candidates"] += passing

            def set_ilp(self, status, *, reason=None, exception_type=None):
                if not self.enabled:
                    return
                if status not in {"returned", "raised", "unavailable"}:
                    raise ValueError(f"Invalid ILP telemetry status: {status}")
                record = {"status": status}
                if reason is not None:
                    record["reason"] = str(reason)
                if exception_type is not None:
                    record["exception_type"] = str(exception_type)
                with self._lock:
                    self._require_dataset()["ilp"] = record

            def record_retention(self, record):
                if not self.enabled:
                    return
                if not isinstance(record, dict):
                    raise TypeError("Retention telemetry record must be a dictionary")
                with self._lock:
                    current = self._require_dataset()
                    if record.get("dataset") != current["dataset"]:
                        raise RuntimeError("Retention dataset does not match active telemetry dataset")
                    payload = dict(record)
                    payload["dataset_path"] = current["dataset_path"]
                    self._write("retention", payload)

            def capture_coordinates(self, *, dataset, payload, shape, dtype):
                if not self.enabled:
                    return
                raw = bytes(payload)
                normalized_shape = [int(value) for value in shape]
                if dtype != "<i2" or len(normalized_shape) != 2 or normalized_shape[1] != 4:
                    raise ValueError("Coordinates must have shape (N, 4) and dtype <i2")
                if len(raw) != math.prod(normalized_shape) * 2:
                    raise ValueError("Coordinate payload size does not match shape and dtype")
                with self._lock:
                    current = self._require_dataset()
                    if current["dataset"] != str(dataset):
                        raise RuntimeError("Coordinate dataset does not match active telemetry dataset")
                    digest = hashlib.sha256(raw).hexdigest()
                    safe = _safe_component(dataset)
                    name_hash = hashlib.sha256(str(dataset).encode("utf-8")).hexdigest()[:12]
                    invocation_component = _safe_component(current["invocation_id"])
                    relative = (
                        Path("coordinates")
                        / invocation_component
                        / f"{safe}-{name_hash}.i16le.bin"
                    )
                    destination = self.process_dir / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".partial")
                    if destination.exists() or temporary.exists():
                        raise FileExistsError(
                            f"Coordinate artifact already exists for this invocation: {destination}"
                        )
                    with temporary.open("wb") as handle:
                        handle.write(raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, destination)
                    frame_counts = {}
                    for offset in range(0, len(raw), 8):
                        frame = int.from_bytes(raw[offset : offset + 2], "little", signed=True)
                        frame_counts[frame] = frame_counts.get(frame, 0) + 1
                    artifact_path = (Path(self.root.name) / destination.relative_to(self.root)).as_posix()
                    artifact = {
                        "path": artifact_path,
                        "bytes": len(raw),
                        "dtype": dtype,
                        "shape": normalized_shape,
                        "sha256": digest,
                    }
                    coordinate_record = {
                        "dataset": current["dataset"],
                        "dataset_path": current["dataset_path"],
                        "stage": "post_detection_pre_graph_pre_ilp",
                        "columns": ["t", "z", "y", "x"],
                        "dtype": dtype,
                        "rows": normalized_shape[0],
                        "coordinate_sha256": digest,
                        "frame_counts": [
                            [frame, frame_counts[frame]] for frame in sorted(frame_counts)
                        ],
                        "artifact": artifact,
                        "coordinate_space": "original_voxel_zyx",
                    }
                    current["coordinate_artifact"] = {
                        "status": "available",
                        **coordinate_record,
                    }
                    self._write("coordinate_artifact", coordinate_record)
                    return artifact

            def finish_dataset(self, *, detected_nodes, pre_ilp_edges, output_edges):
                if not self.enabled:
                    return
                with self._lock:
                    current = self._require_dataset()
                    self._write_dataset(
                        current,
                        outcome="returned",
                        detected_nodes=int(detected_nodes),
                        pre_ilp_edges=int(pre_ilp_edges),
                        output_edges=int(output_edges),
                    )
                    self._current = None

            def _duration_records(self, current):
                result = {}
                for stage in _STAGES:
                    if stage in current["measured_stages"]:
                        result[stage] = {
                            "status": "available",
                            "value": current["duration_ns"][stage] / 1_000_000_000,
                            "unit": "seconds",
                            "duration_ns": current["duration_ns"][stage],
                        }
                    else:
                        result[stage] = {"status": "unavailable", "reason": "not_reached"}
                return result

            def _write_dataset(self, current, *, outcome, **counts):
                payload = {
                    "dataset": current["dataset"],
                    "dataset_path": current["dataset_path"],
                    "outcome": outcome,
                    "durations": self._duration_records(current),
                    "counts": {
                        "pair_universe": {
                            "status": "available",
                            "value": current["pair_universe"],
                            "unit": "ordered_source_target_pairs",
                        },
                        "threshold_passing_edge_candidates": {
                            "status": "available",
                            "value": current["threshold_passing_edge_candidates"],
                            "unit": "edges",
                        },
                    },
                    "ilp": dict(current["ilp"]),
                    "coordinate_artifact": dict(current["coordinate_artifact"]),
                }
                for name, value in counts.items():
                    payload["counts"][name] = {
                        "status": "available",
                        "value": value,
                        "unit": "nodes" if name == "detected_nodes" else "edges",
                    }
                self._write("dataset_summary", payload)

            def _require_dataset(self):
                if self._current is None:
                    raise RuntimeError("No active telemetry dataset")
                return self._current

            def _require_invocation(self):
                if self._invocation is None:
                    raise RuntimeError("No active telemetry invocation")
                return self._invocation

            def _base_record(self, record_type):
                return {
                    "schema_version": 1,
                    "record_type": record_type,
                    "process_id": self.process_id,
                    "pid": self.pid,
                    "gpu_shard": self.shard,
                    "diagnostic_arm": self.diagnostic_arm,
                    "invocation_id": (
                        self._invocation["invocation_id"] if self._invocation else None
                    ),
                }

            def _write(self, record_type, payload):
                record = self._base_record(record_type)
                record.update(payload)
                encoded = json.dumps(
                    record, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())

            @staticmethod
            def _unavailable(reason):
                return {"status": "unavailable", "reason": reason}

            def _cuda_peaks(self):
                if self._torch is None:
                    unavailable = self._unavailable("torch_not_attached")
                    return unavailable, unavailable
                try:
                    if not bool(self._torch.cuda.is_available()):
                        unavailable = self._unavailable("cuda_unavailable")
                        return unavailable, unavailable
                    if not self._cuda_reset:
                        unavailable = self._unavailable("peak_reset_failed")
                        return unavailable, unavailable
                    allocated = int(self._torch.cuda.max_memory_allocated())
                    reserved = int(self._torch.cuda.max_memory_reserved())
                    return (
                        {"status": "available", "value": allocated, "unit": "bytes"},
                        {"status": "available", "value": reserved, "unit": "bytes"},
                    )
                except Exception as exc:
                    unavailable = self._unavailable(
                        f"cuda_query_failed:{type(exc).__name__}"
                    )
                    return unavailable, unavailable

            def _host_peak(self):
                try:
                    import resource

                    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    if sys.platform == "darwin":
                        unit = "bytes"
                        normalized_bytes = raw
                    else:
                        unit = "kibibytes"
                        normalized_bytes = raw * 1024
                    return {
                        "status": "available",
                        "value": raw,
                        "unit": unit,
                        "normalized_bytes": normalized_bytes,
                    }
                except Exception as exc:
                    return self._unavailable(
                        f"resource_ru_maxrss_unavailable:{type(exc).__name__}"
                    )

            def finalize_process(self):
                if not self.enabled:
                    return
                with self._lock:
                    if self._finalized:
                        return
                    if self._current is not None:
                        self._write_dataset(self._current, outcome="incomplete_process_exit")
                        self._current = None
                    if self._invocation is not None:
                        self._write(
                            "invocation_finish",
                            {
                                "identity": dict(self._invocation),
                                "outcome": "incomplete_process_exit",
                            },
                        )
                        self._invocation = None
                    allocated, reserved = self._cuda_peaks()
                    self._write(
                        "process_summary",
                        {
                            "invocation_ids": list(self._invocation_ids),
                            "cuda_peak_allocated": allocated,
                            "cuda_peak_reserved": reserved,
                            "host_ru_maxrss": self._host_peak(),
                        },
                    )
                    self._finalized = True


        telemetry = Telemetry()
        atexit.register(telemetry.finalize_process)
        '''
    ).lstrip()


_SUPPORT_HELPERS = f"""\

# {SUPPORT_TELEMETRY_MARKER}
from {RUNTIME_MODULE_NAME} import telemetry as _e0_telemetry

_e0_telemetry.attach_torch(torch)


def _e0_encode(model, *args, **kwargs):
    _e0_started = _e0_telemetry.started("encode_tta")
    try:
        return model.encode(*args, **kwargs)
    finally:
        _e0_telemetry.add_duration("encode_tta", _e0_started)


def _e0_predict_edges(model, *args, **kwargs):
    _e0_started = _e0_telemetry.started("pair_score")
    try:
        return model.predict_edges(*args, **kwargs)
    finally:
        _e0_telemetry.add_duration("pair_score", _e0_started)


def _e0_build_graph(coords, edges):
    _e0_started = _e0_telemetry.started("graph_build")
    try:
        return build_graph(coords, edges)
    finally:
        _e0_telemetry.add_duration("graph_build", _e0_started)


def _e0_solve(solver, graph):
    _e0_started = _e0_telemetry.started("ilp")
    try:
        result = solver.solve(graph)
    except BaseException as _e0_error:
        _e0_telemetry.add_duration("ilp", _e0_started)
        _e0_telemetry.set_ilp("raised", exception_type=type(_e0_error).__name__)
        raise
    _e0_telemetry.add_duration("ilp", _e0_started)
    _e0_telemetry.set_ilp("returned")
    return result


def _e0_save_graph(graph, path):
    _e0_started = _e0_telemetry.started("geff")
    try:
        return save_graph(graph, path)
    finally:
        _e0_telemetry.add_duration("geff", _e0_started)
"""


def _replace_once(source: str, old: str, new: str, label: str) -> tuple[str, tuple[str, int]]:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"E0 telemetry anchor {label!r} expected one match, found {count}")
    return source.replace(old, new, 1), (label, count)


def patch_support_source(
    source: str,
    *,
    expected_public_patched_sha256: str = PUBLIC_PATCHED_SUPPORT_SHA256,
) -> PatchedSupportSource:
    """Instrument the exact support source after all pinned public patches."""

    input_sha256 = _sha256_text(source)
    if input_sha256 != expected_public_patched_sha256:
        raise ValueError(
            "Public-patched support SHA-256 mismatch: "
            f"expected {expected_public_patched_sha256}, got {input_sha256}"
        )

    anchors: list[tuple[str, int]] = []

    def replace(old: str, new: str, label: str) -> None:
        nonlocal source
        source, count = _replace_once(source, old, new, label)
        anchors.append(count)

    replace(
        "from biohub_tracking.metrics import summarise\n",
        "from biohub_tracking.metrics import summarise\n" + _SUPPORT_HELPERS,
        "runtime_import_and_wrappers",
    )
    replace(
        "def _load_frame(\n",
        "def _e0_uninstrumented_load_frame(\n",
        "rename_load_frame",
    )
    replace(
        "    return frame\n\n\n# =============================================================================\n"
        "# Inference\n",
        """    return frame


def _load_frame(zarr_arr, t, target_shape, downsample=(1, 1, 1)):
    _e0_started = _e0_telemetry.started("data_read")
    try:
        return _e0_uninstrumented_load_frame(zarr_arr, t, target_shape, downsample)
    finally:
        _e0_telemetry.add_duration("data_read", _e0_started)


# =============================================================================
# Inference
""",
        "load_frame_wrapper",
    )
    replace(
        "def _detect_cells_pooled(\n",
        "def _e0_uninstrumented_detect_cells_pooled(\n",
        "rename_detector",
    )
    replace(
        "    return np.concatenate([t_col, coords], axis=1).astype(np.int16)\n\n\n"
        "@torch.no_grad()\n",
        """    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


def _detect_cells_pooled(det_logits, t, det_threshold=0.5, pool_kernel=(3, 3, 3)):
    _e0_started = _e0_telemetry.started("detector_extraction")
    try:
        return _e0_uninstrumented_detect_cells_pooled(
            det_logits, t, det_threshold, pool_kernel
        )
    finally:
        _e0_telemetry.add_duration("detector_extraction", _e0_started)


@torch.no_grad()
""",
        "detector_wrapper",
    )

    encode_calls = (
        (
            "        unet_out, det_logits = model.encode(imgs)",
            "        unet_out, det_logits = _e0_encode(model, imgs)",
            "encode_primary",
        ),
        (
            "                _u_flip, det_flip = model.encode(imgs_flip)",
            "                _u_flip, det_flip = _e0_encode(model, imgs_flip)",
            "encode_primary_flip",
        ),
        (
            "                _u_rot, det_rot = model.encode(imgs_rot)",
            "                _u_rot, det_rot = _e0_encode(model, imgs_rot)",
            "encode_primary_rot",
        ),
        (
            "            _u_t, det_t = model.encode(imgs_t)",
            "            _u_t, det_t = _e0_encode(model, imgs_t)",
            "encode_primary_transpose",
        ),
        (
            "            _u_at, det_at = model.encode(imgs_at)",
            "            _u_at, det_at = _e0_encode(model, imgs_at)",
            "encode_primary_antitranspose",
        ),
        (
            "            secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)",
            (
                "            secondary_unet_out, secondary_det_logits = "
                "_e0_encode(secondary_model, imgs)"
            ),
            "encode_secondary",
        ),
        (
            (
                "                        _, secondary_det_flip = "
                "secondary_model.encode(secondary_imgs_flip)"
            ),
            (
                "                        _, secondary_det_flip = "
                "_e0_encode(secondary_model, secondary_imgs_flip)"
            ),
            "encode_secondary_flip",
        ),
        (
            (
                "                        _, secondary_det_rot = "
                "secondary_model.encode(secondary_imgs_rot)"
            ),
            (
                "                        _, secondary_det_rot = "
                "_e0_encode(secondary_model, secondary_imgs_rot)"
            ),
            "encode_secondary_rot",
        ),
        (
            "                    _, secondary_det_t = secondary_model.encode(secondary_imgs_t)",
            (
                "                    _, secondary_det_t = "
                "_e0_encode(secondary_model, secondary_imgs_t)"
            ),
            "encode_secondary_transpose",
        ),
        (
            "                    _, secondary_det_at = secondary_model.encode(secondary_imgs_at)",
            (
                "                    _, secondary_det_at = "
                "_e0_encode(secondary_model, secondary_imgs_at)"
            ),
            "encode_secondary_antitranspose",
        ),
    )
    for old, new, label in encode_calls:
        replace(old, new, label)

    replace(
        "            edge_logits_pair = model.predict_edges(\n",
        "            edge_logits_pair = _e0_predict_edges(model,\n",
        "score_forward",
    )
    replace(
        "                reverse_logits_native = model.predict_edges(\n",
        "                reverse_logits_native = _e0_predict_edges(model,\n",
        "score_reverse",
    )
    replace(
        "                secondary_logits_pair = secondary_model.predict_edges(\n",
        "                secondary_logits_pair = _e0_predict_edges(secondary_model,\n",
        "score_secondary",
    )
    replace(
        """                        with guard_log.open("a") as guard_handle:
""",
        """                        _e0_telemetry.record_retention(guard_record)
                        with guard_log.open("a") as guard_handle:
""",
        "invocation_scoped_retention",
    )

    replace(
        """    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])
""",
        """    _e0_telemetry.begin_dataset(ds_path.stem, str(ds_path.resolve()))
    _e0_data_started = _e0_telemetry.started("data_read")
    try:
        ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)
        if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
            raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")
        zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
        q_low = float(ds.quantiles["0.001"])
        q_high = float(ds.quantiles["0.999"])
    finally:
        _e0_telemetry.add_duration("data_read", _e0_data_started)
""",
        "dataset_and_metadata_read",
    )

    replace(
        """            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > cfg.threshold
                ],
                reverse=True,
            )
""",
        """            _e0_threshold_started = _e0_telemetry.started("threshold")
            try:
                raw = edge_logits_pair[0]
                if cfg.edge_activation == "softmax":
                    probs = torch.softmax(raw, dim=0).cpu().numpy()
                else:
                    probs = torch.sigmoid(raw).cpu().numpy()

                candidates = sorted(
                    [
                        (probs[i, j], i, j)
                        for i in range(n_src)
                        for j in range(n_tgt)
                        if probs[i, j] > cfg.threshold
                    ],
                    reverse=True,
                )
            finally:
                _e0_telemetry.add_duration("threshold", _e0_threshold_started)
            _e0_telemetry.add_pair_counts(
                pair_universe=n_src * n_tgt,
                threshold_passing=len(candidates),
            )
""",
        "activation_threshold_and_counts",
    )

    replace(
        """        _coordinate_record = {
            "columns": ["t", "z", "y", "x"],
""",
        """        _e0_coordinate_artifact = _e0_telemetry.capture_coordinates(
            dataset=ds_path.stem,
            payload=_coordinate_array.tobytes(order="C"),
            shape=_coordinate_array.shape,
            dtype="<i2",
        )
        _coordinate_record = {
            "artifact": _e0_coordinate_artifact,
            "columns": ["t", "z", "y", "x"],
""",
        "coordinate_artifact",
    )

    replace(
        """        graph = build_graph(coords, edges)
        if cfg.use_ilp and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
            )
            with suppress_output():
                graph = solver.solve(graph)
        save_graph(graph, output_dir / f"{name}.geff")
""",
        """        graph = _e0_build_graph(coords, edges)
        if cfg.use_ilp and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
            )
            with suppress_output():
                graph = _e0_solve(solver, graph)
        else:
            _e0_telemetry.set_ilp(
                "unavailable",
                reason="disabled" if not cfg.use_ilp else "empty_graph",
            )
        _e0_save_graph(graph, output_dir / f"{name}.geff")
        _e0_telemetry.finish_dataset(
            detected_nodes=len(coords),
            pre_ilp_edges=len(edges),
            output_edges=graph.num_edges(),
        )
""",
        "graph_ilp_geff",
    )

    replace(
        """    print(
        f"Fold {fold}: {len(test_names)} datasets | "
        f"weights={weights_path} | device={device} | window_size={window_size} | pool_kernel_um={cfg.pool_kernel_um}",
        flush=True,
    )

    for name in tqdm(test_names, desc="Predicting", disable=not INTERACTIVE):
""",
        """    print(
        f"Fold {fold}: {len(test_names)} datasets | "
        f"weights={weights_path} | device={device} | window_size={window_size} | pool_kernel_um={cfg.pool_kernel_um}",
        flush=True,
    )

    _e0_telemetry.begin_invocation(
        data_root=str(data_dir.resolve()),
        output_dir=str(output_dir.resolve()),
        method=method,
        fold=fold,
        test_names=test_names,
    )
    for name in tqdm(test_names, desc="Predicting", disable=not INTERACTIVE):
""",
        "invocation_identity",
    )
    replace(
        """
    print(f"Saved {len(test_names)} predictions to {output_dir}", flush=True)
""",
        """
    _e0_telemetry.finish_invocation()
    print(f"Saved {len(test_names)} predictions to {output_dir}", flush=True)
""",
        "invocation_finish",
    )

    compile(source, "<e0-telemetry-patched-support>", "exec")
    output_sha256 = _sha256_text(source)
    runtime_source = runtime_helper_source()
    return PatchedSupportSource(
        source=source,
        pre_public_sha256=PRE_PUBLIC_SUPPORT_SHA256,
        public_patched_input_sha256=input_sha256,
        telemetry_output_sha256=output_sha256,
        runtime_module_name=RUNTIME_MODULE_NAME,
        runtime_module_sha256=_sha256_text(runtime_source),
        anchor_counts=tuple(anchors),
    )
