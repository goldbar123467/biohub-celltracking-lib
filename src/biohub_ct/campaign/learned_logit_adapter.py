"""Faithful per-tile pre-sigmoid cache for the frozen learned detector."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biohub_ct.campaign.detection_diagnostics import (
    CacheIdentity,
    CacheValidationError,
    _atomic_write,
    cache_manifest_path,
    hash_array,
    hash_file,
    hash_json,
)
from biohub_ct.pipelines.learned import LearnedConfig

PAYLOAD_SCHEMA = "biohub.learned.per_tile_logits.v1"


@dataclass(frozen=True)
class LearnedTilePayload:
    """Pre-sigmoid tiles and every value needed to replay probability blending."""

    logits: np.ndarray
    blend_weights: np.ndarray
    origins_zyx: np.ndarray
    stops_zyx: np.ndarray
    raw_shape_zyx: tuple[int, int, int]
    normalized_shape_zyx: tuple[int, int, int]
    normalization: Mapping[str, Any]
    tiling: Mapping[str, Any]
    native_logit_dtype: str
    inference_precision: str
    inference_device_type: str
    tile_size: int
    overlap: int
    schema: str = PAYLOAD_SCHEMA


def _normalization_metadata(raw: np.ndarray, image: np.ndarray, xy_stride: int) -> dict[str, Any]:
    # Recompute only the scalar audit fields. ``image`` itself came from the production function.
    value = raw.astype(np.float32)
    _, y, x = value.shape
    pad_y, pad_x = (-y) % xy_stride, (-x) % xy_stride
    if xy_stride > 1:
        value = np.pad(value, ((0, 0), (0, pad_y), (0, pad_x)), mode="edge")
        value = value.reshape(
            value.shape[0],
            (y + pad_y) // xy_stride,
            xy_stride,
            (x + pad_x) // xy_stride,
            xy_stride,
        ).mean(axis=(2, 4))
    low, high = np.percentile(value, (1.0, 99.8))
    return {
        "implementation": "biohub_ct.training.data.normalize_frame",
        "input_dtype": raw.dtype.str,
        "xy_stride": xy_stride,
        "xy_edge_padding": [int(pad_y), int(pad_x)],
        "downsample": "nonoverlapping_xy_block_mean_after_edge_padding",
        "percentiles": [1.0, 99.8],
        "percentile_values": [float(low), float(high)],
        "denominator_floor": 1.0,
        "clip": [0.0, 1.0],
        "output_dtype": image.dtype.str,
        "normalized_frame_sha256": hash_array(image),
    }


def _tile_starts(shape: Sequence[int], tile_size: int, overlap: int) -> list[list[int]]:
    starts: list[list[int]] = []
    for size in shape:
        positions = list(range(0, max(1, size - tile_size + 1), tile_size - overlap))
        starts.append(sorted(set(positions + [max(0, size - tile_size)])))
    return starts


def _blend_weights(shape: Sequence[int], overlap: int) -> np.ndarray:
    weights = np.ones(tuple(shape), np.float32)
    for axis, length in enumerate(shape):
        ramp = np.minimum(np.arange(length) + 1, np.arange(length, 0, -1))
        axis_shape = [1, 1, 1]
        axis_shape[axis] = length
        weights *= np.minimum(ramp, max(1, overlap)).reshape(axis_shape)
    return weights


def validate_learned_payload(payload: LearnedTilePayload) -> LearnedTilePayload:
    if not isinstance(payload, LearnedTilePayload) or payload.schema != PAYLOAD_SCHEMA:
        raise TypeError("Expected a learned per-tile logit payload")
    arrays = (payload.logits, payload.blend_weights, payload.origins_zyx, payload.stops_zyx)
    if not all(isinstance(value, np.ndarray) for value in arrays):
        raise TypeError("Learned payload arrays must be actual NumPy arrays")
    if payload.logits.dtype != np.float32 or payload.blend_weights.dtype != np.float32:
        raise TypeError("Learned logits and blend weights must be float32")
    if payload.origins_zyx.dtype != np.int64 or payload.stops_zyx.dtype != np.int64:
        raise TypeError("Learned tile bounds must be int64")
    if payload.logits.ndim != 4 or payload.logits.shape != payload.blend_weights.shape:
        raise ValueError("Learned payload requires matching N-Z-Y-X logits and weights")
    count = payload.logits.shape[0]
    if count < 1 or payload.origins_zyx.shape != (count, 3) or payload.stops_zyx.shape != (count, 3):
        raise ValueError("Learned payload has inconsistent tile bounds")
    if not np.isfinite(payload.logits).all() or not np.isfinite(payload.blend_weights).all():
        raise ValueError("Learned payload contains NaN or infinity")
    if np.any(payload.blend_weights <= 0):
        raise ValueError("Production blend weights must be positive")
    shape = np.asarray(payload.normalized_shape_zyx, dtype=np.int64)
    raw_shape = np.asarray(payload.raw_shape_zyx, dtype=np.int64)
    if shape.shape != (3,) or raw_shape.shape != (3,) or np.any(shape <= 0) or np.any(raw_shape <= 0):
        raise ValueError("Learned payload shapes must be positive ZYX triples")
    if np.any(payload.origins_zyx < 0) or np.any(payload.stops_zyx > shape):
        raise ValueError("Learned tile bounds leave the normalized frame")
    extents = payload.stops_zyx - payload.origins_zyx
    if np.any(extents <= 0) or np.any(extents != np.asarray(payload.logits.shape[1:])):
        raise ValueError("Every learned tile bound must match its stored array extent")
    if payload.tile_size < 8 or not 0 <= payload.overlap < payload.tile_size:
        raise ValueError("Invalid learned tiling metadata")
    if payload.inference_precision not in {"native_amp", "full_float32"}:
        raise ValueError("Unknown learned inference precision")
    if payload.inference_device_type not in {"cpu", "cuda"}:
        raise ValueError("Unknown learned inference device type")
    if payload.native_logit_dtype not in {"torch.float16", "torch.bfloat16", "torch.float32"}:
        raise ValueError("Unsupported native logit dtype")
    if payload.inference_precision == "full_float32" and payload.native_logit_dtype != "torch.float32":
        raise ValueError("Full-float32 inference must emit float32 logits")
    if payload.inference_precision == "native_amp" and payload.inference_device_type != "cuda":
        raise ValueError("Native AMP payloads must come from CUDA")
    expected_normalization = {
        "implementation", "input_dtype", "xy_stride", "xy_edge_padding", "downsample",
        "percentiles", "percentile_values", "denominator_floor", "clip", "output_dtype",
        "normalized_frame_sha256",
    }
    if set(payload.normalization) != expected_normalization:
        raise ValueError("Normalization metadata schema mismatch")
    expected_tiling = {
        "axis_order": "ZYX",
        "iteration_order": "Z_outer_Y_middle_X_inner",
        "tile_input_padding_zyx": [[0, 0], [0, 0], [0, 0]],
        "configured_overlap_zyx": [payload.overlap] * 3,
        "stride_zyx": [payload.tile_size - payload.overlap] * 3,
        "border_blend": "separable_positive_edge_ramp",
        "weight_cap": max(1, payload.overlap),
    }
    if payload.tiling != expected_tiling:
        raise ValueError("Tiling/halo/padding metadata differs from production")
    xy_stride = payload.normalization["xy_stride"]
    if not isinstance(xy_stride, int) or isinstance(xy_stride, bool) or xy_stride < 1:
        raise ValueError("Normalization XY stride must be a positive integer")
    expected_shape = (
        int(raw_shape[0]),
        (int(raw_shape[1]) + xy_stride - 1) // xy_stride,
        (int(raw_shape[2]) + xy_stride - 1) // xy_stride,
    )
    if tuple(shape) != expected_shape:
        raise ValueError("Normalized shape does not match raw shape and XY stride")
    starts = _tile_starts(shape, payload.tile_size, payload.overlap)
    expected_origins = [
        (z, y, x)
        for z in starts[0]
        for y in starts[1]
        for x in starts[2]
    ]
    if payload.origins_zyx.tolist() != [list(origin) for origin in expected_origins]:
        raise ValueError("Learned tile bounds are not in production order")
    coverage = np.zeros(tuple(shape), np.float32)
    for index, (origin, stop) in enumerate(zip(payload.origins_zyx, payload.stops_zyx)):
        region = tuple(slice(int(a), int(b)) for a, b in zip(origin, stop))
        expected_stop = np.minimum(origin + payload.tile_size, shape)
        if not np.array_equal(stop, expected_stop):
            raise ValueError("Learned tile stop does not match production tiling")
        if not np.array_equal(
            payload.blend_weights[index],
            _blend_weights(payload.logits[index].shape, payload.overlap),
        ):
            raise ValueError("Learned blend weights differ from production weights")
        coverage[region] += payload.blend_weights[index]
    if not np.isfinite(coverage).all() or np.any(coverage <= 0):
        raise ValueError("Learned tiles do not cover the normalized frame")
    return payload


class LearnedLogitAdapter:
    """Capture the production learned detector immediately before per-tile sigmoid."""

    cache_suffix = ".learned-tiles.npz"
    output_schema = PAYLOAD_SCHEMA

    def __init__(
        self,
        model: Any,
        config: LearnedConfig | None = None,
        *,
        inference_precision: str = "native_amp",
    ) -> None:
        if inference_precision not in {"native_amp", "full_float32"}:
            raise ValueError("inference_precision must be native_amp or full_float32")
        self.model = model
        self.config = LearnedConfig() if config is None else config
        self.inference_precision = inference_precision
        self.name = f"learned-frozen-per-tile-{inference_precision}-v1"

    def capture_payload(
        self,
        frame_zyx: np.ndarray,
        *,
        deadline_at: float,
        progress: Callable[[], None],
    ) -> LearnedTilePayload:
        import torch

        from biohub_ct.training.data import normalize_frame

        image = normalize_frame(frame_zyx, self.config.xy_stride)
        starts = _tile_starts(image.shape, self.config.tile_size, self.config.overlap)
        try:
            device = next(self.model.parameters()).device
        except StopIteration as exc:
            raise ValueError("Learned model must have at least one parameter") from exc
        if self.inference_precision == "native_amp" and device.type != "cuda":
            raise ValueError("native_amp capture requires a CUDA model")
        was_training = self.model.training
        logits: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        origins: list[tuple[int, int, int]] = []
        stops: list[tuple[int, int, int]] = []
        native_dtype: str | None = None
        self.model.eval()
        try:
            with torch.inference_mode():
                for z in starts[0]:
                    for y in starts[1]:
                        for x in starts[2]:
                            if time.monotonic() >= deadline_at:
                                raise TimeoutError("Learned logit capture deadline reached")
                            origin = (z, y, x)
                            stop = tuple(
                                min(point + self.config.tile_size, size)
                                for point, size in zip(origin, image.shape)
                            )
                            region = tuple(slice(a, b) for a, b in zip(origin, stop))
                            tile = torch.from_numpy(image[region].copy())[None, None].to(device)
                            amp = self.inference_precision == "native_amp"
                            with torch.autocast(device_type=device.type, enabled=amp):
                                output = self.model(tile)
                            if output.shape != (1, 1, *tile.shape[2:]):
                                raise ValueError("Learned model output must match B1ZYX tile shape")
                            this_dtype = str(output.dtype)
                            if native_dtype is not None and this_dtype != native_dtype:
                                raise RuntimeError("Learned model changed output dtype between tiles")
                            native_dtype = this_dtype
                            value = output[0, 0].float().cpu().numpy()
                            if not np.isfinite(value).all():
                                raise RuntimeError("Nonfinite learned pre-sigmoid logits")
                            logits.append(value)
                            weights.append(_blend_weights(value.shape, self.config.overlap))
                            origins.append(origin)
                            stops.append(stop)
                            progress()
        finally:
            self.model.train(was_training)
        payload = LearnedTilePayload(
            logits=np.stack(logits).astype(np.float32, copy=False),
            blend_weights=np.stack(weights).astype(np.float32, copy=False),
            origins_zyx=np.asarray(origins, dtype=np.int64),
            stops_zyx=np.asarray(stops, dtype=np.int64),
            raw_shape_zyx=tuple(map(int, frame_zyx.shape)),
            normalized_shape_zyx=tuple(map(int, image.shape)),
            normalization=_normalization_metadata(frame_zyx, image, self.config.xy_stride),
            tiling={
                "axis_order": "ZYX",
                "iteration_order": "Z_outer_Y_middle_X_inner",
                "tile_input_padding_zyx": [[0, 0], [0, 0], [0, 0]],
                "configured_overlap_zyx": [self.config.overlap] * 3,
                "stride_zyx": [self.config.tile_size - self.config.overlap] * 3,
                "border_blend": "separable_positive_edge_ramp",
                "weight_cap": max(1, self.config.overlap),
            },
            native_logit_dtype=native_dtype or "",
            inference_precision=self.inference_precision,
            inference_device_type=device.type,
            tile_size=self.config.tile_size,
            overlap=self.config.overlap,
        )
        return validate_learned_payload(payload)

    def save_cache(self, path: Path, payload: LearnedTilePayload, identity: CacheIdentity) -> dict[str, Any]:
        return save_learned_tile_cache(path, payload, identity, adapter=self.name)

    def load_cache(self, path: Path, identity: CacheIdentity) -> tuple[LearnedTilePayload, dict[str, Any]]:
        return load_learned_tile_cache(path, expected_identity=identity)


def reconstruct_probabilities(
    payload: LearnedTilePayload,
    *,
    activation: str = "native",
    logit_clamp: tuple[float, float] | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Replay per-tile sigmoid followed by production-order float32 weighted blending."""
    import torch

    validate_learned_payload(payload)
    if activation not in {"native", "float32"}:
        raise ValueError("activation must be native or float32")
    if logit_clamp is not None:
        if len(logit_clamp) != 2 or not all(math.isfinite(value) for value in logit_clamp):
            raise ValueError("logit_clamp must contain two finite bounds")
        if logit_clamp[0] >= logit_clamp[1]:
            raise ValueError("logit_clamp lower bound must be below upper bound")
    target_device = device or payload.inference_device_type
    if target_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Native CUDA activation replay requires available CUDA")
    dtype_by_name = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    total = np.zeros(payload.normalized_shape_zyx, np.float32)
    count = np.zeros(payload.normalized_shape_zyx, np.float32)
    for index, (origin, stop) in enumerate(zip(payload.origins_zyx, payload.stops_zyx)):
        dtype = dtype_by_name[payload.native_logit_dtype] if activation == "native" else torch.float32
        tensor = torch.from_numpy(payload.logits[index]).to(device=target_device, dtype=dtype)
        if logit_clamp is not None:
            tensor = tensor.clamp(*logit_clamp)
        probability = tensor.sigmoid().float().cpu().numpy()
        region = tuple(slice(int(a), int(b)) for a, b in zip(origin, stop))
        total[region] += probability * payload.blend_weights[index]
        count[region] += payload.blend_weights[index]
    return total / count


def _payload_metadata(payload: LearnedTilePayload) -> dict[str, Any]:
    return {
        "schema": payload.schema,
        "raw_shape_zyx": list(payload.raw_shape_zyx),
        "normalized_shape_zyx": list(payload.normalized_shape_zyx),
        "normalization": dict(payload.normalization),
        "tiling": dict(payload.tiling),
        "native_logit_dtype": payload.native_logit_dtype,
        "inference_precision": payload.inference_precision,
        "inference_device_type": payload.inference_device_type,
        "tile_size": payload.tile_size,
        "overlap": payload.overlap,
    }


def save_learned_tile_cache(
    path: Path | str,
    payload: LearnedTilePayload,
    identity: CacheIdentity,
    *,
    adapter: str,
) -> dict[str, Any]:
    if identity.output_schema != PAYLOAD_SCHEMA:
        raise ValueError("Learned tile cache requires the learned output schema identity")
    value = validate_learned_payload(payload)
    destination = Path(path)
    arrays = {
        "logits": value.logits,
        "blend_weights": value.blend_weights,
        "origins_zyx": value.origins_zyx,
        "stops_zyx": value.stops_zyx,
    }
    _atomic_write(
        destination,
        lambda stream: np.savez(stream, **arrays),
        binary=True,
    )
    manifest = {
        "schema_version": 1,
        "output_kind": "per_tile_raw_pre_sigmoid_logits_and_blend_weights",
        "axis_order": "NZYX",
        "identity": asdict(identity),
        "identity_sha256": identity.sha256,
        "payload": _payload_metadata(value),
        "payload_sha256": hash_json(_payload_metadata(value)),
        "array_sha256": {name: hash_array(array) for name, array in arrays.items()},
        "file_sha256": hash_file(destination),
        "adapter": adapter,
    }
    _atomic_write(
        cache_manifest_path(destination),
        lambda stream: stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n"),
        binary=False,
    )
    _, checked = load_learned_tile_cache(destination, expected_identity=identity)
    return checked


def load_learned_tile_cache(
    path: Path | str, *, expected_identity: CacheIdentity
) -> tuple[LearnedTilePayload, dict[str, Any]]:
    if expected_identity.output_schema != PAYLOAD_SCHEMA:
        raise ValueError("Learned tile cache requires the learned output schema identity")
    source = Path(path)
    try:
        manifest = json.loads(cache_manifest_path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheValidationError("Missing or malformed learned tile manifest") from exc
    required = {
        "schema_version", "output_kind", "axis_order", "identity", "identity_sha256",
        "payload", "payload_sha256", "array_sha256", "file_sha256", "adapter",
    }
    if set(manifest) != required or manifest["schema_version"] != 1:
        raise CacheValidationError("Learned tile manifest schema mismatch")
    if manifest["output_kind"] != "per_tile_raw_pre_sigmoid_logits_and_blend_weights" or manifest["axis_order"] != "NZYX":
        raise CacheValidationError("Unsupported learned tile representation")
    if manifest["identity"] != asdict(expected_identity) or manifest["identity_sha256"] != expected_identity.sha256:
        raise CacheValidationError("Learned tile cache identity was invalidated")
    if hash_json(manifest["payload"]) != manifest["payload_sha256"]:
        raise CacheValidationError("Learned tile metadata checksum mismatch")
    try:
        if hash_file(source) != manifest["file_sha256"]:
            raise CacheValidationError("Learned tile file checksum mismatch")
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {"logits", "blend_weights", "origins_zyx", "stops_zyx"}:
                raise CacheValidationError("Learned tile archive member mismatch")
            arrays = {name: archive[name].copy() for name in archive.files}
    except CacheValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise CacheValidationError("Learned tile archive cannot be loaded") from exc
    if {name: hash_array(array) for name, array in arrays.items()} != manifest["array_sha256"]:
        raise CacheValidationError("Learned tile array checksum mismatch")
    metadata = manifest["payload"]
    metadata_fields = {
        "schema", "raw_shape_zyx", "normalized_shape_zyx", "normalization", "tiling",
        "native_logit_dtype", "inference_precision", "inference_device_type",
        "tile_size", "overlap",
    }
    if not isinstance(metadata, dict) or set(metadata) != metadata_fields:
        raise CacheValidationError("Learned tile metadata schema mismatch")
    try:
        payload = LearnedTilePayload(
            **arrays,
            raw_shape_zyx=tuple(metadata["raw_shape_zyx"]),
            normalized_shape_zyx=tuple(metadata["normalized_shape_zyx"]),
            normalization=metadata["normalization"],
            tiling=metadata["tiling"],
            native_logit_dtype=metadata["native_logit_dtype"],
            inference_precision=metadata["inference_precision"],
            inference_device_type=metadata["inference_device_type"],
            tile_size=metadata["tile_size"],
            overlap=metadata["overlap"],
            schema=metadata["schema"],
        )
        validate_learned_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheValidationError("Learned tile payload contract mismatch") from exc
    return payload, manifest


def create_adapter(config: Mapping[str, Any], *, model_file: Path | str) -> LearnedLogitAdapter:
    """CLI factory for either a trusted training checkpoint or exported state dict."""
    import torch

    from biohub_ct.training.model import ModelConfig, PointDetector3D

    allowed = {"model_format", "model_config", "inference_config", "device", "inference_precision"}
    if set(config) - allowed:
        raise ValueError(f"Unknown learned adapter configuration: {sorted(set(config) - allowed)}")
    model_format = config.get("model_format", "training_checkpoint")
    device = str(config.get("device", "cuda"))
    if device not in {"cpu", "cuda"}:
        raise ValueError("Learned adapter device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Learned adapter requested CUDA but CUDA is unavailable")
    # The campaign model file is a trusted local artifact selected by its outer SHA-256 identity.
    state = torch.load(
        Path(model_file), map_location="cpu", weights_only=model_format == "state_dict"
    )
    if model_format == "training_checkpoint":
        model_config = state["model_config"]
        weights = state["model"]
    elif model_format == "state_dict":
        model_config = config.get("model_config", {})
        weights = state
    else:
        raise ValueError("model_format must be training_checkpoint or state_dict")
    model = PointDetector3D(ModelConfig(**model_config)).to(device)
    model.load_state_dict(weights, strict=True)
    if not all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()):
        raise ValueError("Learned model contains nonfinite parameters")
    learned_config = LearnedConfig(**config.get("inference_config", {}))
    return LearnedLogitAdapter(
        model,
        learned_config,
        inference_precision=str(config.get("inference_precision", "native_amp")),
    )
