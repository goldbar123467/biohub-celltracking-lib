#!/usr/bin/env python3
"""Bounded compatibility profile for the pinned E0 public detector.

This is deliberately not a submission run.  It reads two frames from one fixed
training Zarr, profiles the two pinned detector checkpoints on one centered
tile, and writes a JSON receipt.  Optional runtime libraries are imported only
after the command-line and artifact contracts have been checked; nothing is
installed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import signal
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_STEM = "44b6_0113de3b"
FRAME_INDICES = (0, 1)
UPSTREAM_NOTEBOOK_SHA256 = "521cb97f0f457643379a51b60c4f71e3f4cc7d1823fd98cbb97633ffaa515ec4"
EXPECTED_RAW_SHAPE = (100, 64, 256, 256)
MAX_TILE_YX = 64
DEFAULT_TILE_YX = 64
DEFAULT_WALL_SECONDS = 120.0
# The documented default leaves 30 seconds for its outer 150-second process guard.
MAX_WALL_SECONDS = 540.0

EXPECTED_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
    "pool_kernel_um": 5.0,
}
EXPECTED_SHA256 = {
    "predict_source": "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9",
    "train_source": "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea",
    "temporal_unet_source": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac",
    "node_transformer_source": "b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e",
    "package_init_source": "26a18d8da84e40da73281a48ebc3017d847a2e57431ab63e8629d2109e6e8571",
    "models_init_source": "ab7587ef79856bae50d24b62e5805092d0459ee1c586522b763f9ef70c093e1d",
    "config": "e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a",
    "primary_checkpoint": "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771",
    "secondary_checkpoint": "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f",
}

# These are recorded so a timing result cannot be mistaken for a differently
# configured public method.  Only the detector/TTA subset is executed here.
PUBLIC_CONSTANTS = {
    "detection_threshold": 0.965,
    "secondary_detection_weight": 0.80,
    "secondary_edge_logit_weight": 0.15,
    "secondary_edge_fusion_mode": "low_margin_consensus",
    "secondary_low_margin_max": 0.35,
    "edge_candidate_threshold": 0.48,
    "bidirectional_edge_weight": 0.15,
    "bidirectional_fusion_mode": "harmonic_probability",
    "primary_edge_feature_tta": True,
    "secondary_edge_feature_tta": False,
    "detector_tta_views": 8,
    "unique_effective_planar_transforms": 7,
    "duplicated_upstream_transform": {
        "view": "anti_transpose",
        "equivalent_to": "flip_x",
        "source_expression": "rot90(1).transpose(-1, -2)",
    },
    "ilp_edge_weight": -1.0,
    "ilp_appearance_weight": 0.0,
    "ilp_disappearance_weight": 2.0,
    "ilp_division_weight": 1.2,
    "gap_close_max_gap": 2,
    "gap_close_um": 5.0,
    "minimum_track_length": 6,
}

TTA_VIEWS = (
    "identity",
    "flip_x",
    "flip_y",
    "flip_xy",
    "rot90",
    "rot270",
    "transpose",
    "anti_transpose",
)
TTA_EFFECTIVE_TRANSFORMS = {
    "identity": "identity",
    "flip_x": "flip_x",
    "flip_y": "flip_y",
    "flip_xy": "flip_xy",
    "rot90": "rot90",
    "rot270": "rot270",
    "transpose": "transpose",
    # This is the exact upstream expression rot90(1).transpose(-1, -2).
    # Algebraically it duplicates flip_x; it is not anti-diagonal reflection.
    "anti_transpose": "flip_x",
}

CAMPAIGN_ENVIRONMENT = (
    "BIOHUB_RUN_ID",
    "BIOHUB_RUN_SPEC_SHA256",
    "BIOHUB_INTENT_ID",
    "BIOHUB_FENCING_TOKEN",
    "BIOHUB_ATTEMPT_DIR",
    "BIOHUB_PROGRESS_PATH",
)
_CAMPAIGN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ProfileError(RuntimeError):
    """Base class for failures that should be represented in the receipt."""


class UnavailableDependency(ProfileError):
    """A required library is absent from the existing environment."""


class CompatibilityMismatch(ProfileError):
    """The supplied source, checkpoint, data, or device violates the pin."""


class BudgetExceeded(ProfileError):
    """The bounded profile crossed its wall-time deadline."""


@dataclass(frozen=True)
class ProfileRequest:
    data_root: Path
    support_root: Path
    checkpoints: tuple[Path, Path]
    output: Path
    tile_yx: int = DEFAULT_TILE_YX
    max_wall_seconds: float = DEFAULT_WALL_SECONDS


@dataclass(frozen=True)
class SupportPaths:
    source_root: Path
    package_init_source: Path
    models_init_source: Path
    predict_source: Path
    train_source: Path
    temporal_unet_source: Path
    node_transformer_source: Path


@dataclass(frozen=True)
class CampaignContext:
    run_id: str
    run_spec_sha256: str
    intent_id: str
    fencing_token: int
    attempt_dir: Path
    progress_path: Path
    result_manifest_path: Path

    def identity(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_spec_sha256": self.run_spec_sha256,
            "intent_id": self.intent_id,
            "fencing_token": self.fencing_token,
        }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_request(request: ProfileRequest) -> None:
    if not (32 <= request.tile_yx <= MAX_TILE_YX):
        raise CompatibilityMismatch(f"tile_yx must be in [32, {MAX_TILE_YX}]")
    if request.tile_yx % 4:
        raise CompatibilityMismatch("tile_yx must be divisible by four for two U-Net pools")
    if not (0.0 < request.max_wall_seconds <= MAX_WALL_SECONDS):
        raise CompatibilityMismatch(f"max_wall_seconds must be in (0, {MAX_WALL_SECONDS}]")
    if len(request.checkpoints) != 2:
        raise CompatibilityMismatch("exactly two checkpoints are required")


def centered_strided_slice(length: int, stride: int, output_size: int) -> slice:
    """Return a raw-axis slice for one centered tile after striding."""
    if length <= 0 or stride <= 0 or output_size <= 0:
        raise ValueError("length, stride, and output_size must be positive")
    downsampled = (length + stride - 1) // stride
    if output_size > downsampled:
        raise CompatibilityMismatch(
            f"requested tile {output_size} exceeds downsampled axis {downsampled}"
        )
    start_index = (downsampled - output_size) // 2
    return slice(start_index * stride, (start_index + output_size) * stride, stride)


def apply_planar_view(tensor: Any, view: str) -> Any:
    """Apply one exact upstream planar view on the final Y/X axes."""
    if view == "identity":
        return tensor
    if view == "flip_x":
        return tensor.flip((-1,))
    if view == "flip_y":
        return tensor.flip((-2,))
    if view == "flip_xy":
        return tensor.flip((-2, -1))
    if view == "rot90":
        return tensor.rot90(1, (-2, -1))
    if view == "rot270":
        return tensor.rot90(3, (-2, -1))
    if view == "transpose":
        return tensor.transpose(-1, -2)
    if view == "anti_transpose":
        return tensor.rot90(1, (-2, -1)).transpose(-1, -2)
    raise ValueError(f"unknown planar view: {view}")


def invert_planar_view(tensor: Any, view: str) -> Any:
    """Map a view-space output back to the identity orientation."""
    if view in {"identity", "flip_x", "flip_y", "flip_xy", "transpose"}:
        return apply_planar_view(tensor, view)
    if view == "rot90":
        return tensor.rot90(-1, (-2, -1))
    if view == "rot270":
        return tensor.rot90(-3, (-2, -1))
    if view == "anti_transpose":
        return tensor.transpose(-1, -2).rot90(-1, (-2, -1))
    raise ValueError(f"unknown planar view: {view}")


def resolve_support_paths(root: Path) -> SupportPaths:
    candidates = (root / "repo" / "src", root / "src", root)
    source_root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "biohub_tracking/models/temporal_unet.py").is_file()
        ),
        None,
    )
    if source_root is None:
        raise CompatibilityMismatch(
            "support_root must be an extracted support root, repo root, or repo/src root"
        )
    repo_root = source_root.parent if source_root.name == "src" else root
    paths = SupportPaths(
        source_root=source_root,
        package_init_source=source_root / "biohub_tracking/__init__.py",
        models_init_source=source_root / "biohub_tracking/models/__init__.py",
        predict_source=repo_root / "scripts/predict_unet_transformer.py",
        train_source=repo_root / "scripts/train_unet_transformer.py",
        temporal_unet_source=source_root / "biohub_tracking/models/temporal_unet.py",
        node_transformer_source=source_root / "biohub_tracking/models/simple_node_transformer.py",
    )
    missing = [str(path) for path in paths.__dict__.values() if not Path(path).exists()]
    if missing:
        raise CompatibilityMismatch(f"support source files missing: {missing}")
    return paths


def verify_file(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompatibilityMismatch(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise CompatibilityMismatch(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": actual}


def verify_sources(paths: SupportPaths) -> dict[str, dict[str, Any]]:
    return {
        "biohub_tracking/__init__.py": verify_file(
            paths.package_init_source,
            EXPECTED_SHA256["package_init_source"],
            "biohub_tracking package source",
        ),
        "biohub_tracking/models/__init__.py": verify_file(
            paths.models_init_source,
            EXPECTED_SHA256["models_init_source"],
            "biohub_tracking.models package source",
        ),
        "predict_unet_transformer.py": verify_file(
            paths.predict_source, EXPECTED_SHA256["predict_source"], "prediction source"
        ),
        "train_unet_transformer.py": verify_file(
            paths.train_source, EXPECTED_SHA256["train_source"], "training source"
        ),
        "temporal_unet.py": verify_file(
            paths.temporal_unet_source,
            EXPECTED_SHA256["temporal_unet_source"],
            "TemporalUNet3D source",
        ),
        "simple_node_transformer.py": verify_file(
            paths.node_transformer_source,
            EXPECTED_SHA256["node_transformer_source"],
            "SimpleNodeTransformer source",
        ),
    }


def load_and_verify_config(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = checkpoint.parent / "config.json"
    identity = verify_file(config_path, EXPECTED_SHA256["config"], "model config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != EXPECTED_CONFIG:
        raise CompatibilityMismatch(
            f"model config differs from effective public config: {config!r}"
        )
    return config, identity


def import_runtime_dependencies(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> tuple[ModuleType, ModuleType, ModuleType]:
    modules: list[ModuleType] = []
    missing: list[str] = []
    for name in ("numpy", "torch", "zarr"):
        try:
            modules.append(importer(name))
        except (ImportError, ModuleNotFoundError):
            missing.append(name)
    if missing:
        raise UnavailableDependency(
            "required existing libraries are unavailable; no install attempted: "
            + ", ".join(missing)
        )
    return modules[0], modules[1], modules[2]


def make_model_class(torch: ModuleType, simple_node_transformer: type) -> type:
    """Construct the checkpoint wrapper without importing the training CLI."""

    class UNetNodeTransformer(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, unet: Any, unet_out_channels: int) -> None:
            super().__init__()
            self.unet = unet
            self.unet_out_channels = unet_out_channels
            self.detect_head = torch.nn.Conv3d(unet_out_channels, 1, kernel_size=1)
            self.transformer = simple_node_transformer(
                feat_dim=unet_out_channels + 32,
                hidden_dim=128,
                n_heads=4,
                n_blocks=4,
                dropout=0.3,
            )

        def encode(self, imgs: Any) -> tuple[Any, list[Any]]:
            if imgs.ndim != 5:
                raise ValueError("imgs must have shape (B, W, Z, Y, X)")
            window = imgs.unsqueeze(2)
            unet_out = self.unet(window)
            det_logits = [
                self.detect_head(unet_out[:, index]) for index in range(unet_out.shape[1])
            ]
            return unet_out, det_logits

    return UNetNodeTransformer


def import_pinned_models(paths: SupportPaths, torch: ModuleType) -> tuple[type, type]:
    source_text = str(paths.source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        temporal_module = importlib.import_module("biohub_tracking.models.temporal_unet")
        transformer_module = importlib.import_module(
            "biohub_tracking.models.simple_node_transformer"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise UnavailableDependency(
            f"pinned support model source could not import with existing libraries: {exc}"
        ) from exc
    return temporal_module.TemporalUNet3D, make_model_class(
        torch, transformer_module.SimpleNodeTransformer
    )


def _sync(torch: ModuleType) -> None:
    torch.cuda.synchronize()


def _tensor_summary(tensor: Any, np: ModuleType) -> dict[str, Any]:
    array = tensor.detach().float().cpu().contiguous().numpy()
    return {
        "shape": list(tensor.shape),
        "source_dtype": str(tensor.dtype),
        "hash_dtype": str(array.dtype),
        "sha256": sha256_bytes(array.tobytes(order="C")),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array, dtype=np.float64)),
        "finite": bool(np.isfinite(array).all()),
    }


def _array_summary(array: Any, np: ModuleType) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": sha256_bytes(contiguous.tobytes(order="C")),
        "min": float(np.min(contiguous)),
        "max": float(np.max(contiguous)),
        "finite": bool(np.isfinite(contiguous).all()),
    }


def _check_deadline(started: float, seconds: float, stage: str) -> None:
    elapsed = time.monotonic() - started
    if elapsed > seconds:
        raise BudgetExceeded(
            f"profile exceeded {seconds:.3f}s after {stage}; elapsed={elapsed:.3f}s"
        )


@contextmanager
def _alarm_deadline(seconds: float) -> Iterator[None]:
    """Apply a Linux wall alarm in addition to explicit stage checks."""
    if os.name == "nt" or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(_signum: int, _frame: Any) -> None:
        raise BudgetExceeded(f"profile reached its {seconds:.3f}s wall alarm")

    previous_handler = signal.signal(signal.SIGALRM, handler)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _metadata_identities(zarr_path: Path) -> list[dict[str, Any]]:
    candidates = (
        zarr_path / "zarr.json",
        zarr_path / ".zattrs",
        zarr_path / ".zgroup",
        zarr_path / "0" / "zarr.json",
        zarr_path / "0" / ".zarray",
        zarr_path / "0" / ".zattrs",
    )
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in candidates
        if path.is_file()
    ]


def _load_frames(
    request: ProfileRequest,
    np: ModuleType,
    torch: ModuleType,
    zarr: ModuleType,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}
    zarr_path = request.data_root / f"{DATASET_STEM}.zarr"
    if not zarr_path.is_dir():
        raise CompatibilityMismatch(f"fixed profile Zarr is missing: {zarr_path}")

    then = time.perf_counter()
    group = zarr.open_group(str(zarr_path), mode="r")
    attrs = dict(group.attrs)
    array = group["0"]
    raw_shape = tuple(int(value) for value in array.shape)
    timings["metadata_seconds"] = time.perf_counter() - then
    if raw_shape != EXPECTED_RAW_SHAPE:
        raise CompatibilityMismatch(
            f"fixed profile expects raw TZYX shape {EXPECTED_RAW_SHAPE}, found {raw_shape}"
        )

    quantiles = attrs.get("image_statistics", {}).get("quantiles", {})
    try:
        q_low = float(quantiles["0.001"])
        q_high = float(quantiles["0.999"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompatibilityMismatch(
            "Zarr attrs must contain numeric image_statistics.quantiles 0.001 and 0.999"
        ) from exc
    if not q_high > q_low:
        raise CompatibilityMismatch(f"invalid quantile range: {q_low} .. {q_high}")

    dz, dy, dx = EXPECTED_CONFIG["downsample"]
    y_slice = centered_strided_slice(raw_shape[2], dy, request.tile_yx)
    x_slice = centered_strided_slice(raw_shape[3], dx, request.tile_yx)
    frames = []
    frame_receipts = []
    then = time.perf_counter()
    for frame_index in FRAME_INDICES:
        native = np.asarray(array[frame_index, slice(None, None, dz), y_slice, x_slice])
        if native.shape[-2:] != (request.tile_yx, request.tile_yx):
            raise CompatibilityMismatch(f"unexpected tile shape: {native.shape}")
        frame_receipts.append({"t": frame_index, **_array_summary(native, np)})
        frames.append(native.astype(np.float32, copy=False))
    timings["zarr_io_and_float32_seconds"] = time.perf_counter() - then

    then = time.perf_counter()
    imgs = torch.from_numpy(np.stack(frames, axis=0))
    imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
    timings["normalization_seconds"] = time.perf_counter() - then
    if tuple(imgs.shape[:1]) != (2,) or imgs.ndim != 4:
        raise CompatibilityMismatch(f"normalized pair must have WZYX shape, found {imgs.shape}")

    receipt = {
        "zarr_path": str(zarr_path.resolve()),
        "axis_order": "TZYX",
        "raw_shape": list(raw_shape),
        "raw_dtype": str(array.dtype),
        "metadata_files": _metadata_identities(zarr_path),
        "quantiles": {"0.001": q_low, "0.999": q_high},
        "downsample": list(EXPECTED_CONFIG["downsample"]),
        "raw_slices": {
            "z": [None, None, dz],
            "y": [y_slice.start, y_slice.stop, y_slice.step],
            "x": [x_slice.start, x_slice.stop, x_slice.step],
        },
        "frames": frame_receipts,
        "normalized_pair": _tensor_summary(imgs, np),
        "interpolation_triggered": False,
    }
    return imgs, receipt, timings


def _load_model(
    checkpoint: Path,
    config: dict[str, Any],
    torch: ModuleType,
    temporal_unet: type,
    wrapper: type,
    device: Any,
) -> tuple[Any, float]:
    then = time.perf_counter()
    unet = temporal_unet(
        in_channels=1,
        out_channels=config["unet_out_channels"],
        layers=config["unet_layers"],
    )
    model = wrapper(unet=unet, unet_out_channels=config["unet_out_channels"])
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    _sync(torch)
    return model, time.perf_counter() - then


def _profile_tta_model(
    label: str,
    model: Any,
    imgs: Any,
    torch: ModuleType,
    np: ModuleType,
    started: float,
    max_wall_seconds: float,
    *,
    edge_feature_tta: bool,
) -> tuple[Any, list[Any], dict[str, Any]]:
    views = []
    feature_acc = None
    detection_acc = None
    identity_features = None
    with torch.inference_mode():
        for view_index, view in enumerate(TTA_VIEWS):
            _check_deadline(started, max_wall_seconds, f"{label}:{view}:start")
            _sync(torch)
            then = time.perf_counter()
            transformed = apply_planar_view(imgs, view)
            _sync(torch)
            transform_seconds = time.perf_counter() - then

            then = time.perf_counter()
            features_view, detections_view = model.encode(transformed)
            _sync(torch)
            forward_seconds = time.perf_counter() - then

            then = time.perf_counter()
            detections = [invert_planar_view(item, view) for item in detections_view]
            if view_index == 0:
                identity_features = features_view
                feature_acc = features_view.clone() if edge_feature_tta else features_view
                detection_acc = detections
            else:
                if edge_feature_tta:
                    feature_acc = feature_acc + invert_planar_view(features_view, view)
                detection_acc = [left + right for left, right in zip(detection_acc, detections)]
            _sync(torch)
            inverse_and_accumulate_seconds = time.perf_counter() - then
            views.append(
                {
                    "name": view,
                    "effective_transform": TTA_EFFECTIVE_TRANSFORMS[view],
                    "input_shape": list(transformed.shape),
                    "input_transform_seconds": transform_seconds,
                    "forward_seconds": forward_seconds,
                    "inverse_and_accumulate_seconds": inverse_and_accumulate_seconds,
                }
            )
            del transformed, features_view, detections_view, detections
            _check_deadline(started, max_wall_seconds, f"{label}:{view}:complete")

        then = time.perf_counter()
        averaged_features = feature_acc / len(TTA_VIEWS) if edge_feature_tta else identity_features
        averaged_detections = [item / len(TTA_VIEWS) for item in detection_acc]
        mean_abs_feature_delta = float(
            (averaged_features - identity_features).abs().float().mean().item()
        )
        output = {
            "label": label,
            "edge_feature_tta": edge_feature_tta,
            "executed_view_count": len(TTA_VIEWS),
            "unique_effective_transform_count": len(set(TTA_EFFECTIVE_TRANSFORMS.values())),
            "views": views,
            "feature_output": _tensor_summary(averaged_features, np),
            "detection_outputs": [_tensor_summary(item, np) for item in averaged_detections],
            "mean_abs_feature_delta_from_identity": mean_abs_feature_delta,
        }
        _sync(torch)
        output["average_and_extract_seconds"] = time.perf_counter() - then
    return averaged_features, averaged_detections, output


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def campaign_context_from_environment(
    request: ProfileRequest, environ: dict[str, str] | None = None
) -> CampaignContext | None:
    """Read the identity injected by ``campaign.worker_ticket.execute_ticket``."""
    values = os.environ if environ is None else environ
    present = {name: values.get(name) for name in CAMPAIGN_ENVIRONMENT}
    populated = {name for name, value in present.items() if value is not None}
    if not populated:
        return None
    missing = set(CAMPAIGN_ENVIRONMENT) - populated
    if missing:
        raise CompatibilityMismatch(f"partial campaign environment; missing {sorted(missing)}")
    run_id = str(present["BIOHUB_RUN_ID"])
    run_spec_sha256 = str(present["BIOHUB_RUN_SPEC_SHA256"])
    intent_id = str(present["BIOHUB_INTENT_ID"])
    if not _CAMPAIGN_ID.fullmatch(run_id):
        raise CompatibilityMismatch("BIOHUB_RUN_ID must be a resolved campaign identifier")
    if not _SHA256.fullmatch(run_spec_sha256):
        raise CompatibilityMismatch("BIOHUB_RUN_SPEC_SHA256 must be a lowercase SHA-256")
    if not _CAMPAIGN_ID.fullmatch(intent_id):
        raise CompatibilityMismatch("BIOHUB_INTENT_ID must be a resolved campaign identifier")
    try:
        fencing_token = int(str(present["BIOHUB_FENCING_TOKEN"]))
    except ValueError as exc:
        raise CompatibilityMismatch("BIOHUB_FENCING_TOKEN must be an integer") from exc
    if fencing_token <= 0:
        raise CompatibilityMismatch("BIOHUB_FENCING_TOKEN must be positive")

    attempt_dir = Path(str(present["BIOHUB_ATTEMPT_DIR"])).resolve(strict=True)
    progress_path = Path(str(present["BIOHUB_PROGRESS_PATH"]))
    if not progress_path.is_absolute():
        raise CompatibilityMismatch("BIOHUB_PROGRESS_PATH must be absolute")
    progress_path = progress_path.resolve()
    output = request.output.resolve()
    if not output.is_relative_to(attempt_dir):
        raise CompatibilityMismatch("profile output must be inside BIOHUB_ATTEMPT_DIR")
    if not progress_path.is_relative_to(attempt_dir):
        raise CompatibilityMismatch("campaign progress must be inside BIOHUB_ATTEMPT_DIR")
    if not attempt_dir.is_relative_to((PROJECT_ROOT / "reports/campaign-workers").resolve()):
        raise CompatibilityMismatch("BIOHUB_ATTEMPT_DIR must be in reports/campaign-workers")
    if attempt_dir.name != run_id:
        raise CompatibilityMismatch("BIOHUB_ATTEMPT_DIR must end with BIOHUB_RUN_ID")
    result_manifest_path = attempt_dir / "result.json"
    if result_manifest_path.exists():
        raise CompatibilityMismatch("campaign result manifest already exists")
    return CampaignContext(
        run_id=run_id,
        run_spec_sha256=run_spec_sha256,
        intent_id=intent_id,
        fencing_token=fencing_token,
        attempt_dir=attempt_dir,
        progress_path=progress_path,
        result_manifest_path=result_manifest_path,
    )


def _write_campaign_progress(
    context: CampaignContext, *, completed_units: int, error: str | None
) -> None:
    _atomic_write_json(
        context.progress_path,
        {
            **context.identity(),
            "completed_units": completed_units,
            "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": error,
        },
    )


def _write_campaign_result(context: CampaignContext, output: Path) -> None:
    output = output.resolve(strict=True)
    try:
        artifact_key = output.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise CompatibilityMismatch("campaign output must be inside the project root") from exc
    _atomic_write_json(
        context.result_manifest_path,
        {
            **context.identity(),
            "status": "COMPLETE",
            "completed_units": 1,
            "artifact_sha256": {artifact_key: sha256_file(output)},
        },
    )


def _validate_finite_summary(summary: dict[str, Any], label: str) -> None:
    if summary.get("finite") is not True:
        raise CompatibilityMismatch(f"{label} contains NaN or infinity")
    for field in ("min", "max", "mean"):
        if field in summary and not math.isfinite(float(summary[field])):
            raise CompatibilityMismatch(f"{label}.{field} is not finite")


def enforce_pass_criteria(receipt: dict[str, Any]) -> None:
    """Enforce the numerical and shape claims required before status PASS."""
    normalized = receipt["data_identity"]["normalized_pair"]
    _validate_finite_summary(normalized, "normalized_pair")
    for index, frame in enumerate(receipt["data_identity"]["frames"]):
        _validate_finite_summary(frame, f"frame[{index}]")

    profiles = receipt["model_profiles"]
    if [profile.get("label") for profile in profiles] != ["primary", "secondary"]:
        raise CompatibilityMismatch("model profiles must be primary then secondary")
    feature_shapes = []
    expected_views = [(view, TTA_EFFECTIVE_TRANSFORMS[view]) for view in TTA_VIEWS]
    normalized_spatial = normalized["shape"][1:]
    for profile in profiles:
        label = profile["label"]
        if profile.get("executed_view_count") != 8:
            raise CompatibilityMismatch(f"{label} did not execute eight upstream view calls")
        if profile.get("unique_effective_transform_count") != 7:
            raise CompatibilityMismatch(f"{label} effective transform count is not seven")
        observed_views = [
            (view.get("name"), view.get("effective_transform")) for view in profile.get("views", [])
        ]
        if observed_views != expected_views:
            raise CompatibilityMismatch(f"{label} upstream view sequence differs")
        feature = profile["feature_output"]
        _validate_finite_summary(feature, f"{label}.feature_output")
        feature_shapes.append(feature["shape"])
        detections = profile["detection_outputs"]
        if len(detections) != 2:
            raise CompatibilityMismatch(f"{label} must produce two detection tensors")
        for index, detection in enumerate(detections):
            _validate_finite_summary(detection, f"{label}.detection[{index}]")
            if detection["shape"] != [1, 1, *normalized_spatial]:
                raise CompatibilityMismatch(
                    f"{label}.detection[{index}] has unexpected shape {detection['shape']}"
                )
    if feature_shapes[0] != feature_shapes[1]:
        raise CompatibilityMismatch("primary and secondary feature shapes differ")
    expected_prefix = [1, 2, EXPECTED_CONFIG["unet_out_channels"]]
    if feature_shapes[0][:3] != expected_prefix or len(feature_shapes[0]) != 6:
        raise CompatibilityMismatch(f"unexpected averaged feature shape: {feature_shapes[0]}")
    if feature_shapes[0][3:] != normalized_spatial:
        raise CompatibilityMismatch("feature spatial shape differs from normalized input")

    primary_delta = float(profiles[0]["mean_abs_feature_delta_from_identity"])
    secondary_delta = float(profiles[1]["mean_abs_feature_delta_from_identity"])
    if not math.isfinite(primary_delta) or primary_delta <= 0.0:
        raise CompatibilityMismatch("primary edge-feature TTA is nonfinite or a no-op")
    if not math.isfinite(secondary_delta) or secondary_delta != 0.0:
        raise CompatibilityMismatch("secondary identity edge features unexpectedly changed")
    fused_outputs = receipt["secondary_detection_fusion"]["outputs"]
    if len(fused_outputs) != 2:
        raise CompatibilityMismatch("secondary fusion must produce two detection tensors")
    for index, fused in enumerate(fused_outputs):
        _validate_finite_summary(fused, f"fused_detection[{index}]")
        if fused["shape"] != [1, 1, *normalized_spatial]:
            raise CompatibilityMismatch(f"fused_detection[{index}] shape differs")


def _base_receipt(request: ProfileRequest) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "scope": "bounded_e0_detector_compatibility_profile",
        "request": {
            "data_root": str(request.data_root.resolve()),
            "support_root": str(request.support_root.resolve()),
            "checkpoints": [str(path.resolve()) for path in request.checkpoints],
            "output": str(request.output.resolve()),
            "dataset": DATASET_STEM,
            "frames": list(FRAME_INDICES),
            "tile_yx": request.tile_yx,
            "max_wall_seconds": request.max_wall_seconds,
        },
        "effective_public_constants": PUBLIC_CONSTANTS,
        "executed_components": [
            "partial Zarr frame reads",
            "public quantile normalization",
            "primary and secondary TemporalUNet3D encode/detection heads",
            "eight-call detection TTA and primary-model edge-feature TTA",
            "secondary detection mean/std alignment and weighted fusion",
        ],
        "excluded_components": [
            "peak extraction and candidate node generation",
            "transformer edge scoring",
            "bidirectional and secondary edge fusion",
            "ILP graph selection",
            "DeepCenter vetoes and graph post-processing",
            "full-video, CSV, metric, and submission execution",
        ],
    }


def run_profile(request: ProfileRequest) -> dict[str, Any]:
    validate_request(request)
    receipt = _base_receipt(request)
    started = time.monotonic()
    wall_started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with _alarm_deadline(request.max_wall_seconds):
        support_paths = resolve_support_paths(request.support_root)
        receipt["source_identities"] = verify_sources(support_paths)
        entrypoint = Path(__file__).resolve()
        receipt["source_identities"]["profile_public_reference.py"] = {
            "path": str(entrypoint),
            "bytes": entrypoint.stat().st_size,
            "sha256": sha256_file(entrypoint),
        }
        receipt["source_identities"]["upstream_public_notebook_pin"] = {
            "sha256": UPSTREAM_NOTEBOOK_SHA256,
            "verification": "recorded E0 pin; notebook bytes are not read by this bounded profile",
        }

        checkpoint_identities = []
        configs = []
        for label, checkpoint, hash_key in zip(
            ("primary", "secondary"),
            request.checkpoints,
            ("primary_checkpoint", "secondary_checkpoint"),
        ):
            config, config_identity = load_and_verify_config(checkpoint)
            checkpoint_identities.append(
                {
                    "label": label,
                    "checkpoint": verify_file(
                        checkpoint, EXPECTED_SHA256[hash_key], f"{label} checkpoint"
                    ),
                    "config": config_identity,
                }
            )
            configs.append(config)
        receipt["checkpoint_identities"] = checkpoint_identities
        _check_deadline(started, request.max_wall_seconds, "artifact verification")

        np, torch, zarr = import_runtime_dependencies()
        receipt["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": getattr(np, "__version__", "unknown"),
            "torch": getattr(torch, "__version__", "unknown"),
            "zarr": getattr(zarr, "__version__", "unknown"),
            "precision": "torch.float32",
            "autocast": False,
            "gradient_tracking": False,
        }
        if not torch.cuda.is_available():
            raise UnavailableDependency("CUDA is unavailable in the existing Torch runtime")
        visible_devices = int(torch.cuda.device_count())
        if visible_devices != 1:
            raise CompatibilityMismatch(
                f"exactly one visible CUDA device is required, found {visible_devices}"
            )
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        device_name = str(torch.cuda.get_device_name(device))
        if "4070" not in device_name:
            raise CompatibilityMismatch(f"single RTX 4070 required, found {device_name!r}")
        properties = torch.cuda.get_device_properties(device)
        receipt["environment"]["cuda"] = {
            "visible_devices": visible_devices,
            "selected": "cuda:0",
            "name": device_name,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "cudnn": int(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available()
            else None,
        }

        imgs_cpu, data_identity, data_timings = _load_frames(request, np, torch, zarr)
        receipt["data_identity"] = data_identity
        receipt["stage_timings"] = data_timings
        _check_deadline(started, request.max_wall_seconds, "partial frame reads")

        then = time.perf_counter()
        imgs = imgs_cpu.unsqueeze(0).to(device)
        _sync(torch)
        receipt["stage_timings"]["host_to_device_seconds"] = time.perf_counter() - then
        if tuple(imgs.shape[:2]) != (1, EXPECTED_CONFIG["window_size"]):
            raise CompatibilityMismatch(f"device input must have B,W=(1,2), found {imgs.shape}")

        temporal_unet, wrapper = import_pinned_models(support_paths, torch)
        models = []
        load_seconds = []
        for checkpoint, config in zip(request.checkpoints, configs):
            model, elapsed = _load_model(checkpoint, config, torch, temporal_unet, wrapper, device)
            models.append(model)
            load_seconds.append(elapsed)
        receipt["stage_timings"]["model_load_seconds"] = load_seconds
        receipt["models"] = [
            {
                "label": label,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "parameter_bytes": sum(
                    parameter.numel() * parameter.element_size() for parameter in model.parameters()
                ),
            }
            for label, model in zip(("primary", "secondary"), models)
        ]
        receipt["vram"] = {
            "after_both_models_loaded_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "after_both_models_loaded_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
        torch.cuda.reset_peak_memory_stats(device)

        primary_features, primary_detection, primary_profile = _profile_tta_model(
            "primary",
            models[0],
            imgs,
            torch,
            np,
            started,
            request.max_wall_seconds,
            edge_feature_tta=True,
        )
        receipt["vram"]["peak_after_primary_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        receipt["vram"]["peak_after_primary_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
        secondary_features, secondary_detection, secondary_profile = _profile_tta_model(
            "secondary",
            models[1],
            imgs,
            torch,
            np,
            started,
            request.max_wall_seconds,
            edge_feature_tta=False,
        )
        receipt["model_profiles"] = [primary_profile, secondary_profile]
        receipt["vram"]["overall_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        receipt["vram"]["overall_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))

        then = time.perf_counter()
        fused_detection = []
        alignment = []
        with torch.inference_mode():
            for primary, secondary in zip(primary_detection, secondary_detection):
                primary_mean = primary.mean()
                secondary_mean = secondary.mean()
                primary_scale = primary.float().std(unbiased=False).clamp_min(1e-4)
                secondary_scale = secondary.float().std(unbiased=False).clamp_min(1e-4)
                scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
                aligned = (secondary - secondary_mean) * scale_ratio + primary_mean
                fused = (
                    1.0 - PUBLIC_CONSTANTS["secondary_detection_weight"]
                ) * primary + PUBLIC_CONSTANTS["secondary_detection_weight"] * aligned
                alignment.append(
                    {
                        "primary_mean": float(primary_mean.item()),
                        "secondary_mean": float(secondary_mean.item()),
                        "primary_scale": float(primary_scale.item()),
                        "secondary_scale": float(secondary_scale.item()),
                        "scale_ratio": float(scale_ratio.item()),
                    }
                )
                fused_detection.append(fused)
            fused_summaries = [_tensor_summary(item, np) for item in fused_detection]
            _sync(torch)
        receipt["secondary_detection_fusion"] = {
            "weight": PUBLIC_CONSTANTS["secondary_detection_weight"],
            "alignment": alignment,
            "outputs": fused_summaries,
        }
        receipt["stage_timings"]["secondary_alignment_fusion_extract_seconds"] = (
            time.perf_counter() - then
        )
        receipt["retained_feature_shapes"] = {
            "primary": list(primary_features.shape),
            "secondary": list(secondary_features.shape),
        }
        _check_deadline(started, request.max_wall_seconds, "profile completion")

    enforce_pass_criteria(receipt)
    receipt["status"] = "PASS"
    receipt["started_utc"] = wall_started_utc
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["interpretation"] = (
        "Compatibility and bounded resource evidence for detector loading/preprocessing/TTA only; "
        "one two-frame detector window only; not full-video throughput, submission validity, "
        "model quality, or public-score reproduction."
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs=2,
        required=True,
        metavar=("PRIMARY", "SECONDARY"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-yx", type=int, default=DEFAULT_TILE_YX)
    parser.add_argument("--max-wall-seconds", type=float, default=DEFAULT_WALL_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = ProfileRequest(
        data_root=args.data_root,
        support_root=args.support_root,
        checkpoints=tuple(args.checkpoints),
        output=args.output,
        tile_yx=args.tile_yx,
        max_wall_seconds=args.max_wall_seconds,
    )
    started = time.monotonic()
    campaign = None
    try:
        campaign = campaign_context_from_environment(request)
        if campaign is not None:
            _write_campaign_progress(campaign, completed_units=0, error=None)
        receipt = run_profile(request)
        exit_code = 0
    except UnavailableDependency as exc:
        receipt = _base_receipt(request)
        receipt.update(
            status="BLOCKED_UNAVAILABLE_DEPENDENCY",
            error={"type": type(exc).__name__, "message": str(exc)},
            uncertainty=(
                "The compatibility profile did not run. Existing-library availability is the "
                "only diagnosed blocker; no package installation was attempted."
            ),
        )
        exit_code = 2
    except BudgetExceeded as exc:
        receipt = _base_receipt(request)
        receipt.update(
            status="FAILED_WALL_BUDGET",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        exit_code = 3
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        receipt = _base_receipt(request)
        receipt.update(
            status="FAILED_CONTRACT_OR_RUNTIME",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        exit_code = 1
    receipt.setdefault("elapsed_seconds", time.monotonic() - started)
    if campaign is not None:
        receipt["campaign_identity"] = campaign.identity()
    _atomic_write_json(request.output, receipt)
    if campaign is not None:
        if exit_code == 0:
            _write_campaign_result(campaign, request.output)
            _write_campaign_progress(campaign, completed_units=1, error=None)
        else:
            error = receipt.get("error", {})
            _write_campaign_progress(
                campaign,
                completed_units=0,
                error=f"{error.get('type', 'ProfileError')}: {error.get('message', receipt['status'])}",
            )
    print(json.dumps({"status": receipt["status"], "output": str(request.output)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
