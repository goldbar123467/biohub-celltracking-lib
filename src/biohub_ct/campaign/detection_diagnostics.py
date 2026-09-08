"""Frozen-detector logit caching and deterministic peak-extraction ablations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

SHA256_HEX = frozenset("0123456789abcdef")
SUPPORTED_OUTPUT_SCHEMAS = frozenset(
    {"biohub.raw_logits.zyx.float32.v1", "biohub.learned.per_tile_logits.v1"}
)


class CacheValidationError(ValueError):
    """A cached frame is corrupt or belongs to a different frozen identity."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_file(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_array(array: np.ndarray) -> str:
    """Hash array type, shape, and C-order bytes without changing its values."""
    value = np.asarray(array)
    header = canonical_json_bytes(
        {"dtype": value.dtype.str, "shape": list(value.shape), "order": "C"}
    )
    digest = hashlib.sha256(header)
    digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")


@dataclass(frozen=True)
class CacheIdentity:
    """Every identity component that can change compatible raw frame logits."""

    model_sha256: str
    source_sha256: str
    config_sha256: str
    input_frame_sha256: str
    transform_sha256: str
    precision_sha256: str
    tta_sha256: str
    output_schema: str = "biohub.raw_logits.zyx.float32.v1"

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            if field.endswith("_sha256"):
                _require_digest(value, field)
        if self.output_schema not in SUPPORTED_OUTPUT_SCHEMAS:
            raise ValueError("Unsupported diagnostic cache output schema")

    @property
    def sha256(self) -> str:
        return hash_json(asdict(self))


def make_cache_identity(
    *,
    model_sha256: str,
    source_sha256: str,
    config: Mapping[str, Any],
    input_frame: np.ndarray,
    transform: Mapping[str, Any],
    precision: Mapping[str, Any],
    tta: Mapping[str, Any],
    output_schema: str = "biohub.raw_logits.zyx.float32.v1",
) -> CacheIdentity:
    """Bind one frame cache to model, code, preprocessing, numerics, and TTA."""
    return CacheIdentity(
        model_sha256=model_sha256,
        source_sha256=source_sha256,
        config_sha256=hash_json(config),
        input_frame_sha256=hash_array(input_frame),
        transform_sha256=hash_json(transform),
        precision_sha256=hash_json(precision),
        tta_sha256=hash_json(tta),
        output_schema=output_schema,
    )


def cache_manifest_path(logits_path: Path | str) -> Path:
    path = Path(logits_path)
    return path.with_name(path.name + ".manifest.json")


def _atomic_write(path: Path, writer: Callable[[Any], None], *, binary: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        mode = "wb" if binary else "w"
        kwargs = {} if binary else {"encoding": "utf-8", "newline": "\n"}
        with os.fdopen(descriptor, mode, **kwargs) as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_logit_cache(
    logits_path: Path | str,
    logits: np.ndarray,
    identity: CacheIdentity,
    *,
    adapter: str,
) -> dict[str, Any]:
    """Atomically save finite, actual float32 pre-sigmoid logits and reload them."""
    path = Path(logits_path)
    if identity.output_schema != "biohub.raw_logits.zyx.float32.v1":
        raise ValueError("Generic logit cache requires the generic output schema identity")
    if not isinstance(logits, np.ndarray) or logits.dtype != np.dtype("float32"):
        raise TypeError("Diagnostic cache requires an actual NumPy float32 array")
    if logits.ndim != 3 or any(size <= 0 for size in logits.shape):
        raise ValueError("Raw logits must be a nonempty ZYX array")
    if not np.isfinite(logits).all():
        raise ValueError("Raw logits contain NaN or infinity")
    if not isinstance(adapter, str) or not adapter:
        raise ValueError("Adapter identity must be nonempty")

    _atomic_write(path, lambda stream: np.save(stream, logits, allow_pickle=False), binary=True)
    manifest = {
        "schema_version": 1,
        "output_kind": "raw_pre_sigmoid_logits",
        "axis_order": "ZYX",
        "dtype": "float32",
        "shape": list(logits.shape),
        "identity": asdict(identity),
        "identity_sha256": identity.sha256,
        "array_sha256": hash_array(logits),
        "file_sha256": hash_file(path),
        "adapter": adapter,
    }
    manifest_path = cache_manifest_path(path)
    _atomic_write(
        manifest_path,
        lambda stream: stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n"),
        binary=False,
    )
    # A successful write is not evidence until the durable bytes strictly reload.
    _, checked = load_logit_cache(path, expected_identity=identity)
    return checked


def load_logit_cache(
    logits_path: Path | str, *, expected_identity: CacheIdentity
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strictly reload a cache, rejecting corruption and identity invalidation."""
    if expected_identity.output_schema != "biohub.raw_logits.zyx.float32.v1":
        raise ValueError("Generic logit cache requires the generic output schema identity")
    path = Path(logits_path)
    manifest_path = cache_manifest_path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheValidationError("Missing or malformed logit cache manifest") from exc
    required = {
        "schema_version",
        "output_kind",
        "axis_order",
        "dtype",
        "shape",
        "identity",
        "identity_sha256",
        "array_sha256",
        "file_sha256",
        "adapter",
    }
    if set(manifest) != required:
        raise CacheValidationError("Logit cache manifest schema mismatch")
    if (
        manifest["schema_version"] != 1
        or manifest["output_kind"] != "raw_pre_sigmoid_logits"
        or manifest["axis_order"] != "ZYX"
        or manifest["dtype"] != "float32"
    ):
        raise CacheValidationError("Logit cache declares an unsupported representation")
    if manifest["identity"] != asdict(expected_identity):
        raise CacheValidationError("Logit cache identity was invalidated")
    if manifest["identity_sha256"] != expected_identity.sha256:
        raise CacheValidationError("Logit cache identity digest mismatch")
    try:
        actual_file_sha256 = hash_file(path)
    except OSError as exc:
        raise CacheValidationError("Logit cache data cannot be read") from exc
    if actual_file_sha256 != manifest["file_sha256"]:
        raise CacheValidationError("Logit cache file checksum mismatch")
    try:
        logits = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CacheValidationError("Logit cache data cannot be loaded") from exc
    if (
        not isinstance(logits, np.ndarray)
        or logits.dtype != np.dtype("float32")
        or logits.ndim != 3
        or list(logits.shape) != manifest["shape"]
    ):
        raise CacheValidationError("Logit cache array contract mismatch")
    if not np.isfinite(logits).all():
        raise CacheValidationError("Logit cache contains NaN or infinity")
    if hash_array(logits) != manifest["array_sha256"]:
        raise CacheValidationError("Logit cache payload checksum mismatch")
    return logits, manifest


@dataclass(frozen=True)
class PlateauCandidate:
    coord_zyx: tuple[int, int, int]
    logit: float
    plateau_voxels: int
    centroid_zyx: tuple[float, float, float]
    representative_displacement_um: float


def _validate_logits_and_geometry(
    logits: np.ndarray, scale_zyx_um: Sequence[float], threshold_logit: float
) -> tuple[float, float, float]:
    if not isinstance(logits, np.ndarray) or logits.dtype != np.dtype("float32"):
        raise TypeError("Peak extraction requires uncensored float32 raw logits")
    if logits.ndim != 3 or any(size <= 0 for size in logits.shape):
        raise ValueError("Peak extraction requires a nonempty ZYX array")
    if not np.isfinite(logits).all() or not math.isfinite(threshold_logit):
        raise ValueError("Peak extraction inputs must be finite")
    if len(scale_zyx_um) != 3:
        raise ValueError("Physical scale must use ZYX order")
    scale = tuple(float(value) for value in scale_zyx_um)
    if not all(math.isfinite(value) and value > 0 for value in scale):
        raise ValueError("Physical scale must be finite and positive")
    return scale


def connected_plateau_candidates(
    logits: np.ndarray,
    *,
    threshold_logit: float,
    scale_zyx_um: Sequence[float],
) -> tuple[list[PlateauCandidate], int]:
    """Return one physical-centroid representative per connected equal maximum.

    Plateaus use full 26-neighbor connectivity, matching a 3x3x3 local-maximum
    neighborhood. Equal-valued maxima that are disconnected remain independent.
    """
    from scipy.ndimage import label, maximum_filter

    scale = _validate_logits_and_geometry(logits, scale_zyx_um, threshold_logit)
    maxima = (logits >= threshold_logit) & (
        logits == maximum_filter(logits, size=3, mode="constant", cval=-np.inf)
    )
    raw_maxima = int(maxima.sum())
    candidates: list[PlateauCandidate] = []
    # Label each exact value independently. This makes the equality contract
    # explicit even if the local-max neighborhood changes later.
    for value in np.unique(logits[maxima]):
        components, count = label(maxima & (logits == value), structure=np.ones((3, 3, 3)))
        for component_id in range(1, count + 1):
            coords = np.argwhere(components == component_id)
            centroid = coords.mean(axis=0)
            physical_delta = (coords - centroid) * np.asarray(scale)
            squared_distance = np.einsum("ij,ij->i", physical_delta, physical_delta)
            best_distance = float(squared_distance.min())
            tied = coords[np.flatnonzero(squared_distance == best_distance)]
            representative = min(map(tuple, tied.tolist()))
            candidates.append(
                PlateauCandidate(
                    coord_zyx=representative,
                    logit=float(value),
                    plateau_voxels=len(coords),
                    centroid_zyx=tuple(float(item) for item in centroid),
                    representative_displacement_um=math.sqrt(best_distance),
                )
            )
    candidates.sort(key=lambda candidate: (-candidate.logit, candidate.coord_zyx))
    return candidates, raw_maxima


def physical_nms(
    candidates: Sequence[PlateauCandidate],
    *,
    scale_zyx_um: Sequence[float],
    radius_um: float,
) -> list[PlateauCandidate]:
    """Apply deterministic Euclidean suppression in anisotropic physical space."""
    if len(scale_zyx_um) != 3:
        raise ValueError("Physical scale must use ZYX order")
    scale = np.asarray(scale_zyx_um, dtype=np.float64)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Physical scale must be finite and positive")
    if not math.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("NMS radius must be finite and positive")
    ordered = sorted(candidates, key=lambda candidate: (-candidate.logit, candidate.coord_zyx))
    kept: list[PlateauCandidate] = []
    kept_physical: list[np.ndarray] = []
    for candidate in ordered:
        point = np.asarray(candidate.coord_zyx, dtype=np.float64) * scale
        if all(float(np.linalg.norm(point - other)) > radius_um for other in kept_physical):
            kept.append(candidate)
            kept_physical.append(point)
    return kept


@dataclass(frozen=True)
class ExtractionResult:
    raw_maximum_voxels: int
    plateau_count: int
    pre_cap_count: int
    capped: bool
    retained: tuple[PlateauCandidate, ...]


def extract_plateau_peaks(
    logits: np.ndarray,
    *,
    threshold_logit: float,
    scale_zyx_um: Sequence[float],
    radius_um: float,
    max_candidates: int | None = None,
) -> ExtractionResult:
    candidates, raw_maxima = connected_plateau_candidates(
        logits, threshold_logit=threshold_logit, scale_zyx_um=scale_zyx_um
    )
    retained = physical_nms(candidates, scale_zyx_um=scale_zyx_um, radius_um=radius_um)
    pre_cap_count = len(retained)
    if max_candidates is not None:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
            raise ValueError("Candidate cap must be a positive integer when enabled")
        retained = retained[:max_candidates]
    return ExtractionResult(
        raw_maximum_voxels=raw_maxima,
        plateau_count=len(candidates),
        pre_cap_count=pre_cap_count,
        capped=len(retained) < pre_cap_count,
        retained=tuple(retained),
    )


@dataclass(frozen=True)
class ExtractionConfig:
    """CPU extraction settings; precision is deliberately absent."""

    tie_variant: str
    threshold_logit: float
    nms_radius_um: float

    def __post_init__(self) -> None:
        if not self.tie_variant:
            raise ValueError("Tie variant must be named")
        if not math.isfinite(self.threshold_logit):
            raise ValueError("Threshold must be finite")
        if not math.isfinite(self.nms_radius_um) or self.nms_radius_um <= 0:
            raise ValueError("NMS radius must be finite and positive")


@dataclass(frozen=True)
class FrozenAblationPlan:
    precision_variant: str
    initial: tuple[ExtractionConfig, ...]
    refinement: tuple[ExtractionConfig, ...] | None = None

    def __post_init__(self) -> None:
        if not self.precision_variant:
            raise ValueError("Precision variant must be named independently")
        if not 1 <= len(self.initial) <= 15:
            raise ValueError("Initial grid must contain 1 to 15 configurations")
        if self.refinement is not None and not 1 <= len(self.refinement) <= 15:
            raise ValueError("The single refinement must contain 1 to 15 configurations")
        all_configs = self.initial + (() if self.refinement is None else self.refinement)
        if len(set(all_configs)) != len(all_configs):
            raise ValueError("Ablation configurations must be unique")


def freeze_ablation_grid(
    *,
    precision_variant: str,
    tie_variant: str,
    thresholds_logit: Sequence[float],
    radii_um: Sequence[float],
    refinement: Sequence[tuple[float, float]] | None = None,
) -> FrozenAblationPlan:
    initial = tuple(
        ExtractionConfig(tie_variant, float(threshold), float(radius))
        for threshold in thresholds_logit
        for radius in radii_um
    )
    refined = (
        None
        if refinement is None
        else tuple(
            ExtractionConfig(tie_variant, float(threshold), float(radius))
            for threshold, radius in refinement
        )
    )
    return FrozenAblationPlan(precision_variant, initial, refined)


class RawLogitsAdapter(Protocol):
    """Explicit boundary for learned or public-reference model integration."""

    name: str

    def raw_logits(
        self,
        frame_zyx: np.ndarray,
        *,
        deadline_at: float,
        progress: Callable[[], None],
    ) -> np.ndarray:
        """Return actual, finite, pre-sigmoid NumPy float32 ZYX logits."""


def validate_adapter_output(logits: Any) -> np.ndarray:
    """Reject probability substitution, implicit casting, clipping, and nonfinite data."""
    if not isinstance(logits, np.ndarray) or logits.dtype != np.dtype("float32"):
        raise TypeError("Adapter must return an actual NumPy float32 logit array")
    if logits.ndim != 3 or any(size <= 0 for size in logits.shape):
        raise ValueError("Adapter must return a nonempty ZYX array")
    if not np.isfinite(logits).all():
        raise ValueError("Adapter returned NaN or infinity")
    return logits


def sigmoid_scores(logits: np.ndarray, *, arithmetic: str) -> np.ndarray:
    """Apply sigmoid in the declared arithmetic and return float32 scores.

    ``float16`` simulates the original low-precision activation from cached AMP
    logits. ``float32`` tests whether float32 activation alone removes ties. A
    full-float32 model pass requires a separately keyed cache identity.
    """
    values = validate_adapter_output(logits)
    if arithmetic == "float16":
        working = values.astype(np.float16)
        if not np.isfinite(working).all():
            raise ValueError("Float16 activation would overflow cached logits")
    elif arithmetic == "float32":
        working = values
    else:
        raise ValueError("Sigmoid arithmetic must be float16 or float32")
    with np.errstate(over="ignore"):
        scores = 1 / (1 + np.exp(-working))
    scores = scores.astype(np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Sigmoid produced nonfinite scores")
    return scores
