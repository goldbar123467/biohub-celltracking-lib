from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from biohub_ct.data.geff_io import read_geff_graph
from biohub_ct.data.paths import DatasetRecord
from biohub_ct.data.splits import embryo_id
from biohub_ct.data.zarr_io import open_zarr_volume


@dataclass(frozen=True)
class DataConfig:
    patch_shape: tuple[int, int, int] = (32, 32, 32)
    xy_stride: int = 4
    seed: int = 20260905
    background_intensity: float = 0.05
    sigma_voxels: tuple[float, float, float] = (1.0, 1.0, 1.0)
    positive_radius: float = 1.0
    background_exclusion_radius: float = 4.0
    positive_weight: float = 1.0 / 1.05
    steps_per_frame: int = 1
    augment: bool = True

    def __post_init__(self) -> None:
        numerical = (
            *self.sigma_voxels,
            self.background_intensity,
            self.positive_radius,
            self.background_exclusion_radius,
            self.positive_weight,
        )
        if not all(np.isfinite(value) for value in numerical):
            raise ValueError("Data configuration values must be finite")
        if len(self.patch_shape) != 3 or min(self.patch_shape) < 8:
            raise ValueError("Patch shape must have three dimensions >= 8")
        if not 0 <= self.background_intensity < 1:
            raise ValueError("background_intensity must be in [0,1)")
        if len(self.sigma_voxels) != 3 or min(self.sigma_voxels) <= 0:
            raise ValueError("Three positive Gaussian sigmas required")
        if self.positive_radius <= 0 or not 0 < self.positive_weight < 1:
            raise ValueError("Invalid supervision radius or weight")
        if self.steps_per_frame < 1 or self.seed < 0:
            raise ValueError("Invalid frame reuse count or seed")
        if self.xy_stride < 1 or self.background_exclusion_radius < self.positive_radius:
            raise ValueError("Invalid stride or background exclusion radius")


def normalize_frame(frame: np.ndarray, xy_stride: int = 4) -> np.ndarray:
    """Per-frame robust intensity scaling; no cross-frame fitted statistics."""
    if frame.ndim != 3 or frame.dtype != np.uint16:
        raise ValueError("Expected one uint16 ZYX frame")
    if xy_stride < 1:
        raise ValueError("xy_stride must be positive")
    image = frame.astype(np.float32)
    z, y, x = image.shape
    if xy_stride > 1:
        py, px = (-y) % xy_stride, (-x) % xy_stride
        image = np.pad(image, ((0, 0), (0, py), (0, px)), mode="edge")
        image = image.reshape(
            z, (y + py) // xy_stride, xy_stride, (x + px) // xy_stride, xy_stride
        ).mean(axis=(2, 4))
    low, high = np.percentile(image, (1.0, 99.8))
    return np.clip((image - low) / max(float(high - low), 1.0), 0.0, 1.0).astype(np.float32)


def raw_to_model_points(points: np.ndarray, xy_stride: int = 4) -> np.ndarray:
    offset = (xy_stride - 1) / 2
    return (np.asarray(points, dtype=np.float32) - [0, offset, offset]) / [1, xy_stride, xy_stride]


def model_to_raw_points(points: np.ndarray, xy_stride: int = 4) -> np.ndarray:
    offset = (xy_stride - 1) / 2
    return np.asarray(points, dtype=np.float32) * [1, xy_stride, xy_stride] + [0, offset, offset]


def make_sparse_targets(
    image: np.ndarray,
    points: np.ndarray,
    config: DataConfig,
    *,
    background_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Label Gaussians around annotated centers and only dark background.

    Dark-background assumption is a recorded heuristic, not dense ground truth.
    Unlabeled bright tissue is ignored. Coordinates are ZYX patch voxels.
    """
    target = np.zeros(image.shape, dtype=np.float32)
    positive = np.zeros(image.shape, dtype=bool)
    excluded = np.zeros(image.shape, dtype=bool)
    sigmas = np.asarray(config.sigma_voxels)
    radius = np.ceil(sigmas * config.background_exclusion_radius).astype(int)
    for point in points:
        lo = np.maximum(np.floor(point).astype(int) - radius, 0)
        hi = np.minimum(np.ceil(point).astype(int) + radius + 1, image.shape)
        if np.any(hi <= lo):
            continue
        slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        grid = np.ogrid[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
        distance = sum(((g - p) / s) ** 2 for g, p, s in zip(grid, point, sigmas))
        inside = distance <= config.positive_radius**2
        target[slices] = np.maximum(target[slices], np.where(inside, np.exp(-distance / 2), 0))
        positive[slices] |= inside
        excluded[slices] |= distance <= config.background_exclusion_radius**2
    local_max = image.copy()
    for axis in range(3):
        padding = [(0, 0)] * 3
        padding[axis] = (1, 1)
        padded = np.pad(local_max, padding, mode="edge")
        windows = []
        for shift in range(3):
            window = [slice(None)] * 3
            window[axis] = slice(shift, shift + image.shape[axis])
            windows.append(padded[tuple(window)])
        local_max = np.maximum.reduce(windows)
    background = (local_max <= background_threshold) & ~excluded
    weight = np.zeros(image.shape, dtype=np.float32)
    npos, nneg = int(positive.sum()), int(background.sum())
    if npos:
        weight[positive] = config.positive_weight / npos
    if nneg:
        weight[background] = (1 - config.positive_weight) / nneg
    return target, weight


class SparsePatchSampler:
    """Deterministic step-indexed sampling, bounded two-frame cache, no 4D reads."""

    def __init__(
        self,
        records: list[DatasetRecord],
        train_ids: list[str],
        val_ids: list[str],
        config: DataConfig = DataConfig(),
        *,
        dev_ids: list[str] | None = None,
    ) -> None:
        if not train_ids or len(set(train_ids)) != len(train_ids):
            raise ValueError("Training IDs must be nonempty and unique")
        if {embryo_id(x) for x in train_ids} & {embryo_id(x) for x in val_ids}:
            raise ValueError("Training and validation embryos overlap")
        if set(train_ids) & set(dev_ids or []):
            raise ValueError("Training and development IDs overlap")
        by_id = {record.name: record for record in records}
        if set(train_ids) - by_id.keys():
            raise ValueError("Requested training IDs missing from records")
        self.config = config
        self.train_ids = sorted(train_ids)
        self.val_ids = sorted(val_ids)
        self.dev_ids = sorted(dev_ids or [])
        self.records = {key: by_id[key] for key in self.train_ids}
        self.frames: list[tuple[str, int, np.ndarray]] = []
        self.frames_by_clip: dict[str, list[tuple[str, int, np.ndarray]]] = {}
        self.volumes = {}
        self._cache: OrderedDict[tuple[str, int], tuple[np.ndarray, float]] = OrderedDict()
        for name in self.train_ids:
            self.frames_by_clip[name] = []
            record = self.records[name]
            if record.geff_path is None:
                raise ValueError(f"Missing GEFF labels for {name}")
            volume = open_zarr_volume(record.zarr_path, require_complete_chunks=True)
            model_scale = np.asarray(volume.scale) * [1, config.xy_stride, config.xy_stride]
            if not np.allclose(model_scale, [1.625] * 3, rtol=0, atol=1e-6):
                raise ValueError(
                    "Initial detector requires 1.625um isotropic voxels after XY averaging"
                )
            self.volumes[name] = volume
            graph, _ = read_geff_graph(record.geff_path)
            for t, nodes in sorted(graph.nodes_by_time().items()):
                points = np.asarray([node.coord for node in nodes], dtype=np.float32)
                if (
                    not 0 <= t < volume.shape[0]
                    or np.any(points < 0)
                    or np.any(points >= np.asarray(volume.shape[1:]))
                ):
                    raise ValueError(f"GEFF coordinates outside volume in {name} frame {t}")
                self.frames.append((name, t, points))
                self.frames_by_clip[name].append((name, t, points))
        if not self.frames:
            raise ValueError("No labeled training frames")

    def sample(self, step: int, batch_size: int) -> dict:
        if step < 0 or batch_size < 1:
            raise ValueError("Nonnegative step and positive batch size required")
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, step]))
        frame_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, step // self.config.steps_per_frame, 181])
        )
        nonempty_clips = [name for name in self.train_ids if self.frames_by_clip[name]]
        name = nonempty_clips[int(frame_rng.integers(len(nonempty_clips)))]
        clip_frames = self.frames_by_clip[name]
        name, t, points = clip_frames[int(frame_rng.integers(len(clip_frames)))]
        key = (name, t)
        if key not in self._cache:
            raw = self.volumes[name].read_frame(t)
            image = normalize_frame(raw, self.config.xy_stride)
            threshold = self.config.background_intensity
            self._cache[key] = (image, threshold)
            if len(self._cache) > 2:
                self._cache.popitem(last=False)
        image, threshold = self._cache[key]
        self._cache.move_to_end(key)
        points = raw_to_model_points(points, self.config.xy_stride)
        patch_shape = np.asarray(self.config.patch_shape)
        images, targets, weights, metadata = [], [], [], []
        for _ in range(batch_size):
            anchor = points[int(rng.integers(len(points)))]
            jitter = rng.integers(-patch_shape // 4, patch_shape // 4 + 1)
            origin = np.clip(
                np.floor(anchor).astype(int) - patch_shape // 2 + jitter,
                0,
                np.maximum(np.asarray(image.shape) - patch_shape, 0),
            )
            if rng.random() < 0.5:
                origin = rng.integers(0, np.maximum(np.asarray(image.shape) - patch_shape, 0) + 1)
            stop = np.minimum(origin + patch_shape, image.shape)
            slices = tuple(slice(int(a), int(b)) for a, b in zip(origin, stop))
            patch = np.array(image[slices], copy=True)
            valid_shape = patch.shape
            patch = np.pad(patch, tuple((0, int(n - s)) for n, s in zip(patch_shape, patch.shape)))
            local_points = points - origin
            target, weight = make_sparse_targets(
                patch, local_points, self.config, background_threshold=threshold
            )
            # Padded voxels are synthetic and never participate in supervision.
            valid = np.zeros(patch.shape, dtype=bool)
            valid[tuple(slice(0, s) for s in valid_shape)] = True
            weight[~valid] = 0
            if not np.any(weight):
                # A uniform bright unlabeled patch has no valid target; use the
                # already selected labeled anchor, never relabel it as negative.
                origin = np.clip(
                    np.floor(anchor).astype(int) - patch_shape // 2,
                    0,
                    np.maximum(np.asarray(image.shape) - patch_shape, 0),
                )
                stop = np.minimum(origin + patch_shape, image.shape)
                slices = tuple(slice(int(a), int(b)) for a, b in zip(origin, stop))
                patch = np.array(image[slices], copy=True)
                valid_shape = patch.shape
                patch = np.pad(
                    patch, tuple((0, int(n - s)) for n, s in zip(patch_shape, patch.shape))
                )
                target, weight = make_sparse_targets(
                    patch, points - origin, self.config, background_threshold=threshold
                )
                valid = np.zeros(patch.shape, dtype=bool)
                valid[tuple(slice(0, s) for s in valid_shape)] = True
                weight[~valid] = 0
            if self.config.augment:
                for axis in range(3):
                    if rng.random() < 0.5:
                        patch, target, weight = (np.flip(x, axis) for x in (patch, target, weight))
                patch = np.clip(patch * rng.uniform(0.85, 1.15), 0, 1)
            images.append(np.ascontiguousarray(patch[None], dtype=np.float32))
            targets.append(np.ascontiguousarray(target[None], dtype=np.float32))
            weights.append(np.ascontiguousarray(weight[None], dtype=np.float32))
            center = np.clip(np.rint(anchor).astype(int), 0, np.asarray(image.shape) - 1)
            center_window = tuple(
                slice(max(0, int(c) - 1), min(s, int(c) + 2)) for c, s in zip(center, image.shape)
            )
            metadata.append(
                {
                    "dataset": name,
                    "t": t,
                    "origin_zyx": origin.tolist(),
                    "anchor_below_dark_threshold": bool(image[tuple(center)] <= threshold),
                    "anchor_satisfies_dark_rule": bool(image[center_window].max() <= threshold),
                    "background_threshold": threshold,
                    "positive_fraction": float(np.mean((target > 0) & (weight > 0))),
                    "background_fraction": float(np.mean((target == 0) & (weight > 0))),
                    "unknown_fraction": float(np.mean(weight == 0)),
                }
            )
        return {
            "image": np.stack(images),
            "target": np.stack(targets),
            "weight": np.stack(weights),
            "metadata": metadata,
            "loss_normalizer": float(batch_size),
        }
