"""Bounded, frame-at-a-time inference for the experimental point detector."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from biohub_ct.data.schema import Graph, Node
from biohub_ct.data.zarr_io import open_zarr_volume
from biohub_ct.detection.peaks import anisotropic_nms
from biohub_ct.linking.greedy import greedy_link


@dataclass(frozen=True)
class LearnedConfig:
    threshold: float = 0.5
    nms_radius_um: float = 3.0
    link_distance_um: float = 8.0
    xy_stride: int = 4
    tile_size: int = 64
    overlap: int = 16
    max_nodes_per_frame: int = 2000

    def __post_init__(self):
        if not 0 < self.threshold < 1 or self.xy_stride < 1:
            raise ValueError("Invalid threshold or stride")
        if self.tile_size < 8 or not 0 <= self.overlap < self.tile_size:
            raise ValueError("Invalid inference tiling")
        if min(self.nms_radius_um, self.link_distance_um, self.max_nodes_per_frame) <= 0:
            raise ValueError("Positive detection/linking limits required")


def frame_probabilities(model, raw, config=LearnedConfig(), *, deadline_at=None):
    import torch

    from biohub_ct.training.data import normalize_frame

    image = normalize_frame(raw, config.xy_stride)
    total = np.zeros(image.shape, np.float32)
    count = np.zeros(image.shape, np.float32)
    starts = []
    for size in image.shape:
        positions = list(
            range(0, max(1, size - config.tile_size + 1), config.tile_size - config.overlap)
        )
        starts.append(sorted(set(positions + [max(0, size - config.tile_size)])))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for z in starts[0]:
                for y in starts[1]:
                    for x in starts[2]:
                        if deadline_at is not None and time.monotonic() >= deadline_at:
                            raise TimeoutError("Learned inference deadline reached")
                        region = tuple(
                            slice(p, min(p + config.tile_size, n))
                            for p, n in zip((z, y, x), image.shape)
                        )
                        tile = torch.from_numpy(image[region].copy())[None, None].to(device)
                        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                            probability = model(tile).sigmoid()[0, 0].float().cpu().numpy()
                        if not np.isfinite(probability).all():
                            raise RuntimeError("Nonfinite model prediction")
                        # Smoothly downweight tile borders while retaining a positive denominator.
                        weights = np.ones(probability.shape, np.float32)
                        for axis, length in enumerate(probability.shape):
                            ramp = np.minimum(np.arange(length) + 1, np.arange(length, 0, -1))
                            shape = [1, 1, 1]
                            shape[axis] = length
                            weights *= np.minimum(ramp, max(1, config.overlap)).reshape(shape)
                        total[region] += probability * weights
                        count[region] += weights
    finally:
        model.train(was_training)
    return total / count


def probability_nodes(probability, raw_shape, scale, t, start_id, config=LearnedConfig()):
    from scipy.ndimage import maximum_filter

    from biohub_ct.training.data import model_to_raw_points

    selected = (probability >= config.threshold) & (
        probability == maximum_filter(probability, size=3, mode="nearest")
    )
    coordinates = np.argwhere(selected)
    if not len(coordinates):
        return [], False
    scores = probability[tuple(coordinates.T)]
    # Bound pathological flat heatmaps deterministically; expose this truncation.
    candidate_limit = config.max_nodes_per_frame * 10
    order = np.argsort(-scores, kind="stable")[:candidate_limit]
    truncated = len(coordinates) > candidate_limit
    coordinates = model_to_raw_points(coordinates[order], config.xy_stride)
    coordinates = np.clip(np.rint(coordinates), 0, np.asarray(raw_shape) - 1).astype(int)
    keep = anisotropic_nms(
        coordinates.tolist(), scores[order].tolist(), scale=scale, radius_um=config.nms_radius_um
    )
    truncated |= len(keep) > config.max_nodes_per_frame
    nodes = [
        Node(start_id + i, t, *map(int, coordinates[k]))
        for i, k in enumerate(keep[: config.max_nodes_per_frame])
    ]
    return nodes, truncated


def predict_record(model, record, config=LearnedConfig(), *, deadline_at=None):
    return predict_record_thresholds(model, record, [config], deadline_at=deadline_at)[0]


def predict_record_thresholds(model, record, configs, *, deadline_at=None):
    if not configs or any(
        (c.xy_stride, c.tile_size, c.overlap)
        != (configs[0].xy_stride, configs[0].tile_size, configs[0].overlap)
        for c in configs
    ):
        raise ValueError("Threshold candidates must share preprocessing and tiling")
    volume = open_zarr_volume(record.zarr_path, require_complete_chunks=True)
    all_nodes = [[] for _ in configs]
    all_capped = [[] for _ in configs]
    for t in range(volume.shape[0]):
        raw = volume.read_frame(t)
        probabilities = frame_probabilities(model, raw, configs[0], deadline_at=deadline_at)
        for nodes, capped_frames, config in zip(all_nodes, all_capped, configs):
            frame_nodes, capped = probability_nodes(
                probabilities, raw.shape, volume.scale, t, len(nodes), config
            )
            nodes.extend(frame_nodes)
            if capped:
                capped_frames.append(t)
    graphs = []
    for nodes, capped_frames, config in zip(all_nodes, all_capped, configs):
        fallback = not nodes
        if fallback:
            nodes = [Node(0, 0, *(int(n // 2) for n in volume.shape[1:]))]
        graph = Graph(
            nodes,
            greedy_link(
                nodes,
                scale=volume.scale,
                max_distance=config.link_distance_um,
                deadline_at=deadline_at,
            ),
        )
        graph.inference_diagnostics = {"fallback": fallback, "capped_frames": capped_frames}
        graphs.append(graph)
    return graphs
