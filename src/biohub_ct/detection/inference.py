from __future__ import annotations

import time

from biohub_ct.config import DEFAULT_SCALE
from biohub_ct.data.schema import Node
from biohub_ct.data.zarr_io import LazyZarrVolume
from biohub_ct.detection.centroid_refinement import refine_centroid_moment
from biohub_ct.detection.classical import detect_local_maxima_3d
from biohub_ct.detection.peaks import anisotropic_nms


def detect_nodes_in_volume(
    volume: LazyZarrVolume,
    *,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    threshold_abs: float = 0.5,
    nms_radius_um: float = 3.0,
    max_frames: int | None = None,
    deadline_at: float | None = None,
) -> list[Node]:
    if not volume.can_read_chunks:
        return []
    nodes: list[Node] = []
    total_frames = int(volume.shape[0])
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    next_id = 0
    for t in range(total_frames):
        if deadline_at is not None and time.monotonic() >= deadline_at:
            raise TimeoutError("Inference budget exhausted during frame detection")
        frame = volume.read_frame(t)
        peaks = detect_local_maxima_3d(frame, threshold_abs=threshold_abs)
        coords = [(z, y, x) for z, y, x, _ in peaks]
        scores = [score for *_, score in peaks]
        for idx in anisotropic_nms(coords, scores, scale=scale, radius_um=nms_radius_um):
            coord = refine_centroid_moment(frame, coords[idx])
            nodes.append(Node(next_id, t=t, z=coord[0], y=coord[1], x=coord[2]))
            next_id += 1
    return nodes
