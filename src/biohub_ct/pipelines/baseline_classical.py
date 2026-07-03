from __future__ import annotations

from dataclasses import dataclass

from biohub_ct.config import DEFAULT_SCALE
from biohub_ct.data.paths import DatasetRecord
from biohub_ct.data.schema import Edge, Graph, Node
from biohub_ct.data.zarr_io import open_zarr_volume
from biohub_ct.detection.inference import detect_nodes_in_volume
from biohub_ct.linking.greedy import greedy_link


@dataclass(frozen=True)
class ClassicalConfig:
    threshold_abs: float = 0.5
    nms_radius_um: float = 3.0
    link_distance_um: float = 8.0
    debug_max_frames: int | None = 3


def run_classical_baseline(
    record: DatasetRecord,
    *,
    config: ClassicalConfig | None = None,
    debug: bool = False,
) -> Graph:
    cfg = config or ClassicalConfig()
    volume = open_zarr_volume(record.zarr_path)
    max_frames = cfg.debug_max_frames if debug else None
    nodes = detect_nodes_in_volume(
        volume,
        scale=volume.scale,
        threshold_abs=cfg.threshold_abs,
        nms_radius_um=cfg.nms_radius_um,
        max_frames=max_frames,
    )
    if not nodes:
        nodes = [_fallback_node(volume.shape)]
    edges = greedy_link(nodes, scale=volume.scale or DEFAULT_SCALE, max_distance=cfg.link_distance_um)
    return Graph(nodes=nodes, edges=_valid_edges(edges, nodes))


def _fallback_node(shape: tuple[int, ...]) -> Node:
    if len(shape) == 4:
        _, z, y, x = shape
        return Node(0, t=0, z=max(0, z // 2), y=max(0, y // 2), x=max(0, x // 2))
    return Node(0, t=0, z=0, y=0, x=0)


def _valid_edges(edges: list[Edge], nodes: list[Node]) -> list[Edge]:
    node_ids = {n.node_id for n in nodes}
    return [edge for edge in edges if edge.source_id in node_ids and edge.target_id in node_ids]

