from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biohub_ct.data.geff_io import GeffMetadata, read_geff_graph
from biohub_ct.data.paths import DatasetRecord, discover_datasets
from biohub_ct.data.schema import Graph, Node
from biohub_ct.data.zarr_io import LazyZarrVolume, open_zarr_volume
from biohub_ct.metrics.edge import scaled_distance


@dataclass(frozen=True)
class DatasetEdaRow:
    dataset: str
    shape: str
    chunks: str
    dtype: str
    intensity_quantiles: str
    gt_nodes: int
    gt_edges: int
    gt_divisions: int
    estimated_nodes: int | None
    per_frame_gt_counts: str
    edge_disp_um_p50: float
    edge_disp_um_p95: float
    daughter_dist_um_p50: float
    nn_spacing_um_p50: float
    sparse_pattern: str


def write_eda_report(
    data_dir: Path | str,
    output_path: Path | str,
    *,
    max_intensity_frames: int = 3,
) -> list[DatasetEdaRow]:
    rows = collect_eda_rows(data_dir, max_intensity_frames=max_intensity_frames)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_format_markdown(rows, data_dir), encoding="utf-8")
    return rows


def collect_eda_rows(
    data_dir: Path | str,
    *,
    max_intensity_frames: int = 3,
) -> list[DatasetEdaRow]:
    records = discover_datasets(data_dir, require_geff=True)
    rows: list[DatasetEdaRow] = []
    for record in records:
        volume = open_zarr_volume(record.zarr_path)
        graph, meta = _read_graph(record)
        rows.append(_row(record, volume, graph, meta, max_intensity_frames=max_intensity_frames))
    return rows


def _read_graph(record: DatasetRecord) -> tuple[Graph, GeffMetadata]:
    if record.geff_path is None:
        return Graph(), GeffMetadata()
    return read_geff_graph(record.geff_path)


def _row(
    record: DatasetRecord,
    volume: LazyZarrVolume,
    graph: Graph,
    meta: GeffMetadata,
    *,
    max_intensity_frames: int,
) -> DatasetEdaRow:
    edge_disps = _edge_displacements(graph, volume.scale)
    daughter_dists = _daughter_distances(graph, volume.scale)
    nn_spacings = _nearest_neighbor_spacings(graph, volume.scale)
    return DatasetEdaRow(
        dataset=record.name,
        shape=str(tuple(volume.shape)),
        chunks=str(tuple(volume.chunks) if volume.chunks else None),
        dtype=volume.dtype,
        intensity_quantiles=_intensity_quantiles(volume, max_frames=max_intensity_frames),
        gt_nodes=graph.num_nodes,
        gt_edges=graph.num_edges,
        gt_divisions=len(graph.dividing_nodes()),
        estimated_nodes=meta.estimated_number_of_nodes,
        per_frame_gt_counts=_per_frame_counts(graph),
        edge_disp_um_p50=_percentile(edge_disps, 50),
        edge_disp_um_p95=_percentile(edge_disps, 95),
        daughter_dist_um_p50=_percentile(daughter_dists, 50),
        nn_spacing_um_p50=_percentile(nn_spacings, 50),
        sparse_pattern=_sparse_pattern(graph),
    )


def _intensity_quantiles(volume: LazyZarrVolume, *, max_frames: int) -> str:
    if not volume.can_read_chunks:
        return "not_available_metadata_only"
    samples = []
    for t in range(min(int(volume.shape[0]), max_frames)):
        frame = volume.read_frame(t)
        samples.append(np.asarray(frame).ravel()[:: max(1, frame.size // 10000)])
    if not samples:
        return "not_available_empty"
    arr = np.concatenate(samples)
    qs = np.percentile(arr, [0, 1, 50, 99, 100])
    return "q0={:.3g};q1={:.3g};q50={:.3g};q99={:.3g};q100={:.3g}".format(*qs)


def _edge_displacements(graph: Graph, scale: tuple[float, float, float]) -> list[float]:
    out = []
    for edge in graph.edges_list:
        if edge.source_id in graph.nodes and edge.target_id in graph.nodes:
            out.append(scaled_distance(graph.node(edge.source_id), graph.node(edge.target_id), scale))
    return out


def _daughter_distances(graph: Graph, scale: tuple[float, float, float]) -> list[float]:
    out: list[float] = []
    for div_node in graph.dividing_nodes():
        children = graph.successors(div_node)
        for i, left in enumerate(children):
            for right in children[i + 1 :]:
                out.append(scaled_distance(graph.node(left), graph.node(right), scale))
    return out


def _nearest_neighbor_spacings(graph: Graph, scale: tuple[float, float, float]) -> list[float]:
    out: list[float] = []
    for nodes in graph.nodes_by_time().values():
        if len(nodes) < 2:
            continue
        for i, node in enumerate(nodes):
            distances = [
                scaled_distance(node, other, scale)
                for j, other in enumerate(nodes)
                if i != j
            ]
            if distances:
                out.append(min(distances))
    return out


def _per_frame_counts(graph: Graph, *, limit: int = 20) -> str:
    counts = {t: len(nodes) for t, nodes in graph.nodes_by_time().items()}
    items = [f"{t}:{counts[t]}" for t in sorted(counts)[:limit]]
    suffix = "" if len(counts) <= limit else ";..."
    return ";".join(items) + suffix


def _sparse_pattern(graph: Graph) -> str:
    by_t = graph.nodes_by_time()
    if not by_t:
        return "no_gt_nodes"
    times = sorted(by_t)
    empty_inside = sum(1 for t in range(times[0], times[-1] + 1) if t not in by_t)
    return f"frames={len(times)};t_min={times[0]};t_max={times[-1]};empty_inside={empty_inside}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _format_markdown(rows: list[DatasetEdaRow], data_dir: Path | str) -> str:
    header = (
        "# Train EDA\n\n"
        f"Data dir: `{data_dir}`\n\n"
        "| dataset | shape | chunks | dtype | intensity_quantiles | gt_nodes | gt_edges | "
        "gt_divisions | estimated_nodes | per_frame_gt_counts | edge_disp_um_p50 | "
        "edge_disp_um_p95 | daughter_dist_um_p50 | nn_spacing_um_p50 | sparse_pattern |\n"
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |\n"
    )
    body = []
    for row in rows:
        body.append(
            "| {dataset} | {shape} | {chunks} | {dtype} | {intensity_quantiles} | "
            "{gt_nodes} | {gt_edges} | {gt_divisions} | {estimated_nodes} | "
            "{per_frame_gt_counts} | {edge_disp_um_p50:.6g} | {edge_disp_um_p95:.6g} | "
            "{daughter_dist_um_p50:.6g} | {nn_spacing_um_p50:.6g} | {sparse_pattern} |".format(
                **row.__dict__
            )
        )
    if not body:
        body.append("| no_datasets_found | - | - | - | - | 0 | 0 | 0 |  | - | nan | nan | nan | nan | - |")
    return header + "\n".join(body) + "\n"

