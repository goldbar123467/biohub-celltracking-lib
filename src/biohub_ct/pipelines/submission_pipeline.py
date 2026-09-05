from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from biohub_ct.data.paths import discover_datasets
from biohub_ct.data.schema import Edge, Graph, Node
from biohub_ct.data.zarr_io import open_zarr_volume
from biohub_ct.pipelines.baseline_classical import ClassicalConfig, run_classical_baseline
from biohub_ct.submission.validator import validate_graph
from biohub_ct.submission.writer import write_submission


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_digest():
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            p.relative_to(root).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in sorted(root.rglob("*.py"))
        }
    )


def environment_info() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "zarr", "scipy", "tracking-cellmot", "tracksdata", "polars"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    peak_rss = None
    if platform.system() == "Linux":
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "versions": versions,
        "platform": platform.system(),
        "cpu_count": os.cpu_count(),
        "peak_rss_bytes": peak_rss,
    }


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def input_identity(path):
    # Inputs are immutable. Include metadata content and chunk size/mtime inventory
    # so a partial download or changed local chunk invalidates cached inference.
    return digest(
        [
            (
                p.relative_to(path).as_posix(),
                p.stat().st_size,
                p.stat().st_mtime_ns,
                hashlib.sha256(p.read_bytes()).hexdigest()
                if p.name in ("zarr.json", ".zattrs")
                else None,
            )
            for p in sorted(path.rglob("*"))
            if p.is_file()
        ]
    )


def run_submission_pipeline(
    *, data_dir, output_path, debug=False, config=None, cache_dir=None, deadline_seconds=None
):
    records = discover_datasets(data_dir, require_geff=False)
    if not records:
        raise FileNotFoundError(f"No .zarr datasets found under {data_dir}")
    cfg = config or ClassicalConfig()
    started = time.monotonic()
    deadline_at = started + deadline_seconds if deadline_seconds is not None else None
    source = source_digest()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else out.parent / (out.stem + "-graphs")
    cache.mkdir(parents=True, exist_ok=True)
    shapes, rows = {}, []
    provenance = {"source_digest": source, "config": asdict(cfg), "metadata_smoke_only": debug}

    def generate():
        for record in records:
            if deadline_seconds is not None and time.monotonic() - started >= deadline_seconds:
                raise TimeoutError("Inference budget exhausted; completed graphs retained")
            volume = open_zarr_volume(
                record.zarr_path, allow_metadata_only=debug, require_complete_chunks=not debug
            )
            shapes[record.name] = volume.shape
            identity = input_identity(record.zarr_path)
            key = digest({**provenance, "dataset": record.name, "input": identity})
            artifact = cache / (record.name + ".json")
            begin = time.monotonic()
            resumed = False
            if artifact.exists():
                saved = json.loads(artifact.read_text())
                if saved["key"] != key:
                    raise ValueError(
                        f"Cache identity changed for {record.name}; use a new cache directory"
                    )
                if digest(saved["graph"]) != saved["graph_sha256"]:
                    raise ValueError(f"Corrupt cached graph: {artifact}")
                graph = Graph(
                    [Node(**n) for n in saved["graph"]["nodes"]],
                    [Edge(**e) for e in saved["graph"]["edges"]],
                )
                diagnostics = saved["diagnostics"]
                resumed = True
            else:
                graph = run_classical_baseline(
                    record, config=cfg, debug=debug, deadline_at=deadline_at
                )
                diagnostics = graph.inference_diagnostics
                validate_graph(graph, shape=volume.shape)
                payload = {
                    "nodes": [asdict(n) for n in graph.nodes_list],
                    "edges": [asdict(e) for e in graph.edges_list],
                }
                atomic_json(
                    artifact,
                    {
                        "key": key,
                        "graph": payload,
                        "graph_sha256": digest(payload),
                        "diagnostics": diagnostics,
                    },
                )
            validate_graph(graph, shape=volume.shape)
            row = {
                "dataset": record.name,
                "input_identity": identity,
                "shape": volume.shape,
                "nodes": graph.num_nodes,
                "edges": graph.num_edges,
                "resumed": resumed,
                "runtime_s": time.monotonic() - begin,
                "nodes_by_frame": dict(Counter(n.t for n in graph.nodes_list)),
                **diagnostics,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            yield record.name, graph
        if deadline_seconds is not None and time.monotonic() - started >= deadline_seconds:
            raise TimeoutError("Inference budget exhausted; completed graphs retained")

    write_submission(generate(), out, expected_datasets=[r.name for r in records], shapes=shapes)
    report = {
        **provenance,
        "status": "complete",
        "datasets": rows,
        **environment_info(),
        "elapsed_s": time.monotonic() - started,
        "submission_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    atomic_json(out.with_suffix(".manifest.json"), report)
    return out
