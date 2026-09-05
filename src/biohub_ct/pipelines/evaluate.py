from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from biohub_ct.data.geff_io import read_geff_graph
from biohub_ct.data.paths import discover_datasets
from biohub_ct.data.splits import validate_split
from biohub_ct.data.zarr_io import open_zarr_volume
from biohub_ct.metrics.division import evaluate_divisions
from biohub_ct.metrics.edge import evaluate_edges, match_nodes
from biohub_ct.metrics.official_adapter import evaluate_official
from biohub_ct.pipelines.baseline_classical import ClassicalConfig, run_classical_baseline
from biohub_ct.pipelines.submission_pipeline import environment_info, input_identity, source_digest
from biohub_ct.submission.validator import iter_submission_graphs
from biohub_ct.submission.writer import write_submission

OFFICIAL_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
OFFICIAL_METRICS_SHA256 = "cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444"


@dataclass(frozen=True)
class FoldEvaluationSummary:
    dataset_count: int
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    adjusted_edge_jaccard: float
    division_jaccard: float
    score: float


def official_module():
    from tracking_cellmot import metrics

    code = Path(inspect.getfile(metrics)).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(code).hexdigest() != OFFICIAL_METRICS_SHA256:
        raise RuntimeError(f"Official metric source differs from pinned commit {OFFICIAL_COMMIT}")
    return metrics


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_fold(
    *,
    data_dir,
    splits_path,
    fold,
    output_path,
    pipeline="classical",
    metric_backend="official",
    config=None,
    metadata_smoke_only=False,
):
    if pipeline != "classical" or metric_backend not in ("official", "local-probe"):
        raise ValueError("Unsupported pipeline or metric backend")
    if metadata_smoke_only and metric_backend != "local-probe":
        raise ValueError("Metadata fixtures cannot be officially scored")
    metric = official_module() if metric_backend == "official" else None
    cfg = config or ClassicalConfig()
    splits = json.loads(Path(splits_path).read_text())
    split = splits[fold]
    records = {r.name: r for r in discover_datasets(data_dir, require_geff=True)}
    validate_split(split, set(records))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions = out.parent / (out.stem + "-predictions")
    predictions.mkdir(exist_ok=True)
    rows = []
    for name in split["val"]:
        record = records[name]
        gt, meta = read_geff_graph(record.geff_path)
        estimate = meta.estimated_number_of_nodes
        if estimate is None or not math.isfinite(estimate) or estimate <= 0:
            raise ValueError(f"{name} lacks a positive estimated_number_of_nodes")
        volume = open_zarr_volume(record.zarr_path, allow_metadata_only=metadata_smoke_only)
        start = time.monotonic()
        pred = run_classical_baseline(record, config=cfg, debug=metadata_smoke_only)
        csv_path = predictions / (name + ".csv")
        write_submission(
            {name: pred}, csv_path, expected_datasets=[name], shapes={name: volume.shape}
        )
        _, restored = next(iter_submission_graphs(csv_path))
        if pred.nodes_list != restored.nodes_list or pred.edges_set() != restored.edges_set():
            raise RuntimeError("Prediction CSV round-trip changed graph")
        inference_s = time.monotonic() - start
        if metric is not None:
            result = evaluate_official(restored, gt, scale=volume.scale, total_true_nodes=estimate)
            row = asdict(result)
            row["adj_edge_jaccard"] = row.pop("adjusted_edge_jaccard")
        else:
            edge = evaluate_edges(restored, gt, total_true_nodes=estimate)
            div = evaluate_divisions(restored, gt)
            row = asdict(edge)
            row["adj_edge_jaccard"] = row.pop("adjusted_edge_jaccard")
            row.update(
                division_tp=div.tp,
                division_fp=div.fp,
                division_fn=div.fn,
                node_recall=len(match_nodes(restored, gt).gt_to_pred) / gt.num_nodes
                if gt.num_nodes
                else float("nan"),
            )
        row.update(
            dataset=name,
            runtime_s=inference_s,
            annotated_nodes=gt.num_nodes,
            estimated_nodes=estimate,
            node_count_ratio=pred.num_nodes / estimate,
            image_identity=input_identity(record.zarr_path),
            geff_identity=input_identity(record.geff_path),
            csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        )
        rows.append(row)
        print(json.dumps(_json_safe(row)), flush=True)
    if metric is not None:
        aggregate = metric.summarise(rows)
    else:
        weight = sum(r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows)
        adj = (
            sum((r["edge_tp"] + r["edge_fp"] + r["edge_fn"]) * r["adj_edge_jaccard"] for r in rows)
            / weight
            if weight
            else float("nan")
        )
        d = sum(r["division_tp"] + r["division_fp"] + r["division_fn"] for r in rows)
        div = sum(r["division_tp"] for r in rows) / d if d else float("nan")
        aggregate = {
            "adj_edge_jaccard": adj,
            "division_jaccard": div,
            "score": adj + (0.1 * div if d else 0),
        }
    totals = {
        key: sum(r[key] for r in rows)
        for key in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
    }
    summary = FoldEvaluationSummary(
        len(rows),
        **totals,
        adjusted_edge_jaccard=aggregate["adj_edge_jaccard"],
        division_jaccard=aggregate["division_jaccard"],
        score=aggregate["score"],
    )
    report = {
        "backend": metric_backend,
        "metadata_smoke_only": metadata_smoke_only,
        "official_commit": OFFICIAL_COMMIT if metric else None,
        "source_digest": source_digest(),
        **environment_info(),
        "config": asdict(cfg),
        "split": split,
        "fold": fold,
        "split_sha256": hashlib.sha256(Path(splits_path).read_bytes()).hexdigest(),
        "summary": asdict(summary),
        "official_aggregate": aggregate,
        "datasets": rows,
    }
    out.with_suffix(".json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    text = [
        "# Classical baseline evaluation",
        "",
        f"Backend: {metric_backend}; metadata smoke only: {metadata_smoke_only}",
        "",
        f"Fold: {fold}; score: {summary.score:.6g}",
        "",
        "| dataset | runtime_s | nodes | estimated | node_count_ratio | node_recall | edge_tp | edge_fp | edge_fn | adjusted_edge_jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        text.append(
            "| {dataset} | {runtime_s:.3f} | {num_pred_nodes} | {estimated_nodes} | {node_count_ratio:.4f} | {node_recall:.4f} | {edge_tp} | {edge_fp} | {edge_fn} | {adj_edge_jaccard:.6f} |".format(
                **r
            )
        )
    out.write_text("\n".join(text) + "\n")
    return summary
