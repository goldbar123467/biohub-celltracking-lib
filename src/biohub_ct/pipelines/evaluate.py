from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from biohub_ct.data.geff_io import read_geff_graph
from biohub_ct.data.paths import DatasetRecord, discover_datasets
from biohub_ct.metrics.division import evaluate_divisions
from biohub_ct.metrics.edge import evaluate_edges
from biohub_ct.pipelines.baseline_classical import run_classical_baseline


@dataclass(frozen=True)
class DatasetEvaluationRow:
    dataset: str
    runtime_s: float
    pred_nodes: int
    target_nodes: int
    node_count_ratio: float
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    adjusted_edge_jaccard: float
    division_jaccard: float
    score: float


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


def evaluate_fold(
    *,
    data_dir: Path | str,
    splits_path: Path | str,
    fold: str,
    output_path: Path | str,
    pipeline: str = "classical",
) -> FoldEvaluationSummary:
    if pipeline != "classical":
        raise ValueError("Only the classical pipeline is allowed in this stage")
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    if fold not in splits:
        raise KeyError(f"Fold {fold!r} not found in {splits_path}")
    records = {record.name: record for record in discover_datasets(data_dir, require_geff=True)}
    rows = [_evaluate_dataset(records[name]) for name in splits[fold]["val"] if name in records]
    summary = _summarize(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_format_report(rows, summary, data_dir, splits_path, fold, pipeline), encoding="utf-8")
    return summary


def _evaluate_dataset(record: DatasetRecord) -> DatasetEvaluationRow:
    assert record.geff_path is not None
    gt, meta = read_geff_graph(record.geff_path)
    target_nodes = meta.estimated_number_of_nodes or gt.num_nodes
    start = time.perf_counter()
    pred = run_classical_baseline(record, debug=False)
    runtime_s = time.perf_counter() - start
    edge = evaluate_edges(pred, gt, total_true_nodes=target_nodes)
    div = evaluate_divisions(pred, gt)
    division_term = 0.0 if div.division_jaccard != div.division_jaccard else 0.1 * div.division_jaccard
    score = edge.adjusted_edge_jaccard + division_term
    return DatasetEvaluationRow(
        dataset=record.name,
        runtime_s=runtime_s,
        pred_nodes=pred.num_nodes,
        target_nodes=int(target_nodes),
        node_count_ratio=pred.num_nodes / target_nodes if target_nodes else float("nan"),
        edge_tp=edge.edge_tp,
        edge_fp=edge.edge_fp,
        edge_fn=edge.edge_fn,
        division_tp=div.tp,
        division_fp=div.fp,
        division_fn=div.fn,
        adjusted_edge_jaccard=edge.adjusted_edge_jaccard,
        division_jaccard=div.division_jaccard,
        score=score,
    )


def _summarize(rows: list[DatasetEvaluationRow]) -> FoldEvaluationSummary:
    edge_tp = sum(r.edge_tp for r in rows)
    edge_fp = sum(r.edge_fp for r in rows)
    edge_fn = sum(r.edge_fn for r in rows)
    div_tp = sum(r.division_tp for r in rows)
    div_fp = sum(r.division_fp for r in rows)
    div_fn = sum(r.division_fn for r in rows)
    weights = [r.edge_tp + r.edge_fp + r.edge_fn for r in rows]
    total_weight = sum(weights)
    if total_weight:
        adj = sum(w * r.adjusted_edge_jaccard for w, r in zip(weights, rows)) / total_weight
    else:
        adj = float("nan")
    div_denom = div_tp + div_fp + div_fn
    div_j = div_tp / div_denom if div_denom else float("nan")
    div_term = 0.0 if div_j != div_j else 0.1 * div_j
    score = adj + div_term
    return FoldEvaluationSummary(
        dataset_count=len(rows),
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        division_tp=div_tp,
        division_fp=div_fp,
        division_fn=div_fn,
        adjusted_edge_jaccard=adj,
        division_jaccard=div_j,
        score=score,
    )


def _format_report(
    rows: list[DatasetEvaluationRow],
    summary: FoldEvaluationSummary,
    data_dir: Path | str,
    splits_path: Path | str,
    fold: str,
    pipeline: str,
) -> str:
    out = [
        "# Classical Baseline Fold Evaluation",
        "",
        f"Data dir: `{data_dir}`",
        f"Splits: `{splits_path}`",
        f"Fold: `{fold}`",
        f"Pipeline: `{pipeline}`",
        "",
        "## Summary",
        "",
        f"- datasets: {summary.dataset_count}",
        f"- adjusted_edge_jaccard: {summary.adjusted_edge_jaccard:.6g}",
        f"- division_jaccard: {summary.division_jaccard:.6g}",
        f"- final_score: {summary.score:.6g}",
        f"- edge_tp/edge_fp/edge_fn: {summary.edge_tp}/{summary.edge_fp}/{summary.edge_fn}",
        f"- division_tp/division_fp/division_fn: {summary.division_tp}/{summary.division_fp}/{summary.division_fn}",
        "",
        "## Per Dataset",
        "",
        "| dataset | runtime_s | pred_nodes | target_nodes | node_count_ratio | edge_tp | edge_fp | edge_fn | division_tp | division_fp | division_fn | adjusted_edge_jaccard | division_jaccard | score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        out.append(
            "| {dataset} | {runtime_s:.6g} | {pred_nodes} | {target_nodes} | "
            "{node_count_ratio:.6g} | {edge_tp} | {edge_fp} | {edge_fn} | "
            "{division_tp} | {division_fp} | {division_fn} | "
            "{adjusted_edge_jaccard:.6g} | {division_jaccard:.6g} | {score:.6g} |".format(
                **row.__dict__
            )
        )
    out.extend(["", "## Observed Failure Modes", ""])
    for item in _failure_modes(rows):
        out.append(f"- {item}")
    out.append("")
    return "\n".join(out)


def _failure_modes(rows: list[DatasetEvaluationRow]) -> list[str]:
    if not rows:
        return ["No validation datasets were evaluated."]
    modes: list[str] = []
    if any(r.pred_nodes == 1 and r.edge_fn > 0 for r in rows):
        modes.append("Metadata-only or no-detection fallback produced one node and missed GT edges.")
    if sum(r.edge_fn for r in rows) > sum(r.edge_tp for r in rows):
        modes.append("Edge recall is the dominant failure: false negatives exceed true positives.")
    if sum(r.edge_fp for r in rows) > 0:
        modes.append("Some predicted edges touch annotated regions but do not match GT edges.")
    ratios = [r.node_count_ratio for r in rows if r.node_count_ratio == r.node_count_ratio]
    if ratios and sum(ratios) / len(ratios) < 0.75:
        modes.append("Predicted node density is below target node count.")
    if ratios and sum(ratios) / len(ratios) > 1.25:
        modes.append("Predicted node density is above target node count.")
    if sum(r.division_fn for r in rows) > 0:
        modes.append("Division recall is incomplete.")
    while len(modes) < 5:
        modes.append("Need real image data and visual failure artifacts to classify remaining errors.")
    return modes[:5]

