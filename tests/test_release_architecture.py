from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from biohub_ct.data.schema import Edge, Graph, Node
from biohub_ct.data.splits import validate_split
from biohub_ct.detection.peaks import anisotropic_nms
from biohub_ct.linking.costs import physical_distance
from biohub_ct.linking.greedy import greedy_link
from biohub_ct.pipelines.baseline_classical import ClassicalConfig
from biohub_ct.pipelines.submission_pipeline import run_submission_pipeline
from biohub_ct.submission.validator import (
    SubmissionError,
    iter_submission_graphs,
    validate_submission,
)
from biohub_ct.submission.writer import write_submission


def chain():
    return Graph([Node(0, 0, 1, 1, 1), Node(1, 1, 1, 1, 1)], [Edge(0, 1)])


@pytest.mark.parametrize("edges", [[Edge(0, 1), Edge(0, 1)], [Edge(1, 0)], [Edge(0, 0)]])
def test_writer_rejects_invalid_edges_atomically(tmp_path, edges):
    output = tmp_path / "submission.csv"
    output.write_text("previous")
    with pytest.raises(SubmissionError):
        write_submission({"sample": Graph(chain().nodes_list, edges)}, output)
    assert output.read_text() == "previous"
    assert not list(tmp_path.glob("*.tmp"))


def test_csv_roundtrip_bounds_and_exact_coverage(tmp_path):
    path = tmp_path / "submission.csv"
    write_submission({"a": chain()}, path)
    assert next(iter_submission_graphs(path))[1].edges_set() == {(0, 1)}
    with pytest.raises(SubmissionError, match="coverage"):
        validate_submission(path, expected_datasets=["a", "missing"])
    with pytest.raises(SubmissionError, match="coverage"):
        validate_submission(path, expected_datasets=[])
    with pytest.raises(SubmissionError, match="bounds"):
        validate_submission(path, shapes={"a": (1, 2, 2, 2)})


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_node",
        "dangling",
        "discontiguous",
        "extra_column",
        "merge",
        "three_children",
        "gap",
    ],
)
def test_validator_checks_graph_not_only_schema(tmp_path, mutation):
    path = tmp_path / "submission.csv"
    write_submission({"a": chain(), "b": chain()}, path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    if mutation == "duplicate_node":
        rows[1]["node_id"] = "0"
    elif mutation == "dangling":
        rows[2]["target_id"] = "999"
    elif mutation == "discontiguous":
        rows[-1]["dataset"] = "a"
    elif mutation == "extra_column":
        columns = columns + ["extra"]
    elif mutation == "gap":
        rows[1]["t"] = "2"
    else:
        nodes = [Node(0, 0, 0, 0, 0), Node(1, 1, 0, 0, 0), Node(2, 1, 1, 1, 1), Node(3, 1, 2, 2, 2)]
        edges = [Edge(0, 1), Edge(0, 2), Edge(0, 3)]
        if mutation == "merge":
            nodes = [Node(0, 0, 0, 0, 0), Node(1, 0, 1, 1, 1), Node(2, 1, 0, 0, 0)]
            edges = [Edge(0, 2), Edge(1, 2)]
        with pytest.raises(SubmissionError):
            write_submission({"a": Graph(nodes, edges)}, path)
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SubmissionError):
        validate_submission(path)


def test_nms_matches_bruteforce_including_radius_boundary_and_ties():
    rng = np.random.default_rng(12)
    coords = [tuple(p) for p in rng.integers(0, 40, (400, 3))] + [(0, 0, 0), (0, 0, 4)]
    scores = list(rng.integers(0, 5, len(coords)))
    scale, radius = (1.625, 0.40625, 0.40625), 1.625
    kept = []
    for i in sorted(range(len(coords)), key=lambda i: scores[i], reverse=True):
        if all(np.linalg.norm((np.array(coords[i]) - coords[j]) * scale) > radius for j in kept):
            kept.append(i)
    assert anisotropic_nms(coords, scores, scale=scale, radius_um=radius) == kept


@pytest.mark.parametrize("divisions", [False, True])
def test_linker_matches_bruteforce(divisions):
    rng = np.random.default_rng(123)
    nodes = [
        Node(i, i // 80, *(int(x) for x in p)) for i, p in enumerate(rng.integers(0, 30, (240, 3)))
    ]
    scale, radius = (1.625, 0.40625, 0.40625), 4.0
    edges, used, counts = [], set(), {}
    for t in range(2):
        candidates = [
            (physical_distance(s, d, scale), s.node_id, d.node_id)
            for s in nodes
            if s.t == t
            for d in nodes
            if d.t == t + 1
            if physical_distance(s, d, scale) <= radius
        ]
        for _, s, d in sorted(candidates):
            if d not in used and counts.get(s, 0) < (2 if divisions else 1):
                edges.append(Edge(s, d))
                used.add(d)
                counts[s] = counts.get(s, 0) + 1
    assert greedy_link(nodes, scale=scale, max_distance=radius, allow_divisions=divisions) == edges


@pytest.mark.parametrize(
    "split",
    [
        {"train": ["a_1"], "val": ["a_2"]},
        {"train": ["a_1"], "val": ["b_missing"]},
        {"train": ["a_1"], "val": []},
        {"train": ["a_1"], "val": ["b_1", "b_1"]},
    ],
)
def test_split_rejects_leakage_and_missing_ids(split):
    with pytest.raises(ValueError):
        validate_split(split, {"a_1", "a_2", "b_1"})


def test_resume_cache_corruption_config_and_deadline(tmp_path):
    from test_pipeline_smoke import make_fake_zarr

    data = tmp_path / "data"
    data.mkdir()
    make_fake_zarr(data / "a.zarr")
    output = tmp_path / "submission.csv"
    run_submission_pipeline(data_dir=data, output_path=output, debug=True)
    original = output.read_bytes()
    run_submission_pipeline(data_dir=data, output_path=output, debug=True)
    report = json.loads(output.with_suffix(".manifest.json").read_text())
    assert report["datasets"][0]["resumed"] and report["metadata_smoke_only"]
    with pytest.raises(TimeoutError):
        run_submission_pipeline(data_dir=data, output_path=output, debug=True, deadline_seconds=0)
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="identity"):
        run_submission_pipeline(
            data_dir=data, output_path=output, debug=True, config=ClassicalConfig(threshold_abs=0.6)
        )
    artifact = tmp_path / "submission-graphs/a.json"
    saved = json.loads(artifact.read_text())
    saved["graph"]["nodes"][0]["x"] += 1
    artifact.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Corrupt"):
        run_submission_pipeline(data_dir=data, output_path=output, debug=True)
    assert output.read_bytes() == original


def test_package_is_deterministic_and_cells_compile(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "packager", root / "scripts/package_kaggle_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    one, two = tmp_path / "one", tmp_path / "two"
    assert module.build(one, "example/biohub") == module.build(two, "example/biohub")
    assert (one / "submission.ipynb").read_bytes() == (two / "submission.ipynb").read_bytes()
    notebook = json.loads((one / "submission.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<notebook>", "exec")
    metadata = json.loads((one / "kernel-metadata.json").read_text())
    assert metadata["enable_internet"] is False and metadata["enable_gpu"] is False


def test_interrupted_second_clip_resumes_first_and_matches_clean_run(tmp_path, monkeypatch):
    import biohub_ct.pipelines.submission_pipeline as pipeline
    from test_pipeline_smoke import make_fake_zarr

    data = tmp_path / "data"
    data.mkdir()
    for name in ("a", "b"):
        make_fake_zarr(data / f"{name}.zarr")
    output = tmp_path / "submission.csv"
    output.write_text("previous successful release")
    original = pipeline.run_classical_baseline

    def interrupted(record, **kwargs):
        if record.name == "b":
            raise RuntimeError("Simulated process failure on second clip")
        return original(record, **kwargs)

    monkeypatch.setattr(pipeline, "run_classical_baseline", interrupted)
    with pytest.raises(RuntimeError, match="Simulated"):
        pipeline.run_submission_pipeline(data_dir=data, output_path=output, debug=True)
    assert output.read_text() == "previous successful release"
    assert (tmp_path / "submission-graphs/a.json").is_file()
    monkeypatch.setattr(pipeline, "run_classical_baseline", original)
    pipeline.run_submission_pipeline(data_dir=data, output_path=output, debug=True)
    rows = json.loads(output.with_suffix(".manifest.json").read_text())["datasets"]
    assert [r["resumed"] for r in rows] == [True, False]
    clean = tmp_path / "clean.csv"
    pipeline.run_submission_pipeline(data_dir=data, output_path=clean, debug=True)
    assert output.read_bytes() == clean.read_bytes()
