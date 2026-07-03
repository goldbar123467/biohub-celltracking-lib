from __future__ import annotations

import csv

import pytest

from biohub_ct.data.schema import Edge, Graph, Node
from biohub_ct.submission.validator import SubmissionError, validate_submission
from biohub_ct.submission.writer import write_submission


def test_writer_emits_exact_schema_and_consecutive_ids(tmp_path):
    graph = Graph(
        nodes=[
            Node(node_id=10, t=0, z=1, y=2, x=3),
            Node(node_id=11, t=1, z=1, y=3, x=4),
        ],
        edges=[Edge(source_id=10, target_id=11)],
    )
    out = tmp_path / "submission.csv"

    write_submission({"sample_a": graph}, out)

    with out.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == [
        "id",
        "dataset",
        "row_type",
        "node_id",
        "t",
        "z",
        "y",
        "x",
        "source_id",
        "target_id",
    ]
    assert [int(r["id"]) for r in rows] == [0, 1, 2]
    assert rows[0]["row_type"] == "node"
    assert rows[-1]["row_type"] == "edge"
    assert validate_submission(out).row_count == 3


def test_validator_rejects_non_consecutive_ids(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n"
        "0,d,node,1,0,1,2,3,-1,-1\n"
        "2,d,node,2,1,1,2,3,-1,-1\n",
        encoding="utf-8",
    )

    with pytest.raises(SubmissionError, match="consecutive"):
        validate_submission(bad)


def test_empty_graph_gets_safe_fallback_node(tmp_path):
    out = tmp_path / "submission.csv"

    write_submission({"empty_ds": Graph()}, out)

    result = validate_submission(out, expected_datasets=["empty_ds"])
    assert result.row_count == 1
    with out.open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["row_type"] == "node"
    assert row["node_id"] == "0"

