from __future__ import annotations

import json

from biohub_ct.data.geff_io import read_geff_graph


def test_read_json_fallback_graph(tmp_path):
    geff = tmp_path / "sample.geff"
    geff.mkdir()
    (geff / "graph.json").write_text(
        json.dumps(
            {
                "nodes": [{"node_id": 1, "t": 0, "z": 1, "y": 2, "x": 3}],
                "edges": [],
                "estimated_number_of_nodes": 10,
            }
        ),
        encoding="utf-8",
    )

    graph, meta = read_geff_graph(geff)

    assert graph.num_nodes == 1
    assert meta.estimated_number_of_nodes == 10

