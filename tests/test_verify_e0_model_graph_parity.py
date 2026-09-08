from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_e0_model_graph_parity.py"
SPEC = importlib.util.spec_from_file_location("_verify_e0_model_graph_parity_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
parity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parity
SPEC.loader.exec_module(parity)


class FakeFrame:
    def __init__(self, columns: list[str], schema: dict[str, str], rows: list[dict[str, Any]]):
        self.columns = columns
        self.schema = schema
        self._rows = rows

    def iter_rows(self, *, named: bool):
        assert named
        return iter(self._rows)


class FakeGraph:
    def __init__(self, nodes: FakeFrame, edges: FakeFrame):
        self._nodes = nodes
        self._edges = edges

    def node_attrs(self) -> FakeFrame:
        return self._nodes

    def edge_attrs(self) -> FakeFrame:
        return self._edges

    def num_nodes(self) -> int:
        return len(self._nodes._rows)

    def num_edges(self) -> int:
        return len(self._edges._rows)


def make_graph(
    *,
    node_z: float = 1.0,
    edge_prob: float = 0.5,
    node_schema: str = "Float64",
    edge_id: int = 7,
    reverse_rows: bool = False,
) -> FakeGraph:
    nodes = [
        {"node_id": 1, "t": 0, "z": node_z, "y": 2.0, "x": 3.0},
        {"node_id": 2, "t": 1, "z": 4.0, "y": 5.0, "x": 6.0},
    ]
    edges = [
        {
            "source_id": 1,
            "target_id": 2,
            "edge_id": edge_id,
            "edge_prob": edge_prob,
            "edge_dist": 1.0,
        }
    ]
    if reverse_rows:
        nodes.reverse()
        edges.reverse()
    return FakeGraph(
        FakeFrame(
            ["x", "node_id", "z", "t", "y"],
            {"node_id": "Int64", "t": "Int64", "z": node_schema, "y": "Float64", "x": "Float64"},
            nodes,
        ),
        FakeFrame(
            ["edge_dist", "target_id", "edge_id", "source_id", "edge_prob"],
            {
                "source_id": "Int64",
                "target_id": "Int64",
                "edge_id": "Int64",
                "edge_prob": "Float64",
                "edge_dist": "Float64",
            },
            edges,
        ),
    )


def test_graph_canonicalization_detects_value_and_schema_changes() -> None:
    baseline = parity.canonical_graph(make_graph())
    assert baseline == parity.canonical_graph(make_graph(reverse_rows=True))
    assert baseline != parity.canonical_graph(make_graph(node_z=np.nextafter(1.0, 2.0)))
    assert baseline != parity.canonical_graph(make_graph(edge_prob=np.nextafter(0.5, 1.0)))
    assert baseline != parity.canonical_graph(make_graph(node_schema="Float32"))


def test_geff_readback_projection_ignores_only_storage_identity_and_schema() -> None:
    baseline = parity.canonical_graph(make_graph())
    changed_edge_id = parity.canonical_graph(make_graph(edge_id=99))
    changed_schema = parity.canonical_graph(make_graph(node_schema="Float32"))
    changed_value = parity.canonical_graph(make_graph(edge_prob=np.nextafter(0.5, 1.0)))

    baseline_semantics = parity.canonical_graph_semantics(baseline)
    assert baseline_semantics == parity.canonical_graph_semantics(changed_edge_id)
    assert baseline_semantics == parity.canonical_graph_semantics(changed_schema)
    assert baseline_semantics != parity.canonical_graph_semantics(changed_value)


def test_candidate_comparator_preserves_order_dtype_and_float_bits() -> None:
    coords = np.array([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=np.int16)
    edges = [(0, 1, 0.5, 1.0), (1, 0, np.nextafter(0.5, 1.0), 2.0)]
    baseline = parity.candidate_arrays(coords, edges, np)
    reversed_edges = parity.candidate_arrays(coords, list(reversed(edges)), np)

    assert baseline["coordinates"].dtype == np.dtype("<i2")
    assert baseline["edge_probability"].dtype == np.dtype("<f8")
    assert baseline["edge_probability"].tobytes() != reversed_edges["edge_probability"].tobytes()
    with pytest.raises(parity.ContractError, match="native int16"):
        parity.candidate_arrays(coords.astype(np.int32), edges, np)
    with pytest.raises(parity.ContractError, match="nonfinite"):
        parity.candidate_arrays(coords, [(0, 1, float("nan"), 1.0)], np)


class FakeArray:
    def __init__(self, values: Any, attrs: dict[str, Any] | None = None):
        self._values = np.asarray(values)
        self.attrs = attrs or {}

    def __getitem__(self, key: Any) -> np.ndarray:
        assert key is Ellipsis
        return self._values


class FakeGroup:
    def __init__(
        self,
        *,
        attrs: dict[str, Any] | None = None,
        arrays: dict[str, FakeArray] | None = None,
        groups: dict[str, FakeGroup] | None = None,
    ):
        self.attrs = attrs or {}
        self._arrays = arrays or {}
        self._groups = groups or {}

    def arrays(self):
        return iter(reversed(list(self._arrays.items())))

    def groups(self):
        return iter(reversed(list(self._groups.items())))


class FakeZarr:
    def __init__(self, root: FakeGroup):
        self.root = root

    def open_group(self, path: str, mode: str) -> FakeGroup:
        assert path == "fixture.geff"
        assert mode == "r"
        return self.root


def geff_fixture(*, edge_value: float = 0.5, directed: bool = True) -> FakeGroup:
    return FakeGroup(
        attrs={"geff": {"directed": directed, "axes": [{"name": "t", "scale": 1.0}]}},
        arrays={"node_ids": FakeArray([1, 2], {"dimension_names": ["node"]})},
        groups={
            "edges": FakeGroup(
                arrays={"probability": FakeArray([edge_value], {"unit": "probability"})}
            )
        },
    )


def test_logical_geff_comparator_detects_array_and_attribute_changes() -> None:
    baseline = parity.canonical_geff(Path("fixture.geff"), np, FakeZarr(geff_fixture()))
    changed_array = parity.canonical_geff(
        Path("fixture.geff"), np, FakeZarr(geff_fixture(edge_value=np.nextafter(0.5, 1.0)))
    )
    changed_attrs = parity.canonical_geff(
        Path("fixture.geff"), np, FakeZarr(geff_fixture(directed=False))
    )

    assert baseline != changed_array
    assert baseline != changed_attrs
    assert set(baseline["arrays"]) == {"node_ids", "edges/probability"}
