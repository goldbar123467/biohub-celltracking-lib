from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from biohub_ct.campaign import e0_support_telemetry as support_telemetry

PINNED_RECONSTRUCTION = Path("work/e0-reference/public-patched-support-root.py")


def _load_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.delenv("BIOHUB_E0_TELEMETRY_DIR", raising=False)
    namespace: dict[str, object] = {"__name__": "test_e0_runtime"}
    source = support_telemetry.runtime_helper_source()
    exec(compile(source, "<test-e0-runtime>", "exec"), namespace)  # noqa: S102
    return namespace


def _records(process_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (process_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_runtime_records_exact_counts_durations_coordinates_and_peaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load_runtime(monkeypatch)
    telemetry_type = namespace["Telemetry"]

    class Clock:
        value = 0

        def __call__(self) -> int:
            self.value += 100
            return self.value

    class FakeCuda:
        reset_calls = 0

        @staticmethod
        def is_available() -> bool:
            return True

        @classmethod
        def reset_peak_memory_stats(cls) -> None:
            cls.reset_calls += 1

        @staticmethod
        def max_memory_allocated() -> int:
            return 1234

        @staticmethod
        def max_memory_reserved() -> int:
            return 5678

        @staticmethod
        def synchronize() -> None:
            return None

    telemetry = telemetry_type(tmp_path, clock_ns=Clock())
    telemetry.attach_torch(types.SimpleNamespace(cuda=FakeCuda))
    telemetry.begin_invocation(
        data_root=tmp_path / "competition-test",
        output_dir=tmp_path / "predictions",
        method="public-reference",
        fold=0,
        test_names=["embryo.zarr"],
    )
    telemetry.begin_dataset("embryo.zarr", tmp_path / "competition-test/embryo.zarr")

    for stage in (
        "data_read",
        "encode_tta",
        "detector_extraction",
        "pair_score",
        "threshold",
        "graph_build",
        "ilp",
        "geff",
    ):
        started = telemetry.started(stage)
        telemetry.add_duration(stage, started)
    telemetry.add_pair_counts(pair_universe=12, threshold_passing=3)
    telemetry.add_pair_counts(pair_universe=2, threshold_passing=0)
    telemetry.set_ilp("returned")
    telemetry.record_retention(
        {
            "dataset": "embryo.zarr",
            "frame": 0,
            "primary_candidates": 2,
            "blended_candidates": 2,
            "retention": 1.0,
            "minimum_retention": 0.9,
            "use_primary": False,
        }
    )

    coordinate_bytes = bytes(range(16))
    returned_artifact = telemetry.capture_coordinates(
        dataset="embryo.zarr",
        payload=coordinate_bytes,
        shape=(2, 4),
        dtype="<i2",
    )
    with pytest.raises(FileExistsError, match="already exists for this invocation"):
        telemetry.capture_coordinates(
            dataset="embryo.zarr",
            payload=b"\xff" * 16,
            shape=(2, 4),
            dtype="<i2",
        )
    telemetry.finish_dataset(detected_nodes=2, pre_ilp_edges=3, output_edges=2)
    telemetry.finish_invocation()
    telemetry.finalize_process()
    telemetry.finalize_process()

    assert FakeCuda.reset_calls == 1
    records = _records(telemetry.process_dir)
    assert [record["record_type"] for record in records] == [
        "invocation_start",
        "retention",
        "coordinate_artifact",
        "dataset_summary",
        "invocation_finish",
        "process_summary",
    ]
    assert Path(records[1]["dataset_path"]) == (
        tmp_path / "competition-test" / "embryo.zarr"
    ).resolve()
    summary = records[3]
    assert summary["counts"]["pair_universe"] == {
        "status": "available",
        "unit": "ordered_source_target_pairs",
        "value": 14,
    }
    assert summary["counts"]["threshold_passing_edge_candidates"]["value"] == 3
    assert summary["ilp"] == {"status": "returned"}
    assert all(
        duration
        == {
            "status": "available",
            "value": 0.0000001,
            "unit": "seconds",
            "duration_ns": 100,
        }
        for duration in summary["durations"].values()
    )

    coordinate_record = records[2]
    artifact = coordinate_record["artifact"]
    assert returned_artifact == artifact
    coordinate_path = tmp_path.parent / artifact["path"]
    assert coordinate_path.read_bytes() == coordinate_bytes
    assert artifact["sha256"] == hashlib.sha256(coordinate_bytes).hexdigest()
    assert artifact["shape"] == [2, 4]
    assert artifact["dtype"] == "<i2"
    assert coordinate_record["coordinate_sha256"] == artifact["sha256"]
    assert coordinate_record["rows"] == 2
    assert coordinate_record["frame_counts"] == [[256, 1], [2312, 1]]
    assert coordinate_record["coordinate_space"] == "original_voxel_zyx"
    assert not Path(artifact["path"]).is_absolute()

    process = records[-1]
    assert process["cuda_peak_allocated"] == {
        "status": "available",
        "value": 1234,
        "unit": "bytes",
    }
    assert process["cuda_peak_reserved"] == {
        "status": "available",
        "value": 5678,
        "unit": "bytes",
    }
    assert process["host_ru_maxrss"]["status"] in {"available", "unavailable"}


def test_runtime_marks_unreached_and_unsupported_measurements_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry_type = _load_runtime(monkeypatch)["Telemetry"]

    class NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    telemetry = telemetry_type(tmp_path)
    telemetry.attach_torch(types.SimpleNamespace(cuda=NoCuda))
    telemetry.begin_invocation(
        data_root=tmp_path / "test",
        output_dir=tmp_path / "output",
        method="m",
        fold=0,
        test_names=["empty.zarr"],
    )
    telemetry.begin_dataset("empty.zarr", tmp_path / "test/empty.zarr")
    telemetry.add_pair_counts(pair_universe=0, threshold_passing=0)
    telemetry.set_ilp("unavailable", reason="empty_graph")
    telemetry.finish_dataset(detected_nodes=0, pre_ilp_edges=0, output_edges=0)
    telemetry.finish_invocation()
    telemetry.finalize_process()

    records = _records(telemetry.process_dir)
    summary = next(record for record in records if record["record_type"] == "dataset_summary")
    assert summary["counts"]["pair_universe"]["status"] == "available"
    assert summary["counts"]["pair_universe"]["value"] == 0
    assert all(value["status"] == "unavailable" for value in summary["durations"].values())
    assert summary["coordinate_artifact"] == {
        "status": "unavailable",
        "reason": "not_reached",
    }
    process = records[-1]
    assert process["cuda_peak_allocated"] == {
        "status": "unavailable",
        "reason": "cuda_unavailable",
    }
    assert process["cuda_peak_reserved"] == {
        "status": "unavailable",
        "reason": "cuda_unavailable",
    }


def test_generated_call_wrappers_preserve_results_and_record_ilp_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Recorder:
        def __init__(self) -> None:
            self.durations: list[str] = []
            self.ilp: list[tuple[str, dict[str, object]]] = []

        def attach_torch(self, torch: object) -> None:
            self.torch = torch

        def started(self, stage: str | None = None) -> int:
            return 10

        def add_duration(self, stage: str, started: int) -> None:
            assert started == 10
            self.durations.append(stage)

        def set_ilp(self, status: str, **details: object) -> None:
            self.ilp.append((status, details))

    recorder = Recorder()
    runtime_module = types.ModuleType(support_telemetry.RUNTIME_MODULE_NAME)
    runtime_module.telemetry = recorder
    monkeypatch.setitem(sys.modules, support_telemetry.RUNTIME_MODULE_NAME, runtime_module)

    graph_result = object()
    saved_result = object()
    namespace = {
        "torch": object(),
        "build_graph": lambda coords, edges: graph_result,
        "save_graph": lambda graph, path: saved_result,
    }
    exec(compile(support_telemetry._SUPPORT_HELPERS, "<support-wrappers>", "exec"), namespace)  # noqa: S102

    encode_result = object()
    score_result = object()

    class Model:
        def encode(self, value: object) -> object:
            return encode_result

        def predict_edges(self, value: object) -> object:
            return score_result

    class Solver:
        def solve(self, graph: object) -> object:
            return graph_result

    token = object()
    assert namespace["_e0_encode"](Model(), token) is encode_result
    assert namespace["_e0_predict_edges"](Model(), token) is score_result
    assert namespace["_e0_build_graph"]([], []) is graph_result
    assert namespace["_e0_solve"](Solver(), token) is graph_result
    assert namespace["_e0_save_graph"](token, "path") is saved_result
    assert recorder.durations == ["encode_tta", "pair_score", "graph_build", "ilp", "geff"]
    assert recorder.ilp == [("returned", {})]

    class RaisingSolver:
        def solve(self, graph: object) -> object:
            raise LookupError("solver failed")

    with pytest.raises(LookupError, match="solver failed"):
        namespace["_e0_solve"](RaisingSolver(), token)
    assert recorder.ilp[-1] == ("raised", {"exception_type": "LookupError"})


def test_patch_fails_closed_before_anchor_matching() -> None:
    with pytest.raises(ValueError, match="Public-patched support SHA-256 mismatch"):
        support_telemetry.patch_support_source("print('unknown source')\n")


@pytest.mark.skipif(
    not PINNED_RECONSTRUCTION.is_file(),
    reason="ignored independently reconstructed support source is unavailable",
)
def test_exact_pinned_public_reconstruction_is_uniquely_patched_and_compiles() -> None:
    source = PINNED_RECONSTRUCTION.read_text(encoding="utf-8")
    result = support_telemetry.patch_support_source(source)

    assert result.public_patched_input_sha256 == (support_telemetry.PUBLIC_PATCHED_SUPPORT_SHA256)
    assert result.telemetry_output_sha256 == (
        "36ad40a380eaa199e5d54209ea13d8411fb2672ae3e40052fa3b8e43506206ac"
    )
    assert len(result.anchor_counts) == 25
    assert set(dict(result.anchor_counts).values()) == {1}
    assert result.source.count(support_telemetry.SUPPORT_TELEMETRY_MARKER) == 1
    assert ast.parse(result.source)
    assert result.source_chain()["pre_public_support_sha256"] == (
        support_telemetry.PRE_PUBLIC_SUPPORT_SHA256
    )
    assert result.source_chain()["runtime_module_sha256"] == result.runtime_module_sha256
