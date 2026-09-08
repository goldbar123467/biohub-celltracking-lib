"""Compare extracted, exact upstream CPU kernels with their instrumented versions.

This checks the actual reviewed source functions, not an independently rewritten
detector. It does not claim model, CUDA, graph, or complete notebook parity.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from biohub_ct.campaign.e0_support_telemetry import (
    patch_support_source,
    runtime_helper_source,
)

SOURCE = Path("work/e0-reference/public-patched-support-root.py")
pytestmark = pytest.mark.skipif(not SOURCE.is_file(), reason="Pinned source unavailable")


def _kernels(source: str, names: set[str], torch: object, telemetry=None) -> dict:
    tree = ast.parse(source)
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + functions,
        type_ignores=[],
    )
    namespace = {"np": np, "torch": torch, "F": torch.nn.functional, "_e0_telemetry": telemetry}
    exec(compile(ast.fix_missing_locations(module), "<exact-public-kernels>", "exec"), namespace)  # noqa: S102
    return namespace


def test_actual_cpu_frame_and_detector_kernels_preserve_values(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("BIOHUB_E0_TELEMETRY_DIR", raising=False)
    runtime = {"__name__": "parity_runtime"}
    exec(compile(runtime_helper_source(), "<telemetry-runtime>", "exec"), runtime)  # noqa: S102
    telemetry = runtime["Telemetry"](tmp_path / "telemetry")
    # No Torch attachment: these are CPU tests and must not initialize CUDA.
    telemetry.begin_invocation(
        data_root=tmp_path / "test",
        output_dir=tmp_path / "out",
        method="cpu-kernel-parity",
        fold=0,
        test_names=["sample"],
    )
    telemetry.begin_dataset("sample", tmp_path / "test/sample.zarr")
    source = SOURCE.read_text(encoding="utf-8")
    instrumented_source = patch_support_source(source).source
    original = _kernels(source, {"_load_frame", "_detect_cells_pooled"}, torch)
    patched = _kernels(
        instrumented_source,
        {
            "_load_frame",
            "_detect_cells_pooled",
            "_e0_uninstrumented_load_frame",
            "_e0_uninstrumented_detect_cells_pooled",
        },
        torch,
        telemetry,
    )
    generator = np.random.default_rng(628)
    video = generator.normal(size=(2, 7, 11, 13)).astype(np.float32)
    for target, stride in [([7, 11, 13], (1, 1, 1)), ([5, 4, 6], (2, 3, 2))]:
        expected = original["_load_frame"](video, 1, target, stride)
        actual = patched["_load_frame"](video, 1, target, stride)
        assert torch.equal(expected, actual)
        assert actual.dtype == expected.dtype == torch.float32
    cases = [
        np.full((1, 5, 7, 9), -20, dtype=np.float32),
        np.zeros((1, 5, 7, 9), dtype=np.float32),
        generator.normal(size=(1, 5, 7, 9)).astype(np.float32),
    ]
    cases[1][0, 1:3, 2:4, 3:5] = 8  # Equal-score plateau.
    for values in cases:
        tensor = torch.from_numpy(values)
        for threshold in (0.5, 0.965):
            for kernel in ((1, 1, 1), (3, 3, 3), (1, 3, 5)):
                expected = original["_detect_cells_pooled"](tensor, 3, threshold, kernel)
                actual = patched["_detect_cells_pooled"](tensor, 3, threshold, kernel)
                np.testing.assert_array_equal(actual, expected)
                assert actual.dtype == expected.dtype
    telemetry.finish_dataset(detected_nodes=len(actual), pre_ilp_edges=0, output_edges=0)
    telemetry.finish_invocation()
    telemetry.finalize_process()
