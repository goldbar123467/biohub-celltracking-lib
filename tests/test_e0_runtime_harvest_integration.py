from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from biohub_ct.campaign.e0_notebook_telemetry import (
    _COUNTER_SEMANTICS,
    NotebookTelemetryConfig,
    harvest_e0_telemetry,
)
from biohub_ct.campaign.e0_support_telemetry import (
    PRE_PUBLIC_SUPPORT_SHA256,
    PUBLIC_PATCHED_SUPPORT_SHA256,
    RUNTIME_MODULE_NAME,
    runtime_helper_source,
)

DATASET = "movie-a"
SHAPE_TZYX = (2, 2, 3, 4)


def _write_child_runtime(scripts: Path) -> str:
    runtime_source = runtime_helper_source()
    (scripts / f"{RUNTIME_MODULE_NAME}.py").write_text(runtime_source, encoding="utf-8")
    (scripts / "predict_unet_transformer.py").write_text(
        f"""from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path

from {RUNTIME_MODULE_NAME} import telemetry


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--method", default="unet_transformer")
args = parser.parse_args()

dataset = {DATASET!r}
data_root = Path(args.data_dir).resolve()
output_dir = Path(os.environ["E0_TEST_PREDICTION_ROOT"]) / args.method / "split_0"
output_dir.mkdir(parents=True, exist_ok=True)
validation = os.environ.get("E0_TEST_VALIDATION") == "1"
rows = [(0, 1, 2, 3)] if validation else [(0, 0, 1, 2), (1, 1, 2, 3)]
payload = b"".join(struct.pack("<hhhh", *row) for row in rows)

telemetry.begin_invocation(
    data_root=data_root,
    output_dir=output_dir,
    method=args.method,
    fold=int(args.split),
    test_names=[dataset],
)
telemetry.begin_dataset(dataset, data_root / dataset)
for frame in range({SHAPE_TZYX[0]}):
    telemetry.record_retention({{
        "dataset": dataset,
        "frame": frame,
        "primary_candidates": 1,
        "blended_candidates": 1,
        "retention": 1.0,
        "minimum_retention": 0.9,
        "use_primary": False,
    }})
telemetry.add_pair_counts(pair_universe=1, threshold_passing=0)
telemetry.set_ilp("unavailable", reason="disabled")
telemetry.capture_coordinates(
    dataset=dataset,
    payload=payload,
    shape=(len(rows), 4),
    dtype="<i2",
)
telemetry.finish_dataset(
    detected_nodes=len(rows),
    pre_ilp_edges=0,
    output_edges=0,
)
telemetry.finish_invocation()
telemetry.finalize_process()
""",
        encoding="utf-8",
    )
    return runtime_source


def _run_child(
    workspace: Path,
    *,
    data_root: Path,
    method: str,
    validation: bool,
) -> subprocess.Popen[str]:
    child_python = getattr(sys, "_base_executable", sys.executable)
    argv = [
        child_python,
        "scripts/predict_unet_transformer.py",
        "--data-dir",
        str(data_root),
        "--split",
        "0",
    ]
    if method != "unet_transformer":
        argv.extend(["--method", method])
    environment = {
        **os.environ,
        "BIOHUB_E0_TELEMETRY_DIR": str(workspace / "e0_support_telemetry"),
        "BIOHUB_GPU_SHARD": "unavailable",
        "BIOHUB_DIAGNOSTIC_ARM": "production",
        "E0_TEST_PREDICTION_ROOT": str(workspace / "predictions"),
        "E0_TEST_VALIDATION": "1" if validation else "0",
    }
    process = subprocess.Popen(
        argv,
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        raise
    assert process.returncode == 0, (stdout, stderr)
    return process


def _write_outer_evidence(output: Path) -> None:
    (output / "public_reference_run_manifest.json").write_text(
        json.dumps({"actual_input_shapes_tzyx": {DATASET: list(SHAPE_TZYX)}}) + "\n",
        encoding="utf-8",
    )
    stats_fields = [
        "dataset",
        "raw_nodes",
        "raw_edges",
        "nodes",
        "edges",
        *_COUNTER_SEMANTICS,
    ]
    with (output / "run_stats.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=stats_fields)
        writer.writeheader()
        writer.writerow(
            {
                **{field: 0 for field in stats_fields if field != "dataset"},
                "dataset": DATASET,
                "raw_nodes": 2,
                "nodes": 2,
            }
        )
    with (output / "submission.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "row_type"])
        writer.writeheader()
        writer.writerows(
            [
                {"dataset": DATASET, "row_type": "node"},
                {"dataset": DATASET, "row_type": "node"},
            ]
        )


def test_real_support_runtime_is_strictly_joined_to_production_harvest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    scripts = output / "scripts"
    production_root = tmp_path / "production-test"
    validation_root = tmp_path / "validation-train"
    scripts.mkdir(parents=True)
    production_root.mkdir()
    validation_root.mkdir()
    runtime_source = _write_child_runtime(scripts)

    production = _run_child(
        output,
        data_root=production_root,
        method="unet_transformer",
        validation=False,
    )
    validation = _run_child(
        output,
        data_root=validation_root,
        method="unet_transformer_val",
        validation=True,
    )
    assert production.pid != validation.pid

    support_root = output / "e0_support_telemetry"
    output_source_hash = hashlib.sha256(b"instrumented support fixture").hexdigest()
    source_chain = {
        "schema_version": 1,
        "kind": "BIOHUB_E0_SUPPORT_TELEMETRY_SOURCE_CHAIN",
        "pre_public_support_sha256": PRE_PUBLIC_SUPPORT_SHA256,
        "public_patched_support_sha256": PUBLIC_PATCHED_SUPPORT_SHA256,
        "telemetry_output_support_sha256": output_source_hash,
        "runtime_module_name": RUNTIME_MODULE_NAME,
        "runtime_module_sha256": hashlib.sha256(runtime_source.encode("utf-8")).hexdigest(),
        "anchor_counts": {},
    }
    launch_events = [
        {"event": "SUPPORT_PATCHED", "source_chain": source_chain},
        {
            "event": "SHARD_LAUNCHED",
            "pid": production.pid,
            "argv": list(production.args),
            "source_sha256": output_source_hash,
        },
        {
            "event": "LAUNCH_OBSERVATION_FINISHED",
            "launch_count": 1,
            "expected_launch_count": 1,
            "status": "PASS",
        },
    ]
    (support_root / "launch-events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in launch_events),
        encoding="utf-8",
    )
    _write_outer_evidence(output)

    result = harvest_e0_telemetry(
        NotebookTelemetryConfig(
            output_dir=str(output.resolve()),
            expected_public_cell_sha256=("0" * 64,),
            expected_shapes_tzyx={DATASET: SHAPE_TZYX},
            production_data_root=str(production_root.resolve()),
            production_test_names=(DATASET,),
        )
    )

    assert result["support"]["ignored_nonproduction_invocation_count"] == 1
    selected = result["support"]["selected_invocations"]
    assert len(selected) == 1
    assert selected[0]["process_id"].startswith(f"{production.pid}-")
    assert selected[0]["resources"]["cuda_peak_allocated"] == {
        "status": "unavailable",
        "reason": "torch_not_attached",
    }
    assert selected[0]["resources"]["cuda_peak_reserved"] == {
        "status": "unavailable",
        "reason": "torch_not_attached",
    }

    coordinates = result["coordinates"]["datasets"][DATASET]
    artifact = coordinates["artifact"]
    expected_bytes = b"".join(
        int(value).to_bytes(2, "little", signed=True)
        for row in ((0, 0, 1, 2), (1, 1, 2, 3))
        for value in row
    )
    assert coordinates["dense_frame_counts"] == [1, 1]
    assert coordinates["coordinate_sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert artifact["status"] == "VERIFIED"
    assert artifact["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert Path(artifact["path"]).read_bytes() == expected_bytes
    assert result["retention"]["datasets"][DATASET]["fallback_frames"] == 0
