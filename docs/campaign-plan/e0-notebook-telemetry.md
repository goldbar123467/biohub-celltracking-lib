# E0 notebook telemetry

Date: 2026-09-08

## Scope

`biohub_ct.campaign.e0_notebook_telemetry` provides the outer, additive telemetry layer for an opt-in E0 notebook package. It does not modify the 12 public cell sources, does not patch the support inference script, does not launch a job, and does not change the frozen R3 package.

The module covers:

- exact public-cell source identity and order at IPython pre/post boundaries;
- elapsed time for the 12 public cells and explicitly allowed integrity/validation cells;
- a narrowly scoped timer around writes to the target `submission.csv`; frozen
  R4 observed `pandas.DataFrame.to_csv`, while the current unrehearsed source
  also observes `csv.DictWriter`;
- low-rate process-tree host RSS and per-GPU used/total memory samples;
- strict retention, detector-coordinate, run-stat, and final-submission harvest;
- a self-contained standard-library runtime that a packager can embed in a prepended cell;
- a finalizer that preserves existing elapsed fields and adds a distinct telemetry record to the existing run manifest.

Support-script stage instrumentation and the narrowly scoped `subprocess.Popen`
interception are separate modules integrated by the
[instrumented package](instrumented-public-reference-package.md). The outer
sampler cannot attribute asynchronous GPU work to model, peak extraction, pair
scoring, ILP, or GEFF serialization.

The later bounded eight-frame real-model differential passed for exact detector
candidates, solved graphs, and logical GEFF content. The full private R4 run
later completed at Kaggle after 5,706.2 seconds with 465 files. Independent
exact-version validation accepted the frozen release structure. The 241,400-row
CSV, all 12 detector-coordinate hashes and counts, and all 12 logical GEFF
graphs matched R3 exactly; the R4 raw coordinate artifacts were independently
rehashed and decoded. The outer telemetry contract still failed because
the frozen notebook used `csv.DictWriter` for `submission.csv`, outside the
installed pandas timer, leaving serialization timing unavailable. Canonical R4
is therefore settled `FAILED`, not telemetry-accepted. A follow-up hook remains
unrehearsed and cannot revise the frozen run. See the
[R4 telemetry differential report](../experiments/e0-r4-telemetry-differential-2026-09-08.md).

## API

Create a validated `NotebookTelemetryConfig`, then call:

```python
from biohub_ct.campaign.e0_notebook_telemetry import (
    NotebookTelemetryConfig,
    build_notebook_telemetry_sources,
)

config = NotebookTelemetryConfig(
    output_dir="/kaggle/working",
    expected_public_cell_sha256=("...",),  # all 12, in source order
    allowed_auxiliary_cell_sha256=(
        "...",  # existing exact input-integrity cell
        "...",  # existing exact CSV-validation cell
    ),
    sample_interval_seconds=5.0,
    nvidia_smi_timeout_seconds=2.0,
)
sources = build_notebook_telemetry_sources(config)
```

`sources.prepended_source` must execute before the existing input-integrity cell. The 12 public cells remain byte-for-byte unchanged. `sources.final_source` must execute after the existing CSV-validation cell. The factory automatically allows the exact final-cell hash and returns hashes for the runtime and both generated sources.

The generated final cell obtains the production input root and ordered dataset
stems from the public notebook's `TEST_DIR` and `test_stems`. For a standalone
harvest, supply these explicitly. The pinned public split uses stems such as
`movie-a`, without the `.zarr` suffix; copying directory names with that suffix
would change the identity contract.

The prepended source:

1. verifies the embedded runtime source SHA-256;
2. refuses to replace an already registered runtime module;
3. registers the runtime in `sys.modules` before `exec`, which is required for dataclasses;
4. registers IPython pre/post callbacks;
5. starts the resource sampler without importing Torch or creating a CUDA context.

The final source unregisters callbacks, restores any pandas wrapper, verifies one successful execution of each public source hash in exact order, runs strict harvest, stops and joins the sampler, writes telemetry, and adds `e0_telemetry` to the run manifest.

`harvest_e0_telemetry(config)` is also public for offline verification of retrieved artifacts. It performs no provider or network operation.

## Cell identity and enforcement boundary

Each pre callback hashes `info.raw_cell` as UTF-8. Each post callback independently hashes `result.info.raw_cell` and requires it to match the pre hash. Public hashes must appear once in the configured order. Explicitly allowed non-public hashes use role `instrumentation`, so input-integrity and independent-validation time remains separate from public algorithm cells.

Any unexpected hash, overlap, unavailable source, or post-boundary mismatch is recorded in memory and immediately appended to `e0_cell_violations.jsonl`. Finalization rejects the evidence even if all 12 public cells later execute successfully.

IPython's event manager may catch a callback exception and continue executing a cell. The callbacks therefore invalidate final evidence; they are not a security boundary that guarantees the unexpected cell never runs. The package must rely on the final rejection and exact artifact identity, rather than interpreting a raised pre callback as execution prevention.

The bootstrap registers during its own execution, so a post event for that bootstrap can arrive without an observed pre event. That one boundary is ignored when no cell is active. The generated final cell is explicitly allowed, unregisters the callbacks before harvest, and is not reported as a completed timed cell.

## Resource sampler

The sampler writes `e0_resource_samples.jsonl` immediately and flushes each record. It records UTC and monotonic timestamps plus:

- aggregate RSS in bytes for the notebook PID and descendants, read from Linux `/proc`;
- process count and root PID;
- each `nvidia-smi` GPU index and UUID;
- GPU used and total bytes, converted from the command's MiB values;
- backend status and reason when `/proc`, `nvidia-smi`, or an individual sample is unavailable.

The record count is bounded by `max_samples`. The first attempt past the cap increments `dropped_samples` and stops the sampler. A cap exhaustion makes resource status `ERROR`, coverage `PARTIAL_CAP_EXHAUSTED`, and persisted host/GPU peaks `LOWER_BOUND`; an unpersisted sample never contributes to a reported peak. Missing backends are `UNAVAILABLE`, never numeric zero. Write errors or failure to join the thread also make resource status `ERROR`. An `atexit` handler stops the sampler after an abnormal notebook exit when Python shutdown still runs. Finalization also stops it on validation failure.

Even with complete sampling coverage, these are observed sample maxima: brief
memory spikes between samples can be missed. They are not exact continuous
whole-process-tree high-water marks. The support processes separately expose
their PyTorch allocator peaks and operating-system `ru_maxrss` measurements.

The host RSS measurement is process-tree RSS summed per sample. Shared memory can therefore be counted in more than one process. GPU memory is device-wide `nvidia-smi` usage, so it can include unrelated processes on a shared device. These are conservative operational measurements, not per-allocation profiler results.

## Required harvest inputs

The output directory must contain:

- `public_reference_run_manifest.json` with actual TZYX shapes, unless shapes were provided in the config;
- `e0_support_telemetry/launch-events.jsonl` from the production launch interceptor;
- one `e0_support_telemetry/invocation-*/events.jsonl` file per support process, containing invocation, retention, coordinate, dataset-summary, and process-summary records;
- `run_stats.csv`;
- `submission.csv` unless the crosscheck is explicitly disabled.

The harvester binds support records to the exact PIDs in `SHARD_LAUNCHED`; it never selects records by filename glob, latest timestamp, diagnostic-arm label, or matching dataset alone. The launch arguments must prove the reviewed entry point, final production data root, fold 0, and one unsliced worker or the complete `0::2`/`1::2` pair. An explicit `--method` must match the invocation identity. When the single-process public command omits it, the reviewed support-source hash binds the parser default and the emitted identity plus output path bind the resolved method. For two workers, each invocation must contain the exact ordered `production_test_names[i::2]` cohort. Selected cohorts must be disjoint and their union must equal the manifest datasets. Unlaunched validation invocations can coexist in the directory but are counted only as ignored evidence.

Each retention dataset must have exactly one record for every expected frame `0..T-1`, including frames with zero primary or blended candidates. The harvester recomputes the retention ratio and `use_primary` decision. Here fallback means selecting primary detector coordinates instead of the blend.

Each coordinate summary must declare:

```json
{
  "dataset": "movie-id",
  "stage": "post_detection_pre_graph_pre_ilp",
  "columns": ["t", "z", "y", "x"],
  "dtype": "<i2",
  "rows": 2,
  "coordinate_sha256": "...",
  "frame_counts": [[0, 1], [1, 1]],
  "artifact": {
    "path": "e0_support_telemetry/invocation-.../coordinates/movie-id.i2",
    "bytes": 16,
    "dtype": "<i2",
    "shape": [2, 4],
    "sha256": "..."
  }
}
```

The artifact path must stay under the output directory and resolve to a regular, non-symlink file. The harvester validates byte count, SHA-256, little-endian int16 shape, dataset bounds, and dense per-frame counts. A summary-only coordinate hash is never treated as independently verified. If `require_coordinate_artifacts=False`, it is retained as `UNAVAILABLE` with a reason.

`run_stats.csv` must contain one row per expected dataset, core node/edge fields, and every distinct required fallback/cap field. The output keeps their definitions and values separate. The harvester records these transitions without assuming equality:

- detector coordinates before graph construction and ILP;
- `raw_nodes` and `raw_edges` from the saved post-ILP GEFF before notebook repair;
- final `nodes` and `edges` after repair.

It reports signed deltas between those stages. It does not reject a legitimate ILP node-count change. Final `nodes` and `edges` must match row-type counts independently read from `submission.csv`.

## Runtime scopes

The final telemetry document keeps these scopes distinct:

- `runtime_scopes.wrapper.elapsed_seconds` starts in the prepended cell after kernel startup and ends in the final cell before telemetry and manifest writes.
- `runtime_scopes.cells.records[].elapsed_seconds` measures an exact public or allowed instrumentation cell between callbacks.
- `runtime_scopes.provider` is `UNAVAILABLE` inside the notebook. Provider-complete elapsed must be added by an external exact-version retrieval step after the provider reports completion.
- `resources.scope` is `whole_notebook_wrapper`; these sampled peaks cover the notebook process tree and device-wide GPU use over the wrapper interval.
- `harvest.support.selected_invocations[].resource_scope` is `production_shard_process`; each record contains that support process's CUDA allocated/reserved peak and normalized `ru_maxrss` evidence.

The finalizer does not overwrite legacy `elapsed_seconds` or `full_runtime_seconds` fields in `public_reference_run_manifest.json`. It adds a separate `e0_telemetry` object with the telemetry path/hash, wrapper elapsed, and an explicit unavailable provider value.

## Support telemetry integration contract

The support telemetry patcher emits one raw coordinate artifact per dataset and one matching JSONL event using the schema above. Coordinates use original-voxel `(t,z,y,x)` order and canonical contiguous `<i2` bytes. The support patcher owns exact inner-stage timers, pair counts, ILP returned/raised outcome, GEFF serialization evidence, and per-process CUDA/host peak records. The outer harvester requires every selected dataset summary and coordinate event to agree field-for-field, every selected invocation to return exactly once, and every selected process summary to name exactly that invocation.

Because source cell 4 applies its dynamic support patch and launches subprocesses
in the same cell, an extra notebook cell cannot run between those actions. The
instrumented packager installs a narrowly scoped launch interceptor before cell
4. It acts only on the reviewed prediction entry point and repository path,
completes the exact hash-chain/anchor patch immediately before process creation,
preserves arguments and return values, and restores the original callable. This
module supplies the outer telemetry runtime and deliberately does not install
that interceptor by itself.
