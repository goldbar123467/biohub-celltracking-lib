# E0 support-predictor telemetry

`biohub_ct.campaign.e0_support_telemetry` instruments the pinned public support
predictor without changing the twelve upstream notebook cell strings. It is an
opt-in release-evidence path. The default R3 package remains byte-identical.

## Source boundary and integration

The patch accepts only the exact support source after the audited public notebook
transformations. Its pinned provenance is:

```text
archived support member:
  c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9
after the 11 public source replacements:
  49613ad0b50ac90c3e07e3f8a803f2adf0926203be3f4a97d577c60b8f756178
```

The second value was independently reconstructed from the exact notebook string
literals: every old anchor matched once, the result was 48,404 UTF-8 bytes and
1,110 lines, and Python compilation succeeded. `patch_support_source` checks that
hash before looking for telemetry anchors, requires all 24 telemetry anchors to
match exactly once, and compiles the result. It returns `PatchedSupportSource`,
whose `source_chain()` includes the archived, public-patched, telemetry-output,
and runtime-helper SHA-256 values.

The opt-in notebook wrapper should do the following at the first recognized
`scripts/predict_unet_transformer.py` shard launch, after the public patches have
run:

1. Read the support script and call `patch_support_source` with the pinned default
   expected hash. Do not replace that expected value with the hash just read.
2. Materialize `runtime_helper_source()` as
   `scripts/_biohub_e0_support_telemetry_runtime.py`, beside the target script.
   This lets the script import it through Python's normal script-directory search
   without changing the upstream `PYTHONPATH`.
3. Atomically write the returned support source, and persist `source_chain()` in
   the instrumented run evidence.
4. Add only `BIOHUB_E0_TELEMETRY_DIR` to each shard environment. Give the run a
   new empty telemetry root. The runtime creates an exclusive
   `invocation-<pid>-<time_ns>` directory, so a later validation invocation cannot
   overwrite a production coordinate artifact. Coordinate paths also include the
   logical invocation ID, covering multiple folds executed by one process.
5. Keep later shard launches on the already instrumented file. The
   `BIOHUB_E0_SUPPORT_TELEMETRY_V1` marker and first-launch interception prevent a
   double patch; unknown or partially patched bytes must fail closed.

Every event carries the PID, process ID, GPU shard and existing
`BIOHUB_DIAGNOSTIC_ARM`. Invocation records additionally bind the resolved input
data root, resolved prediction output directory, method, fold and ordered test
cohort. Dataset records bind the resolved dataset path. This identity is needed
because the production and later validation processes can inherit the same arm
and shard strings. Final release aggregation must select the actual competition
test-root cohort from these path and cohort fields.

## Measurements

The JSONL records use status-tagged values. A measured count of zero is
`status=available, value=0`; an unsupported or unreached measurement is
`status=unavailable` with a reason. The helper never substitutes a zero for
missing evidence.

Stage durations have these exact boundaries:

- `data_read`: the dataset/Zarr open and quantile reads, plus every call to the
  original `_load_frame` implementation;
- `encode_tta`: the sum of all primary and optional secondary `model.encode`
  calls, including every enabled spatial TTA view;
- `detector_extraction`: every `_detect_cells_pooled` call, including the public
  retention-guard probes and the selected detector extraction;
- `pair_score`: the forward, optional reverse, and optional secondary
  `predict_edges` calls;
- `threshold`: edge activation, strict `probability > cfg.threshold` filtering,
  tuple construction and descending sort;
- `graph_build`: the existing `build_graph` call;
- `ilp`: the existing `solver.solve` call only;
- `geff`: the existing `save_graph` call.

CUDA is synchronized immediately before and after the four GPU stages, so their
wall durations do not merely measure asynchronous dispatch. The other stage
boundaries are CPU calls. Stage values are cumulative per dataset and include a
nanosecond integer plus seconds. They are not estimates of code outside the
listed boundaries.

For every scored consecutive frame pair, `pair_universe` adds exactly
`n_source * n_target`. A skipped empty pair contributes zero. The
`threshold_passing_edge_candidates` count adds `len(candidates)` immediately
after the strict threshold and before greedy parent/child filtering. These are
the required pre-graph counts, distinct from selected pre-ILP and output edge
counts.

Every public retention-guard decision is also copied into the process event
stream with its invocation ID and resolved dataset path. This lets the outer
finalizer select the returned production invocation and require exact frame
coverage without mixing later validation calls that inherit the same diagnostic
arm and shard labels. The legacy public retention JSONL write remains unchanged.

ILP status has only three values: `returned`, `raised`, or `unavailable`.
Unavailable includes a reason such as `disabled`, `empty_graph`, or
`not_reached`; raised includes the exception class and re-raises the original
exception. No solver-internal status is claimed because the public API does not
return one.

Immediately before `predict_video` returns, the patch serializes the exact
post-detection coordinates as contiguous little-endian int16 C-order bytes. It
adds the returned artifact object to the existing public
`detector_coordinates_*.jsonl` record, which keeps one coordinate summary per
dataset for the outer harvester. The artifact record includes its safe relative
path, SHA-256, byte
count, shape, dtype, columns, sparse raw-derived frame counts, coordinate space
(`original_voxel_zyx`) and stage (`post_detection_pre_graph_pre_ilp`). The
harvester-facing `artifact.path` is a safe relative path below `/kaggle/working`;
the event's `coordinate_sha256` and `artifact.sha256` both hash the raw bytes.
Writes use a flushed, fsynced partial file followed by `os.replace`; a duplicate
dataset artifact in the same logical invocation fails closed before either file
can be replaced.

At process exit, the helper reports CUDA peak allocated and reserved memory in
bytes after resetting PyTorch peak counters before model loading. It reports
`resource.getrusage(RUSAGE_SELF).ru_maxrss` with its native platform unit and a
normalized byte value. Missing CUDA, a failed peak reset, an unavailable
`resource` module, or a query failure is explicit.

## Verification and remaining evidence

The unit tests execute the standalone runtime with deterministic clocks and fake
CUDA counters, verify raw coordinate bytes and hashes, distinguish real zero
counts from unavailable stages, exercise returned and raised ILP calls, and
check that timing wrappers preserve object identity. When the ignored independent
public-source reconstruction is present, an additional test patches its exact
bytes, checks all anchors, and parses the final AST.

This establishes source identity, compilation and minimal executed telemetry
behavior. It does not establish identical real-model tensors, coordinates,
graphs, GEFF bytes, GPU runtime or memory use. The instrumented package still
needs a bounded differential rehearsal against the uninstrumented pinned path
before release admission.
