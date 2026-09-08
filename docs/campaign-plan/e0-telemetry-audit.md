# E0 telemetry audit

Date: 2026-09-08

## Decision

The frozen E0 R3 package does not provide enough telemetry to support an operational claim about hidden-test runtime or memory headroom. It is suitable as a byte-identified reproduction package, but it records only a notebook-internal elapsed value, a total production-prediction duration, output structure, and selected aggregate counts. It does not measure the required setup, data-read, model, candidate-extraction, pair-scoring, solver, serialization, and validation stages separately, and it records no peak host RAM or GPU memory.

No new execution is implied by this audit. The R3 package remains frozen and unlaunched.

## Audited identities and citation convention

The audit used these exact local artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `work/e0-reference/source-v1/biohub-942tta.ipynb` | `521cb97f0f457643379a51b60c4f71e3f4cc7d1823fd98cbb97633ffaa515ec4` |
| `work/e0-reference/package-r3/submission.ipynb` | `b0c51948112fd407a848de44d52923f2fe70f432e5520060338f9f518afe5e30` |
| `work/e0-reference/package-r3/package-manifest.json` | `26065311f67657122f6436111b5fc1d15f3451a1bc20e1da1d6c27a3808c20c4` |
| `scripts/package_public_reference.py` | `edf831522d65da64eb5158e5c19365b08ba2316cdfb170b0e702cae1eda38fbb` |
| `reports/e0-reference/upstream-v1/biohub-942tta.log` | `930ca025396989f87bcb867eac763680da70c0b505e3bc38608811fdec4d73eb` |
| `reports/e0-reference/upstream-v1/run_stats.csv` | `ee570469cc37973dd40b9b645d259918f74738f5689d74d7f11c71b436c42c4d` |
| `reports/e0-reference/upstream-v1/dual_seed_frame_retention_guard_report.json` | `02ae26ed1a000862fde8f5960355f21cbf1c0f077048b7cf30010edfbf2acc3a` |
| `reports/e0-reference/upstream-v1/bidirectional_production_runtime_integrity.json` | `ae41130ee035d3ddcbaf2d0977f721429a52ffda8133fe1b877f51de5926278d` |

Notebook references below use zero-based source-cell indexes and one-based line numbers within the joined `cell.source`. That convention makes the references reproducible without creating a derived Python file. The relevant public cells have these hashes, also recorded in `work/e0-reference/package-r3/package-manifest.json:51-63` and `:85-97`:

- source cell 4: `a3a9827b4c0119e90111a091083658317e6b2542320d03ef69ffd031c0a64684`
- source cell 5: `ab3be142f357c1eb8e25c40537af64d1b95e4d60b8735afdd533781f6dfb7f9c`
- source cell 6: `239f51c30606f090e1fbc939a541891cef2dbea45d7b1c4a6fef53352e94759d`

The R3 manifest reports 12 source cells and 14 packaged cells, the original 12 hashes in the same order, and only two added instrumentation cells: `e0-input-integrity` and `e0-release-validation` (`package-manifest.json:46-71`, `:83-108`). The builder performs exactly that prepend/append operation at `scripts/package_public_reference.py:547-563`, records the public cell span and hashes at `:590-605`, and records `algorithm_changes: []` at `:626`.

## What R3 measures

### Runtime

R3's appended wrapper starts `_E0_STARTED_MONOTONIC` in its prepended cell (`scripts/package_public_reference.py:264-273`). Its final cell stores `elapsed_seconds` and copies it to `full_runtime_seconds` at `:485-494`.

That value is a **wrapper elapsed time**, not a provider-complete elapsed time. The clock starts only after the Kaggle kernel has begun executing notebook code. The value is captured before the CUDA environment probe and before the run manifest is serialized (`scripts/package_public_reference.py:495-509`). It therefore excludes at least provider queue/startup time and the final CUDA-probe/manifest-write/notebook-conversion tail. The field name `full_runtime_seconds` overstates its scope.

The source manifest separately records the provider-displayed runtime `1h 21m 41s` (`work/e0-reference/package-r3/package-manifest.json:99-107`). That is a **provider-complete display value for the upstream source run**, not a measurement from the unlaunched R3 package. The two values must remain separate in any future schema:

- `wrapper_elapsed_seconds`: first prepended wrapper statement through the final wrapper's capture point.
- `cell_elapsed_seconds`: individual notebook-cell callback duration.
- `provider_elapsed_seconds`: provider-supplied completed-run duration, harvested after completion; unavailable from inside the notebook.

Source cell 4 measures only the total two-shard production prediction interval (cell 4, lines 395-441). The upstream log reports `Prediction completed in 9.65 minutes` at line 1106. The repeated `predict_minutes_total=9.651475127538045` in every row of `run_stats.csv:2-5` is one shared total copied into four dataset rows, not four per-dataset measurements.

### Candidate and graph counts

Source cell 4 writes per-frame retention records containing primary candidate count, blended candidate count, retention ratio, and `use_primary` (cell 4, lines 84-164). It later writes a selected-coordinate manifest by dataset with total row count, observed frame counts, and a hash at the `post_detection_pre_graph_pre_ilp` stage (cell 4, lines 192-209). The current frame-count mapping omits zero-count frames, so it cannot prove that every expected frame was considered.

Source cell 5 writes `run_stats.csv` after graph generation and post-processing (cell 5, lines 1559-1684). The archived rows sum to:

| Counter | Observed total | Meaning |
| --- | ---: | --- |
| `raw_nodes` | 125,328 | Nodes read from the saved production GEFFs before notebook post-processing |
| `raw_edges` | 118,423 | Edges read from those saved GEFFs after the support script's ILP |
| `nodes` | 122,841 | Final post-processed nodes |
| `edges` | 118,559 | Final post-processed edges |
| `gap_candidates` / `gap_pairs_selected` | 841 / 616 | First gap-closing proposal and selection counts |
| `gap2_candidates` / `gap2_pairs_selected` | 372 / 143 | Second gap-closing proposal and selection counts |
| `safe_division_geometric_candidates` / `safe_division_candidates` / `safe_divisions_added` | 730 / 119 / 102 | Successive safe-division filters and accepted additions |

`raw_edges` is not the number of pair candidates before thresholding or before ILP. In the archived support member `repo/scripts/predict_unet_transformer.py`, threshold-passing candidates are created at lines 460-468, further degree-filtered at lines 470-488, then passed through graph construction and ILP at lines 554-563 before the GEFF is saved at line 564. No pre-threshold pair-universe count, threshold-passing count, or ILP input/output count is emitted.

### Fallbacks and caps

The retention report records 65 fallback frames among 400 rows (`dual_seed_frame_retention_guard_report.json:18-46`). In this context, `use_primary=true` selects the primary detector coordinates instead of the blended coordinates for a frame. It does not mean that the notebook fabricated output after a failure.

The current pipeline has no detector-count cap. The archived `_detect_cells_pooled` emits every local maximum above the detection threshold and returns an empty array when none exist (`repo/scripts/predict_unet_transformer.py` inside `work/e0-reference/downloads/support/biohub-tracking-support-pack-50ep-v1.zip`, lines 254-293; archived member SHA-256 `c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9`).

The graph-repair cap counters are incomplete observability, not proof that no cap branch was encountered:

- In gap-2 repair, the global cap counter increments once and breaks, while a per-frame cap branch continues silently (source cell 5, lines 846-914).
- In safe-division repair, the global cap counter increments once and breaks, while a per-frame cap branch breaks silently (source cell 5, lines 998-1122).
- The four `run_stats.csv` rows contain zero for the recorded `gap_skipped_node_cap`, `gap2_skipped_cap`, `safe_division_skipped_cap`, `motion_relink_fallback_raw`, and `motion_relink_skipped_large_frame` fields. Those zeros apply only to the fields' implemented semantics.

Other fields describe separate recovery or skip paths. They must stay distinct: retention primary-selection fallback, motion-relink raw-edge fallback, motion-relink large-frame skip, DeepCenter checkpoint missing, short-track all-filter skip/rescue, and artifact-resolution fallback are not interchangeable. Source cell 5 fails closed if the final graph set is empty (lines 1605-1607). The submission runbook also requires fail-closed handling rather than arbitrary fallback nodes (`docs/campaign-plan/SUBMISSION_RUNBOOK.md:19`).

### Integrity and validation

Source cell 6 validates retention records and summarizes them (cell 6, lines 34-63 and 105-159), but it checks movie coverage rather than requiring one record for every expected frame. The final wrapper validates the CSV, input coverage, coordinates, graph relations, and hashes, but it does not reconcile selected detector coordinates with pre-postprocess or final graph counts.

The runtime-integrity artifact hashes the archived support script as `c44e...234b9` (`bidirectional_production_runtime_integrity.json:16-31`) and explicitly says `verified_before_dynamic_source_patch: true`. Source cell 4 changes that materialized script at runtime. No artifact records the final dynamically patched script hash. That omission prevents exact source attribution for the code that actually ran inside the production shards.

## What can be reconstructed from the upstream log

The upstream log contains timestamped events. Subtracting event timestamps gives useful retrospective intervals:

| Interval | Seconds | Evidence |
| --- | ---: | --- |
| shard launch to production prediction complete | 579.087 | log lines 143 and 1106 |
| production prediction complete to DeepCenter post-process start | 205.532 | lines 1106 and 1107 |
| DeepCenter start to first submission CSV write | 262.192 | lines 1107 and 1135 |
| validator shard launch to merged validation graphs | 987.836 | lines 1249 and 3137 |
| validation merge to base proxy report | 293.273 | lines 3137 and 3222 |
| post-process sweep start to final candidate result | 2,031.731 | lines 3225 and 3569 |
| selected-config rewrite to final CSV | 256.413 | lines 3587 and 3620 |
| final CSV to final logged notebook conversion | 10.623 | lines 3620 and 3641 |

These are **retrospective estimates of phase spans**. The timestamps themselves were emitted by the run, but the phase labels are inferred from sparse boundary messages. A span can include unlogged work, synchronization, scheduling, serialization, or idle time. It is invalid to treat this table as direct component profiling or to sum overlapping spans as independent costs.

The final log timestamp is 4,900.833 seconds (`biohub-942tta.log:3641`), close to the provider display of 4,901 seconds. This corroborates the upstream wall duration. It does not validate R3 timing because R3 was never launched.

Against an assumed 12-hour limit, 4,901 seconds would leave 38,299 seconds, or 10:38:19, and 88.7% arithmetic headroom. A 25% reserve on 4,901 seconds is 1,225.25 seconds, producing a 6,126.25-second planning budget. These are **retrospective arithmetic estimates against an unrefreshed assumption**, not measured current limits or hidden-test headroom. The runbook requires sufficient wall and memory headroom (`SUBMISSION_RUNBOOK.md:23-28`), while `GPU_OPERATIONS.md:71-75` requires stage timing, peak RAM/VRAM, actual frame/tile and candidate-edge counts, measured tail behavior, and an explicitly estimated reserve. R3 cannot meet that contract.

## Minimum additive instrumentation while preserving all 12 public cell sources

An opt-in package can preserve the exact bytes and hashes of all 12 public cells by adding wrapper cells and a standalone telemetry module. The existing R3 default path should remain byte-identical. The instrumented package should use a new release identity and record every added cell and runtime source mutation.

### Outer notebook instrumentation

A prepended wrapper should:

1. Store the ordered expected public-cell SHA-256 list and register IPython `pre_run_cell` and `post_run_cell` callbacks.
2. Hash the exact cell source at both boundaries, record start/end monotonic time, outcome, and any exception class, and fail closed on an unexpected source hash or order. Telemetry cells need a separate allowlist so they cannot be mistaken for public cells.
3. Start one low-rate sampler thread. At each sample, enumerate the notebook process and descendants, record aggregate and maximum RSS in bytes from `/proc`, and query each GPU's used and total memory in bytes with a bounded `nvidia-smi` subprocess when available. Record backend name, sample interval, timestamps, missing samples, and explicit `unavailable` reasons. The sampler must not import Torch or create a CUDA context.
4. Register `atexit` cleanup and also stop/join the sampler explicitly in the final wrapper. A finalizer must report whether cleanup succeeded.

The notebook process must write append-only JSONL through atomic or flushed writes so a partial run retains evidence. All durations use monotonic seconds. Wall timestamps use UTC. Host and GPU memory use integer bytes. Coordinates remain original voxels ordered `(z, y, x)`; physical scale is `z=1.625 um`, `y=0.40625 um`, `x=0.40625 um` as recorded in the packaged artifact lock.

This outer layer provides exact public-cell attribution, cell-level time, wrapper time, process-tree RSS, and whole-run per-GPU memory. It cannot by itself attribute GPU peaks to inner stages because the production work runs in child shard processes and GPU work is asynchronous.

### Support-script instrumentation at the shard boundary

Source cell 4 materializes and dynamically patches `repo/scripts/predict_unet_transformer.py`, then launches two independent CUDA subprocesses later in the same cell (cell 4, lines 395-438). There is no notebook-cell insertion point between those actions. To preserve the source bytes of cell 4, the prepended wrapper must install a narrowly scoped `subprocess.Popen` interceptor before public execution. When, and only when, a command launches the reviewed prediction entry point from the reviewed repository path, the interceptor patches the already materialized support file immediately before process creation. It must restore the original callable after the expected launch set or during final cleanup. The patch step must:

1. Require the expected archived source hash and record the hash after the existing public dynamic patch.
2. Require each instrumentation anchor to match exactly once, compile the patched source, atomically replace the file, and record a three-part hash chain: archived member, public-dynamic patch result, telemetry patch result.
3. Make the child process import a standalone telemetry helper materialized by the wrapper, with no project-relative imports.
4. Emit per-dataset/per-shard stage records for dataset/Zarr open, frame read, base model encode, TTA encode, peak extraction, pair-universe construction, edge-model scoring, threshold passing, graph construction, ILP solve, and GEFF serialization.

The audited archived anchors are:

| Stage | Function and lines in archived support member |
| --- | --- |
| peak extraction | `_detect_cells_pooled`, lines 254-293 |
| dataset/Zarr open | `predict_video`, lines 319-326 |
| frame load/normalization/transfer | `predict_video`, lines 363-370 |
| base encode | `predict_video`, line 372 |
| TTA encode/fusion | `predict_video`, lines 375-388 |
| detection | `predict_video`, lines 392-401 |
| pair universe | `predict_video`, lines 407-425; count `n_src * n_tgt` before gating |
| pair scoring | `predict_video`, lines 441-458 |
| threshold passing and degree filter | `predict_video`, lines 460-488 |
| model load | `predict`, lines 537-543 |
| graph construction | `predict`, line 554 |
| ILP | `predict`, lines 555-563 |
| GEFF serialization | `predict`, line 564 |

GPU timers must synchronize the shard's assigned CUDA device at finite stage boundaries. The telemetry overhead and synchronization policy must be recorded. The current ILP API only supports observing whether `solver.solve(graph)` returned or raised, plus elapsed time and graph counts around the call. Solver status, objective, optimality gap, iteration count, and solver memory are unavailable unless exposed by the solver API and explicitly captured. They must not be inferred from a returned graph.

### Strict final harvest

The final wrapper should reject an incomplete telemetry artifact. For every discovered dataset it must:

- derive expected `T` from the actual input shape;
- require exactly one retention record for every frame `0..T-1`, including zero-primary and zero-blended frames;
- require explicit candidate counts and `use_primary` for every frame, with `retention` either finite under a defined denominator rule or an explicit unavailable reason;
- load the chosen detector-coordinate artifact, require finite in-bounds `(t,z,y,x)` values, recompute a canonical SHA-256, and produce a dense frame-count vector containing zeros;
- reconcile the dense coordinate total to the pre-graph manifest and `run_stats.raw_nodes` per dataset;
- load final GEFF or validated submission counts and reconcile `run_stats.nodes` and `run_stats.edges` per dataset;
- preserve each fallback and cap counter as a separate named field with a definition, and reject missing required counters instead of filling them with zero;
- report unobservable values as unavailable with a reason.

The expected reconciliation direction matters. Selected detector coordinates are `post_detection_pre_graph_pre_ilp`; `run_stats.raw_nodes/raw_edges` are read from post-ILP GEFFs before notebook graph repair; `run_stats.nodes/edges` are after repair. Equality should only be required where the code contract says node identity/count is preserved. Any intended transform needs an explicit delta record.

Provider elapsed time must be merged only after the provider reports a completed run. It is not available to the notebook finalizer. A local final manifest should therefore leave `provider_elapsed_seconds` unavailable, and an external harvest step should add the provider value with its source and retrieval time without overwriting `wrapper_elapsed_seconds`.

## Differential verification before operational use

Instrumentation changes timing and can expose latent nondeterminism. Before using an instrumented run as operational evidence, run an uninstrumented control and an instrumented candidate on the same visible inputs, frozen datasets, weights, environment, settings, and seeds. Then require:

1. all 12 public cell source hashes to equal the R3 list in order;
2. the archived/public-dynamic/telemetry support-source hash chain to be complete;
3. canonical selected-coordinate arrays and per-frame dense counts to match between runs;
4. pre-ILP threshold-passing edge tuples to match when captured in both runs;
5. final GEFF topology and canonical `submission.csv` SHA-256 to match;
6. every telemetry reconciliation and output validator to pass;
7. sampler coverage and cleanup to pass, with missing GPU/RSS backends reported rather than treated as zero;
8. instrumentation overhead to be reported as an estimate from repeated paired runs, not subtracted from a single run.

Only then can the instrumented run supply visible-workload operational evidence. Do not claim measured hidden-test headroom from a visible run. When hidden inputs are unavailable before submission, admission requires refreshed provider limits, a defensible conservative workload envelope with explicit assumptions and uncertainty, protected recovery reserve, and actual runtime enforcement. Hidden-input unavailability alone is not a categorical reason to reject an otherwise defensible bounded run.
