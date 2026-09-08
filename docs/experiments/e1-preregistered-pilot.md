# E1 preregistered frozen-detector pilot

Status: **fixed diagnostic pilot complete; advancement rejected**. All eight capture workers, 16 payloads, 54 initial comparisons, two single-stage refinements and eight overlay sheets completed. The root independently checked aggregates and inspected every overlay. The selected predicted-count reduction was 14.7087%, below 20%; temporal metrics remain unavailable. See [results and evidence limits](e1-pilot-results.md). The original preregistration below is retained as the pre-execution contract, not a claim that downstream work remains unexecuted.

The current ignored machine-readable [`work/e1-diagnostic-plan.json`](../../work/e1-diagnostic-plan.json) must equal that frozen capture plan in every field except one additive `evaluator_contract`. The evaluator enforces this exact equality at runtime, which prevents downstream analysis changes from silently altering the population, model, cache identities, precision arms, grid, or gates used by R5 capture.

## Evidence scope and fixed population

The pilot reuses the four clips and two frames per clip from `work/research-20260907/audit_heatmaps.py`:

| Dataset | Frames | Frame 0 typed-array SHA-256 | Frame 50 typed-array SHA-256 |
| --- | ---: | --- | --- |
| `44b6_0113de3b` | 0, 50 | `3587f4ae86104ae634f8a47d7a8c7a3f6e43aff331062d89ef3916a4511d622a` | `c1dc58206040c4a717325e841f0c6f98ac680e9877ab3254f8fdae1d8c73c7d0` |
| `44b6_0b24845f` | 0, 50 | `9a46ff48b8afd9ac980095fd81cfe70716be26057aabb07183785c4eb7ea6aae` | `8b5031eacfa943ec40be4df5e8165e8e96c73f73a10e63edcf0b4a3f54196d89` |
| `44b6_0db75fae` | 0, 50 | `aa454aa2dae659eb6c438b6c866099c3405b34107c90b715147a5fe1681c461d` | `b5ef5123481de1f0ecd65232c9b6558ed79e5c2db3345969f24a0c4da59bd24f` |
| `44b6_12dfb391` | 0, 50 | `b7b3ac3ec72f937f0380db312b9457f098c27a23683d25b432f0c63866bd56b7` | `0c367a30f8e5e1ca54d3b6207adc5c6b0e05373f47fa07b58afc198c13962845` |

These hashes were computed on Vast by opening each actual Zarr frame with `require_complete_chunks=True` and hashing its `uint16` dtype, `[64,256,256]` shape, and C-order bytes with the cache implementation's typed-array rule. Each source Zarr has shape `[100,64,256,256]` and scale `[1.625,0.40625,0.40625]` micrometers in `TZYX`/`ZYX` order. The cache worker recomputes the hashes after reading the frames. The controller must compare them with this table and reject the run on any mismatch.

The fold0 model did not fit `44b6`, but all eight frames were inspected in the earlier precision audit and their clips occur in the partial outer evaluation. They are now `diagnostic_reuse_not_heldout`. They cannot support a held-out, clean-generalization, release, or leaderboard-selection claim.

The population contains isolated frames rather than complete temporal clips. Annotated recall, localization, counts, plateau structure, candidate/cap behavior, and stage timing can be measured. Whole-clip estimated-node count ratio, per-clip cap rate, and fixed-linker adjusted edge score cannot be validly computed. Those fields must remain `MISSING_NOT_RUN`, and the machine-readable completeness gate must remain blocked. A promising result can justify a separately preregistered complete-clip evaluation; it cannot advance a model directly.

## Frozen model and production identity

The model is the tensor-only fold0 step-32,000 export at:

```text
/workspace/biohub-cell-tracking/work/kaggle-point-model-20260907/detector-weights.pt
```

Its verified SHA-256 is `2cdce85448f833c80fe0be6d5245ea860124ac508965bb7f4dac1d78c8eb2ad5` and its size is 810,837 bytes. The accompanying `inference-model.json` SHA-256 is `6add0ba0273cc1975221f2828fc1610655c26e06d15f8c8704d59a960d4417ab`. The model is `PointDetector3D(base_channels=12)`. Frozen production preprocessing uses XY stride 4, tile size 64, overlap 16, threshold 0.3, physical NMS radius 3.0 micrometers, and the 2,000-node safeguard.

All eight raw frames normalize to `[64,64,64]`, so each frame contains one 64-cubed model tile. There is no tile-input padding. The stored blend weights still remain part of the cache contract, and replay still executes the production probability blend. In this geometry the weights cancel because there is one tile.

The learned production source has no TTA. `tta-disabled.json` therefore records `enabled=false` and an empty transform list. Enabling TTA would require a different identity and implementation.

## Precision and device fidelity

Two forward caches produce three comparisons:

1. `amp_native_sigmoid` is the frozen control: CUDA-autocast model forward, recorded native float16 logits, native sigmoid on the original CUDA device, and production probability blending.
2. `amp_logits_fp32_sigmoid` reuses exactly the AMP cache, casts each pre-sigmoid tile to float32, and evaluates sigmoid on the original CUDA device. This isolates activation arithmetic.
3. `full_fp32` uses a separately keyed CUDA model forward with autocast disabled and float32 sigmoid on CUDA.

The AMP worker must assert `native_logit_dtype == torch.float16`; the full-float32 worker must assert `torch.float32`. Both must record `inference_device_type == cuda`. Before the control contributes any metric, `reconstruct_probabilities(payload, activation="native")` must be bitwise equal to a contemporaneous call through `pipelines.learned.frame_probabilities` on the same frame and device.

Torch sigmoid kernels can differ across CPU and CUDA. A replay with `device="cpu"` must be labeled `DEVICE_CHANGED_CPU_DIAGNOSTIC`; it cannot establish frozen-control parity or replace any of the three planned CUDA variants. Float32 activation replay also remains on CUDA so that its only intended change is activation dtype.

## Frozen extraction grid

Each precision field is evaluated with both the legacy voxel-maxima extraction and the separate connected-plateau extraction. The plateau variant uses 26-connected equal-score maxima, the voxel nearest the physical centroid with lexicographic `ZYX` tie breaking, and deterministic Euclidean NMS in physical coordinates.

The initial grid is the Cartesian product:

- probability thresholds: `0.3`, `0.5`, `0.7`;
- equivalent logit thresholds: `-0.8472978603872036`, `0.0`, `0.8472978603872034` when operating on logits;
- physical NMS radii: `2.0`, `3.0`, `4.0` micrometers.

This is 9 configurations for each precision/extraction variant, below the limit of 15. Across three precision fields and two extraction rules there are 54 planned initial configurations. The exact frozen control is AMP/native sigmoid, legacy extraction, threshold 0.3, radius 3.0 micrometers, and maximum 2,000 nodes.

Exactly one optional refinement stage is allowed. It triggers only after every initial configuration has complete frame-level metrics. One precision/extraction variant is selected by annotated recall, then lower node count, then higher threshold, smaller radius, and lexical variant ID, provided recall loss is at most one percentage point and cap rate does not rise. The refinement evaluates only midpoints to adjacent initial threshold or radius values around that anchor, with at most six unique configurations. If no initial candidate qualifies, refinement is skipped. There is no second refinement or range extension.

## Advancement and missing-metric rule

A later complete-population candidate must satisfy every condition against the frozen control:

- at least 20% relative reduction in node-count ratio;
- no more than 0.01 absolute annotated-recall loss;
- strictly higher fixed-linker adjusted edge score on identical coverage;
- median localization-distance increase no greater than 0.40625 micrometers and p95 increase no greater than 1.625 micrometers;
- no increase in per-frame or per-clip cap rate.

Any null, missing, nonfinite, partial-coverage, identity-mismatched, or failed-parity metric makes the candidate ineligible. Passing these conditions would authorize a broader evaluation, not establish release quality.

Required reporting covers control parity; annotated recall; matched localization median and p95; predicted and estimated counts and their ratio; raw local-maximum voxels; plateau counts, volumes, and representative displacement; pre-cap candidates and clipped fraction; per-frame and per-clip cap rates; fixed-linker adjusted edge score; stage timings; and overlay review. The plan initializes every field as `MISSING_NOT_RUN` rather than inserting historical values.

## Exact worker configuration

The ignored config files are under `work/e1-diagnostic-configs/`:

- `adapter-native-amp.json` and `precision-native-amp.json` for the shared AMP cache;
- `adapter-full-fp32.json` and `precision-full-fp32.json` for the full-float32 cache;
- `transform.json` and `tta-disabled.json` shared by both.

The full file and canonical-JSON hashes are recorded in `work/e1-diagnostic-plan.json`. Before dispatch, the controller must synchronize the source and ignored configs to Vast and verify the six source hashes and combined source identity from the plan. A mismatch requires regenerating and reviewing the plan rather than silently updating an expected digest.

One exact AMP worker invocation for `44b6_0db75fae` is shown below. The plan contains argument arrays for all eight worker invocations: two precision captures by four datasets. The controller must register and approve each run, create its exclusive attempt directory, and replace the three `CONTROLLER_MUST_BIND_*` values with the admitted run-spec digest, intent, and fencing token. Those values do not exist before admission and are deliberately not fabricated here.

```bash
export BIOHUB_RUN_ID=e1-pilot-native-amp-44b6_0db75fae
export BIOHUB_RUN_SPEC_SHA256=CONTROLLER_MUST_BIND_64_LOWERCASE_HEX
export BIOHUB_INTENT_ID=CONTROLLER_MUST_BIND_APPROVED_INTENT
export BIOHUB_FENCING_TOKEN=CONTROLLER_MUST_BIND_NONNEGATIVE_INTEGER
export BIOHUB_ATTEMPT_DIR=/workspace/biohub-cell-tracking/reports/campaign-workers/$BIOHUB_RUN_ID
export BIOHUB_PROGRESS_PATH=$BIOHUB_ATTEMPT_DIR/progress.json

/workspace/biohub-cell-tracking/.venv/bin/python \
  /workspace/biohub-cell-tracking/scripts/cache_detection_diagnostics.py \
  --input-zarr /workspace/biohub-cell-tracking/data/train/44b6_0db75fae.zarr \
  --dataset-id 44b6_0db75fae \
  --output-dir $BIOHUB_ATTEMPT_DIR/cache \
  --model-file /workspace/biohub-cell-tracking/work/kaggle-point-model-20260907/detector-weights.pt \
  --source-file /workspace/biohub-cell-tracking/scripts/cache_detection_diagnostics.py \
  --source-file /workspace/biohub-cell-tracking/src/biohub_ct/campaign/detection_diagnostics.py \
  --source-file /workspace/biohub-cell-tracking/src/biohub_ct/campaign/learned_logit_adapter.py \
  --source-file /workspace/biohub-cell-tracking/src/biohub_ct/pipelines/learned.py \
  --source-file /workspace/biohub-cell-tracking/src/biohub_ct/training/data.py \
  --source-file /workspace/biohub-cell-tracking/src/biohub_ct/training/model.py \
  --adapter biohub_ct.campaign.learned_logit_adapter:create_adapter \
  --adapter-config-json /workspace/biohub-cell-tracking/work/e1-diagnostic-configs/adapter-native-amp.json \
  --transform-json /workspace/biohub-cell-tracking/work/e1-diagnostic-configs/transform.json \
  --precision-json /workspace/biohub-cell-tracking/work/e1-diagnostic-configs/precision-native-amp.json \
  --tta-json /workspace/biohub-cell-tracking/work/e1-diagnostic-configs/tta-disabled.json \
  --max-frames 2 --frame-index 0 --frame-index 50 \
  --max-wall-seconds 120
```

Each worker must atomically produce two NPZ caches and manifests plus `cache-index.json` under its exclusive `reports/campaign-workers/<run-id>/cache/` directory. The cache reference, progress, and `COMPLETE` `result.json` also live in that attempt directory and are bound to the run, intent, fencing token, and run-spec digest. Every run record in the machine-readable plan enumerates the two payloads, two sidecars, index, reference, and progress in `run_spec_expected_artifacts`; `result.json` is declared separately as the result manifest. The worker result must hash exactly that artifact set so the controller can download and independently verify every cache file. Existing caches are reusable only through strict identity reload.

## Bounded cache replay evaluator

[`scripts/evaluate_detection_diagnostics.py`](../../scripts/evaluate_detection_diagnostics.py) is the separately hash-bound cache consumer. It subsequently ran on the CUDA caches; [the results](e1-pilot-results.md) record its completed identity and scope. The evaluator file is deliberately excluded from the cache-capture source identity, so adding or reviewing analysis code cannot relabel captured logits. Its own SHA-256, coordinate rule, 7 micrometer annotation-matching distance, grid bounds, and runtime gates are frozen under `evaluator_contract` in the machine-readable plan.

The evaluator strict-reloads all 16 payloads and sidecars, independently checks each downloaded worker result and completion receipt against its admitted run digest, intent, fencing token, and exact seven-file artifact manifest, checks both cache indexes, reads and rehashes the eight actual raw frames, and requires CUDA identities with `native_amp`/`torch.float16` for the native arm and `full_float32`/`torch.float32` for the full-forward arm. The ordered GEFF node IDs, float64 `ZYX` coordinates, and annotation counts for every frame are separately frozen and rehashed before any metric is calculated. For every frame the evaluator reconstructs native probabilities and requires bitwise equality with a contemporaneous production `frame_probabilities` call on the same CUDA device before any metric is retained. The frozen threshold-0.3/radius-3.0 legacy node coordinates and cap flag must also equal `probability_nodes` exactly. AMP logits with float32 sigmoid and the full-float32 arm replay on CUDA.

Extraction consumes the blended probability fields, so the executable grid uses the frozen probability thresholds `0.3`, `0.5`, and `0.7`; the corresponding logit values remain reference identities rather than a second threshold grid. Candidate coordinates live on the normalized model grid. Physical NMS therefore uses the raw `ZYX` scale multiplied by `[1, xy_stride, xy_stride]`, then maps retained coordinates back to raw `ZYX` for 7 micrometer annotation matching. This prevents treating one downsampled XY voxel as 0.40625 micrometers.

Immediately after cache loading, before the 54-cell grid consumes outputs, the evaluator runs one frozen high-cap diagnostic on `44b6_0113de3b` frame 0 using the native legacy control at threshold 0.3 and radius 3.0. It compares the normal 2,000-node result with `max_nodes=262144`, equal to all voxels in the 64-cubed probability field, and records counts, cap flags, truncation flags, and coordinate hashes. This diagnostic has no grid, refinement, or advancement selection effect. Its result is checkpointed immediately and carried through every later partial report; a high-cap cap or candidate-pool truncation is reported as incomplete.

The evaluator emits all 54 initial rows with eight per-frame records each. It records annotated recall, matched-distance median and p95, predicted counts, local-maximum voxels, actual connected plateau counts and volume histograms, plateau-representative displacement summaries, pre-cap counts, truncation, clipped fraction, per-frame cap rate, and stage timing. Legacy extraction retains equal-maximum voxels as production does, while its morphology fields are computed from actual 26-connected equal-score components; it does not mislabel every maximum voxel as a one-voxel plateau. A pre-NMS candidate-pool truncation makes clipped fraction explicitly missing instead of understating it. Refinement is allowed only after all 54 initial rows are complete, selects one precision/extraction arm by the frozen ordering, evaluates only adjacent threshold/radius midpoints, and cannot exceed one stage or six configurations.

Each frame row stores the typed-array SHA-256 of its deterministic int64 raw-`ZYX` prediction coordinates rather than repeating large coordinate lists in every checkpoint. Root overlay review can regenerate those coordinates from the hash-bound caches and exact cell configuration, require the regenerated hash to match, and then render overlays. The evaluator itself leaves overlay review missing.

Recall and localization use the repository's explicitly diagnostic matching rule: exact dynamic programming when both node sets contain at most 20 nodes, otherwise greedy matching ordered by physical distance, prediction index, and annotation index. The implementation uses a `cKDTree` radius query to avoid constructing every prediction/annotation pair and has parity coverage against the existing local matcher. This is not the pinned organizer matcher and must not be described as official metric evidence.

The evaluator is an admitted bounded worker. It requires `--max-wall-seconds` plus the controller-injected run, run-spec, intent, fencing, attempt, and progress identities. A POSIX wall alarm and cooperative checks bound cache loading, the fixed high-cap diagnostic, and every grid cell. It atomically checkpoints the high-cap result before grid evaluation, then carries it through each `RUNNING` grid/refinement checkpoint with separate initial/refinement counts; refinement cannot begin before all 54 initial rows exist. The supervisor-facing `completed_units` stays zero throughout the sweep because one operational unit means the complete initial grid plus its optional refinement. It becomes one only after the final report has exactly 54 initial rows and at most six refinement rows. This avoids triggering the supervisor's finalization grace while refinement is still running. The worker result hashes the final evaluation report and progress file, with `result.json` bound separately.

Estimated-node count, node-count ratio, per-clip cap rate, and fixed-linker adjusted edge score remain missing with `isolated_frame_panel_has_no_complete_temporal_clip`. Overlay review remains missing until a person performs it. The evaluator therefore always writes `advancement_eligible=false`; its output can motivate a separately preregistered full-clip evaluation but cannot advance or release a model.

## Geometry-based resource bounds

Each cache has one `[1,64,64,64]` float32 logit array, one same-size float32 weight array, and two `[1,3]` int64 bound arrays: 2,097,200 uncompressed array bytes per frame. Sixteen caches contain 33,555,200 array bytes, approximately 32.001 MiB, before small NPZ/JSON headers. The plan reserves 40 MiB.

The complete cache workload is 16 actual frame reads, 128 MiB of raw input bytes, and 16 one-tile model forwards. Workers stream one frame at a time. There are eight serial worker invocations, each capped at 120 seconds, for a 960-second aggregate hard ceiling if every worker consumes its full allowance. These are geometry counts and deadlines, not a throughput measurement. The Vast run must record observed read, normalization, forward, transfer, serialization, and reload times before any runtime claim.
