# Testing and offline release architecture

Decision date: 2026-09-05. Research: `docs/research/alpha-2026-09-05.md`.

## Decision

Build a single frame-streaming inference path used by validation and Kaggle.
Start with the existing classical detector and adjacent-frame greedy linker as a
measurable fallback. Keep detection, association and division selection separate.
The next learned candidate is a small 3D point detector, followed by a local
temporal association model. FOCUS-3D teacher labels and HOCT/Trackastra are
experiments to evaluate later, not dependencies of the first submission.

```text
immutable Zarr inputs + frozen config + source digest
  -> one frame at a time -> detections in original voxel coordinates
  -> adjacent-frame candidate links -> constrained graph
  -> per-dataset graph artifact + timing/count diagnostics
  -> atomic, streaming CSV -> graph/schema/bounds validation
  -> CSV round-trip -> pinned official metric (validation only)
```

No data-service or distributed training layer is needed for two compute platforms.
Windows owns source and experiment records; Vast provides debugging and official
scoring; Kaggle runs the same packaged source with internet disabled. Source Git
does not replace checkpoint/artifact backup.

## Contracts

- Images are uint16 `(T,Z,Y,X)`, loaded one frame at a time. Physical scale is
  read from metadata. Predictions retain original integer voxel coordinates;
  distances and gates use micrometers. Missing dependencies, corrupt arrays and
  missing competition frame chunks are errors, not empty detections.
- Metadata-only fixtures are an explicit smoke-test mode. They cannot establish
  prediction quality. A genuinely empty detection result may use the documented
  fallback node and must be visible in diagnostics.
- A prediction graph has unique node IDs per dataset, valid endpoints, increasing
  consecutive times, at most one parent and at most two children. Time-striding
  is not a runtime recovery strategy. Future gap recovery must reconstruct and
  validate intermediate detections before emitting consecutive edges.
- The CSV has exact columns, consecutive global row IDs, contiguous dataset
  groups, exact test-ID coverage, valid sentinels and in-bounds coordinates.
  Write to a temporary file and replace the final CSV only after validation.
- Validation uses whole-embryo folds in both directions. Missing requested IDs,
  train/validation overlap, duplicate IDs and empty validation sets are errors.
  Read `estimated_number_of_nodes` from GEFF metadata; never substitute the sparse
  annotation count. Report unadjusted/adjusted edge Jaccard, node recall, predicted
  and estimated counts, division TP/FP/FN and per-dataset timing.
- Official evaluation calls the pinned organizer implementation and aggregation.
  Local synthetic probes remain useful unit tests, not an interchangeable scorer.
  Save source/config/split/input-metadata hashes and exact environment versions.

## Packaging and acceptance gates

Build a deterministic source archive embedded in a generated Kaggle notebook.
Verify its hash before extraction and import; install only hash-verified attached
wheels with `--no-index`. Discover the actual test IDs at run time. The release
manifest records source digest, configuration and input metadata identity.
Large arrays and model weights are separate versioned artifacts when required.

Required gates: dependency-light tests; real Zarr/GEFF and corruption tests;
official metric tests including unequal sample weights and sparse node counts;
CSV graph/bounds/round-trip tests; deterministic package rebuild; interrupted
inference restart tests; and an actual internet-off Kaggle batch run.

The first cloud rehearsal uses CPU for the classical baseline. Measure whole-clip
runtime, frame throughput, peak memory and output counts before projecting the
roughly 199-clip hidden workload. Reserve at least 25% of the 12-hour limit for
startup, variability and serialization. A projection is not a hidden-test timing
guarantee. Fail explicitly on budget exhaustion; retain completed per-clip work.

Do not claim an architecture is competitively best before bidirectional
embryo-held-out scores support it. Compare detection first, then linking, then
divisions. Promote a candidate only with component metrics, bounded runtime and
known training provenance. A public checkpoint of unknown training membership
cannot supply honest held-out evidence.
