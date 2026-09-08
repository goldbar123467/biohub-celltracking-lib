# E1 frozen-detector diagnostic tooling

Status: **tooling implemented; all eight admitted GPU cache captures complete and all 16 downloaded payloads strictly validated; replay evaluator not executed and no detector-quality result**.

## Implemented contracts

[`detection_diagnostics.py`](../../src/biohub_ct/campaign/detection_diagnostics.py) provides an immutable frame-logit identity, strict cache I/O, deterministic plateau extraction, anisotropic physical NMS, and bounded ablation-plan construction.

Each cache identity binds these seven independent components:

- model file SHA-256
- source-manifest SHA-256
- effective model/configuration SHA-256
- actual input-frame dtype, shape, and byte SHA-256
- coordinate/preprocessing/tiling transform SHA-256
- precision-policy SHA-256
- TTA-policy SHA-256

The generic cache payload is an uncompressed NumPy array declared as `raw_pre_sigmoid_logits`, `ZYX`, and exactly `float32`. Saving rejects implicit float16/float64 conversion, empty or non-3D output, and NaN/infinity. Values are preserved outside `[0, 1]`; no probability substitution or clipping occurs in the cache layer. The data and manifest are each flushed and atomically replaced, the manifest binds both NPY-file and typed-array hashes, and every save is immediately reloaded through the strict reader. A crash between the two atomic replacements yields an identity or checksum failure rather than a falsely valid cache.

`sigmoid_scores` exposes low-precision versus float32 activation as an explicit CPU diagnostic. An AMP model pass cast to float32 and a full-float32 model pass require different precision identities and therefore different caches. Casting values after a low-precision sigmoid is not represented as full-float32 inference.

## Plateau and NMS behavior

The connected-plateau variant first finds 3-by-3-by-3 local maxima with negative-infinity exterior padding, so valid border peaks remain eligible. It labels exact equal-valued maxima with 26-neighbor connectivity. Each component emits one representative nearest its physical centroid; an exact tie uses lexicographic `ZYX` order. Disconnected equal maxima remain separate.

The extraction record preserves:

- the number of raw maximum voxels before plateau collapse;
- plateau count and each plateau's voxel count;
- plateau centroid and representative displacement in micrometers;
- retained count before any maximum-node safeguard;
- whether and how many candidates were capped.

Physical NMS sorts by descending logit and lexicographic coordinates for score ties. It computes Euclidean distances after multiplying coordinates by the explicit `ZYX` micrometer scale. A point at exactly the radius is suppressed. The maximum-node cap runs after NMS and is always disclosed; it is not a detector-tuning parameter.

`FrozenAblationPlan` keeps the precision variant separate from `ExtractionConfig`, which contains only tie handling, logit threshold, and physical radius. The initial Cartesian grid must contain 1 to 15 unique configurations per precision/tie variant. The plan can contain one separately named refinement list, also bounded to at most 15 configurations. The tool does not choose thresholds or radii from evaluation results.

## Bounded streaming worker

[`cache_detection_diagnostics.py`](../../scripts/cache_detection_diagnostics.py) reads one Zarr frame at a time with complete-chunk checks and never copies a complete 4D movie. It requires an explicit `MODULE:FACTORY` adapter. The factory receives its frozen JSON configuration and returns an object with:

```python
name: str

def raw_logits(frame_zyx, *, deadline_at, progress) -> numpy.ndarray:
    ...  # actual finite pre-sigmoid float32 ZYX values
```

The generic interface remains available for models that genuinely emit one frame-aligned logit volume. The worker also accepts a structured adapter with `capture_payload`, `save_cache`, and `load_cache` methods plus explicit cache suffix and output schema. This keeps model-specific tiling and blending provenance out of the generic three-dimensional array contract. Every adapter must call the supplied progress callback at natural tile/window boundaries and honor `deadline_at`. The worker also emits a heartbeat every 30 seconds while the adapter runs.

## Concrete frozen learned-detector adapter

[`learned_logit_adapter.py`](../../src/biohub_ct/campaign/learned_logit_adapter.py) mirrors `pipelines.learned.frame_probabilities` through the model forward. It calls the same `normalize_frame`, generates tile starts in the same `Z`, `Y`, `X` nesting order, constructs the same separable edge-ramp weights, and restores the model's original training mode. Each cache stores:

- one float32 representation of each raw pre-sigmoid tile, preserving values outside `[0, 1]`;
- the native Torch output dtype and the separately named `native_amp` or `full_float32` model-pass precision;
- every tile origin and exclusive stop in normalized-frame `ZYX` coordinates;
- the complete per-tile blend-weight arrays rather than an inferred overlap rule;
- the `Z`-outer/`Y`-middle/`X`-inner iteration order, configured overlap/stride halo, and explicit zero tile-input padding;
- raw and normalized shapes, XY edge padding, stride/block-mean rule, percentile values, clip rule, and normalized-frame typed-array hash.

The learned payload schema is `biohub.learned.per_tile_logits.v1`, distinct from the generic frame-logit schema. The NPZ and exact array hashes are bound to the same model/source/config/frame/transform/precision/TTA identity used by the worker and strictly reloaded after saving.

`reconstruct_probabilities` replays sigmoid on each tile, transfers that probability to float32 CPU form, and accumulates probability times weight in production tile order. It therefore reproduces the frozen control rather than averaging logits and applying one sigmoid. The alternative `float32` activation and optional finite logit clamp are explicit replay parameters. For a native AMP replay, activation runs on CUDA and casts the cached float32 representation back to the recorded float16 or bfloat16 dtype before sigmoid.

The CLI factory is:

```text
--adapter biohub_ct.campaign.learned_logit_adapter:create_adapter
```

Its JSON config accepts `model_format` (`training_checkpoint` or `state_dict`), `model_config` for an exported state dict, `inference_config` for `LearnedConfig`, `device`, and `inference_precision`. The required `--model-file` is passed to this factory and is still independently SHA-256 bound by the worker.

## Historical eight-frame evidence audit

`work/research-20260907/audit_heatmaps.py` compared AMP logits with native sigmoid, the same AMP logits cast to float32 before sigmoid, and a full-float32 forward on eight frames. Its report showed 1,340 / 1,241 / 1,204 retained nodes on `44b6_0db75fae` frame 0 and 1,284 / 1,180 / 1,128 on frame 50; other frames still hit the 2,000-node cap. This is evidence that activation/model precision affects local-max plateaus but does not by itself remove excess detections.

That script forwarded the entire normalized frame once per precision variant. It did not execute the production tiled path or its probability-space overlap blend. Its node counts therefore remain historical diagnostic evidence, not a reproduced frozen-control comparison. The new adapter is the required path for that comparison.

The CLI requires `--max-frames` and `--max-wall-seconds`; it checks both around every frame and installs a POSIX wall-clock alarm around the complete loop. `--run-id` and `--run-spec-sha256` can come from `BIOHUB_RUN_ID` and `BIOHUB_RUN_SPEC_SHA256`. It also requires `BIOHUB_INTENT_ID`, integer `BIOHUB_FENCING_TOKEN`, absolute `BIOHUB_ATTEMPT_DIR`, and absolute `BIOHUB_PROGRESS_PATH` inside that exclusive attempt directory.

Before and after every frame, progress is atomically written as:

```json
{
  "run_id": "...",
  "run_spec_sha256": "...",
  "completed_units": 0,
  "observed_at": "UTC timestamp",
  "error": null
}
```

On completion, `cache-reference.json` and `result.json` are written inside `reports/campaign-workers/<run_id>`, and the cache root is required to be that attempt's `cache/` subtree. The result binds run, intent, fencing token, completed units, `COMPLETE` status, and the exact project-relative SHA-256 map for both NPZ payloads, both sidecars, `cache-index.json`, `cache-reference.json`, and progress. The controller can therefore download and verify the complete cache without following an external path. An existing cache is reused only after strict identity and corruption checks; an incompatible or partial cache fails rather than being silently overwritten.

## Verification

The focused suite covers:

- exact preservation of negative and greater-than-one raw logits through cache save/reload;
- float16 rejection, NaN rejection, and no implicit dimensional conversion;
- cache-byte corruption and precision-identity invalidation;
- distinct float16 and float32 sigmoid behavior;
- one representative for a connected same-logit plateau;
- separate representatives for disconnected equal maxima, including borders;
- anisotropic Z-versus-XY suppression and deterministic score ties;
- cap disclosure after NMS;
- the 15-configuration initial-grid limit and one refinement field;
- the exact atomic progress schema.
- exact equality between cached learned-tile reconstruction and the actual production `frame_probabilities` forward/blend on an odd-shaped frame with eight overlapping tiles;
- production tile order, exclusive bounds, complete positive weights, model-mode restoration, and per-tile progress;
- learned NPZ strict round trip, cache-identity rejection, explicit float32/clamped replay, and deadline interruption before forward.

Current focused result: `61 passed, 1 skipped` across the evaluator, diagnostic, plateau, and dispatch suites with the local SciPy-capable neural environment; Ruff: `All checks passed` for the evaluator and its tests.

## Executed pilot and remaining evidence

The [preregistered pilot](e1-preregistered-pilot.md) has completed CUDA capture, production-control replay, all 54 initial configurations and two single-stage refinements. All 16 downloaded payloads passed independent strict validation; all 56 aggregate rows were independently recomputed. The root inspected all eight actual fixed overlay sheets and preserved hashes in the [visual review](e1-preregistration/visual-review.json).

The [results](e1-pilot-results.md) show a 14.7087% predicted-count reduction, below 20%, with all 26 sparse annotations matched. The frame panel is diagnostic reuse, not held-out evidence. Estimated-node count ratio, per-clip caps, adjusted temporal edge score and human expert annotation review remain unavailable. No advancement or training is admitted by this pilot.

The public reference remains a separate model family with temporal windows, runtime source patches, dual seeds, TTA and association-specific transforms. Its separate compatibility profile does not establish complete Kaggle execution or clean validation provenance.
