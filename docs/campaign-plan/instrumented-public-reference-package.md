# Instrumented E0 public-reference package

Date: 2026-09-08

Current execution: the title-fixed R4 package was launched once as private,
offline `clarkkitchen/biohub-e0-instrumented-reference/1` on September 8 at
08:51 UTC. Its exact push receipt is CONFIRMED; execution and downloaded
telemetry validation remain pending. The 6,480-second timeout reserves 1.98
quota hours while protecting 26.40 hours. Admission followed the completed R3
reproduction, whose 241,400-row CSV is byte-identical to the upstream artifact.
No model-quality admission follows from either launch or this reproduction.

`scripts/verify_e0_telemetry_outputs.py` independently checks downloaded R4
cell order and source hashes, callback cleanup, outer manifest binding,
resource samples and support-stage records. It resolves recorded Kaggle paths
inside the downloaded tree without rewriting the original evidence. Probe
availability is reported explicitly. Its 34-test telemetry/integration suite
passed, including an actual-harvester fixture; the integrated native Windows
suite passed 392 tests with 62 optional-dependency/platform skips. These local
checks do not substitute for the pending full Kaggle artifact check.

## Boundary

`scripts/package_instrumented_public_reference.py` builds an opt-in telemetry
candidate around the frozen E0 public reference. It calls the existing
`package_public_reference.build_package` in a temporary directory and copies the
resulting 12 public cells without changing their source strings. The existing R3
builder and package are not modified. The candidate is not release-admitted and
the builder does not contact or execute Kaggle.

The final notebook cell order is fixed:

1. outer telemetry bootstrap;
2. support-launch interceptor installation;
3. input-integrity validation generated from the candidate lock;
4. the 12 exact public cells;
5. base submission and graph validation generated from the candidate lock;
6. support-launch restoration and cleanup assertion;
7. outer telemetry finalizer.

The callback target is zero-based public cell 4, SHA-256
`a3a9827b4c0119e90111a091083658317e6b2542320d03ef69ffd031c0a64684`.
It restores `subprocess.Popen` after that cell, expects two observed launches
when `worker_count >= 2 and not SLICE`, and one otherwise. It persists `PENDING`,
`PASS`, or `ERROR` evidence at
`/kaggle/working/e0_support_telemetry/launcher-callback-status.json`. The later
close cell rejects a missing or failed callback even when IPython swallowed the
callback exception, and always attempts callback removal, atexit-hook removal,
environment restoration, and stream closure.

## Lock and failure behavior

The candidate artifact lock extends the base lock with the exact semantic
instrumentation configuration and SHA-256 hashes for:

- the candidate builder;
- notebook telemetry runtime;
- support telemetry patcher;
- embedded support runtime helper;
- launch interceptor.

Generated auxiliary-cell hashes are recorded in the package manifest, outside
the lock, which avoids a recursive dependency between the lock, validation
cells, and telemetry bootstrap. The release digest is recomputed over the
candidate lock without its `release_digest` field. The notebook's existing
`metadata.biohub_e0.release_digest`, candidate metadata, artifact lock, and
package manifest all use that new digest. `base_release_digest` remains explicit
for comparison.

The output directory is exclusive. Construction first creates `INCOMPLETE` and
removes it only after all four final files have been written. Any exception
leaves the sentinel in place, so partial output cannot be mistaken for a ready
candidate.

## Build and verification

The CLI matches the base package inputs:

```bash
python scripts/package_instrumented_public_reference.py \
  --source-notebook work/e0-reference/source-v1/biohub-942tta.ipynb \
  --archive pilkwang/biohub-tracking-support-pack-50ep-v1=/path/to/support.zip \
  --archive pilkwang/biohub-temporal-unet3d-seed314159-v1=/path/to/temporal.zip \
  --archive pilkwang/biohub-deepcenter-unet3d-center-prior-v1=/path/to/center.zip \
  --output-dir /new/exclusive/output-directory \
  --kernel-id owner/e0-instrumented-reference
```

`tests/test_package_instrumented_public_reference.py` rebuilds a fake pinned
candidate twice and compares every output byte, checks the exact public cell
strings and requested order, recomputes the release digest, verifies the base
builder remains byte-identical before and after candidate construction, checks
exclusive-directory and retained-sentinel failures, exercises the direct CLI,
and runs generated callback/close glue through an IPython-like dispatcher that
swallows callback errors. The existing launcher tests separately execute real
subprocess interception. A real Kaggle candidate still needs exact-version
retrieval and differential release review before admission.

## Root verification against the actual pinned inputs

This section records the earlier candidate build. The current title-fixed source
generation is recorded below; neither build has completed a full Kaggle rehearsal.

The root built the candidate twice from the three downloaded pinned archives
and original V1 notebook. All four output files matched between builds. The 12
public cell strings matched the original, all 18 packaged cells compiled, the
lock preimage reproduced its release digest, and preflight passed when supplied
the independently constructed candidate identity. The frozen R3 identity
correctly rejected the candidate. A separate R3 rebuild reproduced all four
frozen R3 file hashes exactly.

The local build receipts are in
`work/e0-reference/telemetry-root-build-verification/candidate-receipt-final.json`.
The current candidate directories are `candidate-final-a` and `candidate-final-b`.
They supersede the earlier build pair after correcting the single-process
launch's omitted `--method` handling. The candidate release
digest is `d9cb48f9efd00c4f6faaa6a72b8e84e17c2d326c6f15ad0778b03ed007696706`;
the notebook SHA-256 is
`d3d07f9d3aee8a579a2fb82fb6062747e1f1cf41c5fc902b12a549d23a89a711`.
These receipts prove build identity only. The candidate identity is not added
to the operator's compiled release allowlist, and no inference or launch was
performed by these build checks.

`tests/test_e0_instrumented_kernel_parity.py` separately executes the actual
upstream and instrumented frame-loading and detector functions with CPU Torch.
Two frame-loading cases and 18 detector cases preserve exact tensor/array
values, including interpolation, empty detections and equal-score plateaus.
This does not establish real-model, CUDA, graph, or complete notebook parity.

The subsequent [real-model support differential](../experiments/e0-model-graph-parity-2026-09-08.md)
ran both checkpoint models on the same eight full-spatial visible frames on the
existing RTX 4070 SUPER. Exact candidates, solved graphs and logical GEFF output
matched between control and instrumented arms. The root independently decoded
the downloaded GEFF arrays and checked them against the solved graphs. The
supervised run took 22.6556 seconds, and four focused comparator tests passed.
This closes the bounded support-path CUDA/graph comparison only. Full notebook,
DeepCenter and later postprocessing parity, complete visible coverage and Kaggle
offline execution remain open; the candidate is still outside the operator's
release allowlist.

## Current source-reviewed R4

The September 8 live R3 launch showed that Kaggle derives the effective notebook
slug from its title. The instrumented builder now derives a deterministic ASCII
title from `--kernel-id`, or validates an explicit `--title` against that exact
slug. Unsupported punctuation, Unicode and mismatched titles fail before output
creation. The base R3 builder and package remain unchanged.

Fresh builds `work/e0-reference/package-r4-title-fixed-a` and `-b` are byte-identical.
The root and an independent reviewer checked the four-file inventory, all 18
compiled cells, all 12 unchanged public source strings, release-digest preimage
and manifest bindings. Installed Kaggle CLI 2.2.4 independently maps the title
`Biohub E0 Instrumented Reference` to the requested slug
`clarkkitchen/biohub-e0-instrumented-reference`.

| Identity | SHA-256 |
|---|---|
| Release digest | `41b2810b88edd3b61a4bed8b3f32d1a0df3454af463f20520473d96bb6244353` |
| Artifact lock | `68312573350cc079561167e25304dd1a5b053a10abc3c7c8ea3869c64fbeb6ff` |
| Kernel metadata | `df623fb2924738f7d21612408ff0590c2d4ab021deca9d1fa615b36c19e61c95` |
| Package manifest | `62bcfb18db088e217f9aeb417f9329b464d8f86ae316fff1680fdc4e666ab049` |
| Notebook | `e7f4fbdf7c94bd475eb00ee00dfff9cfdfa3022aa35ebb959421cd50c3b41fb8` |
| CLI-normalized notebook | `630e6cca6e663888573f19f4efd97721177483fb59495d2cb249365f8d9e90fa` |

`rehearsal_packages.py` now compiles this identity for explicit `--generation r4`
in the launch/preflight and output-validation operators. R3 remains the default.
This selects reviewed source for a bounded rehearsal; it grants no quota,
launch, model-quality or submission permission. Arbitrary identity JSON and hash
overrides remain unavailable. The real R4 CLI preflight returned
`PREFLIGHT_PASS_NO_LAUNCH`; its receipt SHA-256 is
`50f4588d709e785fc98b1e1fd09e9476f9f1fc7c9f44cac29fb681d3584f81c4`.

Root verification passed 14 instrumented-builder tests and 25 operator tests.
An independent combined builder/base-builder review passed 23 tests with one
existing platform skip. Full R4 execution, final-output differential comparison,
telemetry verification and resource admission remain pending the active R3
rehearsal and a fresh quota observation.
