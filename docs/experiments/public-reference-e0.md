# E0 public-reference reproduction package

Status: **ready for root review; not pushed, executed, or submitted**.

## Frozen reference

E0 reproduces [`redoctopusk/biohub-942tta`](https://www.kaggle.com/code/redoctopusk/biohub-942tta), the strongest public notebook found in the 2026-09-07 authenticated inventory. Kaggle displayed public score `0.946`, runtime `1h 21m 41s`, two Tesla T4 accelerators, internet disabled, and image digest `37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`.

The frozen source identity is:

- notebook version: `1`; script version ID: `347821442`
- source SHA-256: `521cb97f0f457643379a51b60c4f71e3f4cc7d1823fd98cbb97633ffaa515ec4`
- 12 public algorithm cells; each cell-source hash is recorded in `package-manifest.json`

Kaggle's version-qualified source endpoint returned HTTP 403. The identity was therefore bound by three agreeing observations: the UI showed `Version 1 of 1` and `Best Score 0.946 V1`, the API reported `currentVersionNumber=1` and script version ID `347821442`, and an authenticated current-source pull produced the source hash above. A future version would invalidate this pin rather than silently update it.

The displayed score and runtime remain upstream Kaggle evidence. They are not results from our package.

## Dataset and checkpoint pins

| Kaggle dataset | Dataset ID | Version | Archive SHA-256 | Critical checkpoint SHA-256 |
| --- | ---: | ---: | --- | --- |
| `pilkwang/biohub-tracking-support-pack-50ep-v1` | 10999845 | 10 | `2ea8f16d8e2df6781f3d48713004b18075ae3e462266773670de05a56f908c8a` | `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771` |
| `pilkwang/biohub-temporal-unet3d-seed314159-v1` | 11184174 | 2 | `b1fe1b4636fb7510d247c95a59ed16c523346388659e11285ffe8eea343c98bf` | `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f` |
| `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` | 11061989 | 5 | `c0fddb8ae879e6e6742fa0a6921e226265bb2f5306858c952ded3314cc2bdaf1` | `8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0` |

All three dataset versions report `CC0-1.0`. Their last updates, respectively, were `2026-07-08T22:42:12.497Z`, `2026-07-20T08:26:25.767Z`, and `2026-07-07T11:53:58.670Z`, before the upstream notebook run. The lock records every member's path, byte count, and SHA-256, not only the three checkpoints. It also requires both dependency-bearing archives to contain at least 62 offline wheels.

The public split manifests include `44b6_*` and `6bba_*` competition-training embryos in model fitting. This is eligible public external reuse under the reviewed competition materials, but it is not independent local validation. E0 is therefore classified `overlap_unknown`; no clean-holdout or OOF claim is attached to the `0.946` score.

## Effective upstream method

The public source runs two temporal 3D U-Net/node-transformer models, learned adjacent-frame association, ILP graph selection, bidirectional association fusion, DeepCenter-based gap/division vetoes, and post-link graph repair. The material settings that determine the final output include:

- detection threshold `0.965`; ILP edge/appearance/disappearance/division weights `-1.0 / 0.0 / 2.0 / 1.2`
- detection pooling kernel size `3.0 um`, from the support `PredictConfig` default; the public launch does not override it. The checkpoint configuration's `5.0 um` value recorded by the feature-only profiler is not the runtime extraction setting.
- secondary detection weight `0.80`, secondary edge-logit weight `0.15`, `low_margin_consensus`, low-margin maximum `0.35`, and candidate threshold `0.48`
- bidirectional edge weight `0.15` with `harmonic_probability` fusion
- eight-view planar detection TTA and edge-feature TTA enabled
- gap close maximum gap `2`, distance `5.0 um`, gap-2 recovery enabled, and DeepCenter gap threshold `0.25`
- safe-division parent/sister limits `9.0 / 14.0 um`, DeepCenter threshold `0.25`, global/frame caps `0.00375 / 0.0076`
- minimum track length `6`, plus bounded short-track rescue
- the embedded proxy sweep selected `tight55`, changing final motion-relink tight distance from `6.0` to `5.5 um`, then rewrote `submission.csv`

The downloaded `dual_seed_frame_retention_guard_report.json` describes an earlier state (`0.96875` detector threshold, `0.475` secondary detection weight, `0.30` bidirectional weight, and other differences). It is preserved as provenance but must not be treated as the final effective configuration. The executed source and final log are authoritative for the final CSV.

The same distinction applies to cell 2: its final print says reverse-time weight `0.200`, but the preceding guard and the executed fusion patch both require `0.15`. The verified effective value is `0.15`; the `0.200` line is stale display text.

Cells 8-11 call their held-out post-process sweep an official-metric validator. The node-count adjustment and run-level aggregation formulas follow the organizer formula, but the division matcher is a handwritten heuristic based on whole weak components rather than the pinned organizer implementation. Its `tight55` choice is upstream proxy evidence only. It is not a pinned-official-score or clean-holdout result.

## Packaging and runtime evidence

[`scripts/package_public_reference.py`](../../scripts/package_public_reference.py) verifies the source, exact dataset reference set, outer ZIP digests, all ZIP members, pinned critical members, artifact manifests, and offline-wheel coverage before writing anything. The generated notebook then:

1. hashes every mounted dataset file and requires the mounted path set, size, and SHA-256 to equal the artifact lock before public code executes;
2. runs the 12 public cell sources unchanged, with stale cell outputs and execution counts cleared;
3. discovers actual hidden test Zarr shapes in `TZYX` order and strictly validates the final CSV header and row widths, sequential IDs, integer encoding, sentinels, nonnegative node and edge IDs, coordinate bounds, contiguous dataset groups, dataset coverage, adjacent-frame edges, dangling edges, duplicate nodes, indegree, and outdegree;
4. writes `/kaggle/working/public_reference_run_manifest.json` with discovered shapes, completed datasets, CSV hash and row counts, effective `BIOHUB_*` values, installed distribution versions, CUDA devices, runtime, and validation status.

The raw notebook manifest is not by itself the controller's final rehearsal receipt. After an exact-version Kaggle run is downloaded, the controller must bind the actual notebook slug/version and the campaign's five-field frozen identity, independently hash and validate the downloaded CSV, and produce the receipt accepted by `verify_release_artifacts`. The notebook does not invent those post-run fields.

The package is private, GPU-enabled, and configured with internet disabled. Its three `dataset_sources` use the installed Kaggle CLI's version-qualified `owner/slug/version` form, so the push request names the reviewed versions rather than resolving whichever versions are current at launch. Kaggle CLI 2.2.4's installed input validator accepted all three exact strings. No Kaggle push or run occurred during this task.

## Local evidence

The upstream selected output downloaded from V1 has:

- CSV SHA-256 `a852d1d07ff8c9307d9b10db7f9b4b12e8b1882f14c5dbeb1316d099f0795b3e`
- 241,400 rows: 122,841 nodes and 118,559 edges
- exact datasets `44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, and `6bba_05db0fb1`
- a passing independent `biohub_ct.submission.validator` run

The earlier retention-guard report records 241,282 rows and SHA-256 `0319ba6d8e864335d3573f6b1a6227c546f17e9247a0c2858fa09b6c2422db3f`. That receipt predates the selected `tight55` rewrite and does not bind the final output.

Focused tests cover deterministic builds, source-cell preservation, instrumentation compilation, private/offline metadata, and fail-closed rejection of malformed pins, reference-set mismatches, archive mismatches, required-member mismatches, negative node or edge IDs, noncontiguous dataset groups, and surplus CSV fields. Result: `9 passed`; Ruff check and format: passed.

The original `work/e0-reference/package` and earlier `package-r2` remain preserved. The version-pinned revision is built separately in `work/e0-reference/package-r3`; all 12 public algorithm cell sources remain byte-identical to the pinned source. The notebook and artifact lock are byte-identical to R2 because the algorithm, instrumentation, and mounted-file lock did not change. Only the launch metadata and its package manifest changed.

Generated `package-r3` hashes:

| File | SHA-256 |
| --- | --- |
| `artifact-lock.json` | `6f12505158a1c42a5d15ef336e86a4c4b9e99d9087cfed407da84d1e9a1e7241` |
| `kernel-metadata.json` | `85ef798c0f980027b334f0613818e0ea62e2eb65c848ab27abfe76142249b812` |
| `package-manifest.json` | `26065311f67657122f6436111b5fc1d15f3451a1bc20e1da1d6c27a3808c20c4` |
| `submission.ipynb` | `b0c51948112fd407a848de44d52923f2fe70f432e5520060338f9f518afe5e30` |

Rebuild from the repository root:

```powershell
$py = 'C:\Users\thecl\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py scripts/package_public_reference.py `
  --source-notebook work/e0-reference/source-v1/biohub-942tta.ipynb `
  --archive 'pilkwang/biohub-tracking-support-pack-50ep-v1=work/e0-reference/downloads/support/biohub-tracking-support-pack-50ep-v1.zip' `
  --archive 'pilkwang/biohub-temporal-unet3d-seed314159-v1=work/e0-reference/downloads/seed314159/biohub-temporal-unet3d-seed314159-v1.zip' `
  --archive 'pilkwang/biohub-deepcenter-unet3d-center-prior-v1=work/e0-reference/downloads/deepcenter/biohub-deepcenter-unet3d-center-prior-v1.zip' `
  --output-dir work/e0-reference/package-r3 `
  --kernel-id clarkkitchen/biohub-e0-public-reference
```

The rebuild is deterministic for the pinned bytes. A second complete build produced identical SHA-256 values for all four package files.
