# Initial testing and offline architecture evidence

Date: 2026-09-05. These are execution and diagnostic measurements, not full
cross-validation, convergence or leaderboard evidence. No learned model was
trained and no competition submission was made by this work.

## Executed checks

- Vast Python 3.12: 50 tests passed, including actual Zarr/GEFF, pinned official
  scoring, sparse estimated counts, unequal aggregation weights, CSV failure
  cases, exact spatial-search parity, interruption/resume and deterministic builds.
- Windows bundled Python with NumPy: 34 tests passed, 12 optional tests skipped
  because Zarr/organizer packages are absent. This is not official-metric parity
  evidence on Windows.
- Ruff checks and formatting ran on changed Python files.
- Cross-platform generated source ZIP and notebook hashes matched after fixing
  ZIP creator-platform metadata. Final source ZIP SHA256:
  `09760a0c59ef2bbc87a3bb6f28594090527dab6acfb5d2ffdc38b1b7af198f94`.
- Private Kaggle CPU rehearsal v1 completed with internet off on all four full
  example clips, producing a structurally valid CSV. Inference/serialization took
  145.98 seconds; this excludes notebook startup and wheel installation. CSV SHA256:
  `97867bf1b35c88d7a0caacc95584e145906cc60d3d3a4150e731ebc288cc5124`.
  This run preceded the intensity-plateau correction.

## Detection bug and controlled diagnostic rerun

The old detector clipped intensities at the frame's 99.5th percentile before
finding maxima. A checked frame contained over 20,000 saturated voxels. Equal
scores plus the 2,000-candidate cap could discard actual cell peaks. The fix
finds and ranks maxima using original intensities while retaining the normalized
threshold. A regression test verifies bright-peak ranking above that quantile.
No threshold, NMS radius or linking gate was tuned between these runs.

| Clip | Old adjusted edge Jaccard | Corrected | Old node recall | Corrected | Corrected pred/estimated count | Corrected runtime |
|---|---:|---:|---:|---:|---:|---:|
| 44b6_0c582fdc | 0.00000 | 0.19953 | 0.0000 | 0.6338 | 2.4370 | 44.59 s |
| 6bba_062c8d37 | 0.02815 | 0.20265 | 0.0796 | 0.9495 | 2.8144 | 22.30 s |

Both are complete 100-frame clips outside the four example-test IDs. Ground truth
contains 71/930 annotated nodes, while estimated totals are 27,958/6,030. The
corrected runs emitted 68,135/16,971 nodes. One clip has no annotated divisions;
the other has one, which the division-free greedy baseline missed. Peak process
RSS over the corrected two-clip scoring run was 556,116 KiB, about 543 MiB.

This establishes a concrete detector failure and working official diagnostics.
The small, deliberately inspected sample cannot establish generalization. Recall,
density, association and division errors still require independent comparisons
on full embryo-held-out folds. Public test copies cannot serve as validation.

Raw JSON/CSV/log artifacts remain under `reports/architecture-benchmark-v1`,
`reports/architecture-benchmark-v2` and `reports/kaggle-submission-v1` on the
configured machines. The v2 benchmark scored the detector fix before later
formatting, metadata validation and reporting-only changes. Runtime is platform
and clip dependent; hidden-test timing remains an estimate, not a guarantee.
