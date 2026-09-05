# Long-run validation and promotion contract

Date: 2026-09-05. Status: execution gates for the first learned detector run.

The experiment must answer whether a sparse-label detector improves tracking
over the frozen classical pipeline on an excluded embryo. Successful training,
falling patch loss, and a valid CSV are separate outcomes from improved tracking.

## Data and supervision

Use the frozen `configs/embryo-splits.json` without removing missing clips.
The complete download and `scripts/verify_data.py --sha256` are prerequisites.
The verifier checks every expected path and byte size against the API inventory
and rehashes contents against ZIP-extraction receipts. This establishes local
integrity; the receipts are not an organizer-signed hash manifest.

Only training IDs may enter patch sampling, fitted statistics, or augmentation
decisions. All views and timepoints of the excluded embryo stay outside fitting.
Unknown voxels have zero supervision weight, except any explicitly recorded
background assumption in the frozen method. A batch with no supervised voxels
must fail before an optimizer update rather than create apparent progress.

## Selection and evidence limits

Freeze the initial architecture, patch policy, and inference settings before
the substantial run. Periodic patch-loss measurements are health diagnostics.
They are not the official tracking metric and cannot establish model quality.

An internal development panel may reserve 10% of clips from the training
embryo, with its exact IDs frozen before fitting. Exclude these IDs from patch
sampling. Because those clips share an embryo with the fit partition, internal
development measures within-embryo behavior and supports checkpoint selection;
it is not independent embryo transfer evidence. Evaluate the other embryo
after fitting with the selected checkpoint and previously frozen inference
settings. Do not feed its labels back into that fold's fitting or selection.

Use a named checkpoint selection rule, record its direction, and preserve
`latest`, `previous`, and `best` separately. If an embryo score selects a
checkpoint or threshold, label that score **selection validation**. It is no
longer an untouched assessment. With only two embryo groups, both directions
offer useful but limited transfer evidence. Do not manufacture additional
independent embryos by randomly splitting their frames or fields of view.

Compare learned and classical graphs on identical complete held-out IDs with
the pinned official metric. Report each direction separately, including node
recall, predicted/estimated counts, edge TP/FP/FN, adjusted edge Jaccard,
division TP/FP/FN and score. Any shortened diagnostic panel must be named and
listed explicitly; do not present it as a full-fold score. Example-test clips
are training copies and measure inference execution only.

The accepted compute ceiling is eight GPU-hours on the existing instance,
including smoke tests and substantial candidates. Reserve time for validation
and checkpoint readback before assigning the training duration. Record observed
throughput and the resulting allocation, enforce the overall deadline, and stop
cleanly when the budget expires. A stopped run remains an experiment even if
its planned training schedule did not finish. Hourly monitoring does not enlarge
the budget or authorize automatic retry after a failure.

## Recovery and release

Before scaling, compare resumed and uninterrupted next updates on one hardware
and dependency stack. Recovery must restore model, optimizer, scheduler, scaler,
RNG, progress/data-order state, configuration and split/source identity.
Checksums must be verified before deserialization and identity mismatches must
fail before mutating the destination training state. Preserve the previous
checkpoint when a save fails. A same-stack recovery test does not prove
cross-version or cross-device numerical determinism.

The existing Kaggle builder and evaluator initially support the classical
pipeline. A learned checkpoint requires all of the following before promotion:

- Learned inference integrated through evaluation and the streaming submission
  entry point, using the same preprocessing, model configuration and thresholds.
- A private, versioned checkpoint attachment with a verified SHA256; checkpoint
  files and competition data must stay out of the public source repository.
- Offline Torch availability and a real model load/predict check in Kaggle's
  environment. CPU/GPU metadata and the measured inference budget must match
  the selected detector; CUDA version strings alone do not prove interchange.
- Manifest binding of source, input identity, split, model, preprocessing,
  inference config, dependencies, validation evidence and reproduction command.
- Complete internet-disabled Kaggle rehearsal, actual test-ID discovery and
  production CSV validation within the competition runtime constraint.

Keep the classical fallback intact. After an authorized scored submission,
record its processed status, identifier and public score. Private leaderboard
performance remains unavailable until Kaggle releases it. Notebook visibility
is separate from public versus private leaderboard partitions.

## Executed recovery checks

On 2026-09-05, the independent training contract and recovery suites passed
21 tests in 5.88 seconds on Vast (RTX 4070 SUPER, Python 3.12.14, Torch
2.9.1+cu128, NumPy 2.5.2). Production trainer tests interrupted loading after
two updates, resumed from the committed checkpoint, and compared four-update
results with uninterrupted training. Model, optimizer, scheduler, scaler, RNG
and best-score states matched exactly on CPU and CUDA with AMP. Additional
tests cover sparse-mask gradients, XY downsample coordinate alignment, fit/dev
exclusion, low-weight background normalization, corruption rejection before
deserialization and save-failure recovery.

The initial CUDA test exposed an unsupported deterministic `avg_pool3d`
backward operation. The model replaced pooling and interpolation with
deterministic block operations before the passing run; determinism enforcement
and equality tolerances remained unchanged. This is synthetic execution and
recovery evidence, not convergence or held-out model-quality evidence. The
ignored `reports/independent-training-checks-2026-09-05.json` records file hashes
and the executed test scope.
