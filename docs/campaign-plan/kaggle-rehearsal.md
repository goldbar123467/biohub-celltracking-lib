# Kaggle notebook rehearsal launch and recovery

Status: **R3 is operationally complete and settled; one R4 launch is confirmed
with terminal evidence pending**. Completion, output validation, quality review
and quota settlement remain separate from launch identity.

`biohub_ct.campaign.kaggle_rehearsal` controls one private, offline GPU rehearsal of the reviewed E0 package. It uses the existing campaign store for approval, quota reservation, immutable intent, dispatch fencing, launch confirmation, and `LAUNCH_UNKNOWN` recovery. Kaggle remains the run host. This path does not pretend that Kaggle is an SSH worker and does not apply the Vast worker source-copy preflight.

## Reviewed package identity

The built-in `E0_R3_PACKAGE_IDENTITY` refers only to `work/e0-reference/package-r3`:

| Identity | Value |
| --- | --- |
| Notebook | `clarkkitchen/biohub-e0-public-reference` |
| Competition | `biohub-cell-tracking-during-development` |
| Machine shape | `NvidiaTeslaT4` |
| Release digest | `e02e7bd80f17ec89fb6217ea86b1591c6ff949befcb655c66988ecb179e23e78` |
| `package-manifest.json` | `26065311f67657122f6436111b5fc1d15f3451a1bc20e1da1d6c27a3808c20c4` |
| `kernel-metadata.json` | `85ef798c0f980027b334f0613818e0ea62e2eb65c848ab27abfe76142249b812` |
| `artifact-lock.json` | `6f12505158a1c42a5d15ef336e86a4c4b9e99d9087cfed407da84d1e9a1e7241` |
| `submission.ipynb` | `b0c51948112fd407a848de44d52923f2fe70f432e5520060338f9f518afe5e30` |
| CLI-normalized notebook source | `c4d12a7d6b1ad3de5b4ef0883329d9fdd8c4cad7c7ee005e17c89e1354ba72d4` |

The package preflight requires exactly those four regular files, rejects symlinks and extra files, and checks all hashes. It also enforces a private notebook, internet disabled, GPU enabled, TPU disabled, the exact competition, and these exact dataset versions:

- `pilkwang/biohub-deepcenter-unet3d-center-prior-v1/5`;
- `pilkwang/biohub-temporal-unet3d-seed314159-v1/2`;
- `pilkwang/biohub-tracking-support-pack-50ep-v1/10`.

The version suffixes are present in the actual `dataset_sources` sent to Kaggle, not only in the local lock. Installed CLI `2.2.4` accepts all three forms. `status: ready_for_root_review_not_launched` and an empty `algorithm_changes` list are required. Byte drift or a metadata change defines another package and fails before any intent or quota reservation is created.

## Installed CLI contract

Read-only checks on the controller found Kaggle CLI `2.2.4` at `/home/thecl/.local/bin/kaggle` in WSL distribution `Ubuntu-24.04`. Its installed help and source support these operational forms:

```text
kaggle kernels push --path PACKAGE --timeout SECONDS --accelerator ACCELERATOR
kaggle kernels status OWNER/SLUG
```

The installed push implementation emits `Kernel version N successfully pushed` when the API returns `versionNumber`. The adapter accepts only one positive version, a canonical HTTPS URL for the exact expected slug, exit code zero, empty stderr, and no invalid-source or push-error warning. This exact acknowledgment is sufficient to identify the launched version. Any other result is ambiguous because a server mutation may precede a local timeout or error.

The installed client changes notebook bytes before upload. It parses JSON, clears outputs from code cells, joins list-valued cell sources, and serializes the result with default `json.dumps` settings. Preflight binds the resulting normalized SHA-256. The installed CLI does not implement exact source retrieval with `kernels pull OWNER/SLUG/VERSION`: the version becomes part of `kernel_slug`, and the live R3 request failed. Recovery now calls SDK `GetKernel` with separate `user_name`, `kernel_slug`, and `version_label="vN"` fields. The SDK can serialize returned notebook JSON differently, so recovery preserves both the raw returned-source hash and its CLI-normalized hash. Exact output retrieval separately uses SDK `ListKernelSessionOutput` with the same explicit version-label field. The imported `SUBMISSION_RUNBOOK.md` remains a byte-preserved source document; its version-qualified CLI pull example is superseded by this executed contract.

The status command has a material limitation. Although its parser accepts `OWNER/SLUG/VERSION`, `KaggleApi.kernels_status` discards the parsed version and calls `get_kernel_session_status` with only owner and slug. The module exposes this honestly as `current_status(OWNER/SLUG)`. It is never exact-version evidence and cannot confirm, fail, or complete a pinned version.

The Windows package directory and WSL `--path` are separate namespaces. Launch therefore requires a path verifier. `WSLPathVerifier("Ubuntu-24.04")` runs literal, read-only `wslpath -a -u` and the launch path must equal its result. The package hashes and mapping are checked again inside the persisted-intent callback immediately before the single push call.

## Permission and quota contract

A launch requires all of the following:

1. The caller holds the store's live OS-backed `ReviewLease`. The module checks the open lock handle and uses its current fencing token.
2. Campaign status is active and `authorization.private_kaggle_rehearsals` or the legacy `authorization.routine_runs_and_submissions` is explicitly true, with a retained authorization source. The private flag authorizes this exact reviewed rehearsal path; it does not permit routine submissions or scheduler activation. Existing Vast-allocation permission alone is insufficient for Kaggle, but the user's earlier explicit Kaggle allowance authorization remains applicable.
3. A fresh live GPU observation supplies finite `used_hours`, `remaining_hours`, `total_hours`, `active_gpu_jobs`, and the exact ledger ID. The three quota quantities must reconcile, the verified active job count must be zero, and the observation timestamp must match the ledger.
4. The ledger has a finite total, uses `quota_hours`, identifies Kaggle GPU quota, and reconciles to the observation.
5. The caller supplies either a measured quota debit rate per wall hour or a documented debit-rate upper bound with its source, plus the rehearsal timeout, the bounded duration of a final attempt, and the verified platform runtime limit. Rate modes are mutually exclusive. Missing, zero, negative, nonfinite, or inconsistent values fail closed. A documented bound is never recorded as measured.
6. The ledger protects at least `2 * admission_rate * final_attempt_runtime_bound`. The rehearsal reserves `admission_rate * notebook_timeout_seconds / 3600` outside that protected reserve. Both the documented bound and its source are part of the immutable run binding. Reconcile actual account debit after execution before further admission.

No rate, runtime, live balance, or platform limit is hard-coded. The historical upstream runtime and stale September 7 quota are not accepted as current measurements. The first-run documented-bound path avoids requiring a measurement before any metering run can occur; its source and planning margin must be explicit. See the [September 8 full rehearsal](../experiments/e0-kaggle-full-rehearsal-2026-09-08.md).

## Immutable run specification

Register the rehearsal as a normal campaign run before review. Its `execution.host` must be exactly `kaggle`, `max_steps_or_clips` must be 1, and `execution.argv` must equal the literal array returned by `KaggleRehearsalCLI.push_argv`. `execution.working_directory` is the verified WSL package path and `execution.max_quota_hours` is the computed worst-case reservation.

Private-scope runs also bind `authorization_source_sha256` to the exact current
campaign authorization source. Launch checks this under the review lease before
creating an intent. Missing, malformed, or changed source hashes reject the run.
Legacy routine-authorized runs can omit this new field; when present it is
validated in either mode.

The specification must contain an exact `kaggle_rehearsal` object produced from the package and quota contract. It binds the slug and title, competition, dataset versions, release and file hashes, CLI-normalized source hash, WSL path, machine shape, notebook timeout, rate evidence, final-attempt bound, platform limit, and required two-attempt reserve. Documented mode binds `quota_rate_mode`, `documented_debit_rate_upper_bound_per_wall_hour`, and `quota_rate_source`; measured mode retains its original field. The existing run digest makes these fields immutable. The source identity maps package manifest, artifact lock, and kernel metadata hashes to the campaign's source, dependency, and effective-config fields.

Under the held review lease, `authorize_and_launch_rehearsal` atomically approves the exact run, creates the worst-case quota reservation, and persists one launch intent through `CampaignStore.authorize_launch`. `validate_dispatch` claims that intent once. Only then can the literal `kernels push` call run. Reusing the run, intent, request, or reservation cannot issue a second push.

## Receipt and recovery behavior

An exact push acknowledgment confirms launch identity directly and records `OWNER/SLUG/VERSION`. It does not claim a kernel execution status. Output retrieval, manifest checks, CSV validation, and run completion remain separate review steps. A slug-current `ERROR`, `COMPLETE`, or other status cannot be assigned to that pinned version by this interface.

A timeout, process exception, nonzero exit, malformed success text, missing version, wrong URL, invalid-source warning, or push-error warning records `LAUNCH_UNKNOWN`. A version parsed from a response with any such defect remains only a candidate. It cannot bypass recovery proof. The quota reservation stays active and the module never retries automatically. It never infers remote absence from an empty list or empty status.

Recovery uses a version stored in a persisted push receipt only when `exact_receipt` is literally `true`. This covers a controller interruption after the exact acknowledgment was persisted. Any other candidate requires a controlled read-only `pull_exact_version` call. That method:

- requires an empty regular recovery directory and a separately verified Windows-to-WSL path;
- requires an explicit literal WSL SDK Python prefix and serializes `version_label="vN"` in the SDK request;
- enforces a helper deadline, parent timeout, 64 MiB source limit and 1 MiB metadata/response limit;
- verifies raw returned-source bytes and the immutable CLI-normalized notebook hash;
- verifies provider ID, exact returned version, privacy, internet, accelerator, kernel type, language and competition;
- checks the unique unversioned dataset names while recording `UNAVAILABLE_FROM_PROVIDER_METADATA` for dataset versions, which remain subject to the frozen notebook input-file integrity check; and
- exclusively persists `source.ipynb` and `kaggle-exact-version-read-receipt.json`, re-reading their hashes before accepting the evidence.

This proves that a candidate version contains the reviewed normalized source and operational metadata. The installed provider surface supplies no authoritative version creation time or other association with the persisted ambiguous intent. The receipt records `provider_creation_time: null` and `intent_association: unresolved`; it does not accept caller-authored authentication claims. Even a fully verified candidate therefore remains `UNKNOWN` with its reservation retained. Two candidate versions can observe the same slug-current status, so status cannot close this gap. Only separately adjudicated authoritative evidence may justify `RUNNING`, `FAILED`, or `ABSENT`; this module does not infer any of them from an ambiguous launch.

## Activation boundary

The library contains no live quota values and grants no launch authorization. The root controller must register the exact run specification, record a current authenticated quota observation and matching finite ledger, acquire the OS review lock, and supply supported rate evidence and runtime bounds. Calling the implementation with fabricated placeholders fails admission. Live attempts are recorded separately from implementation tests in the experiment record.

## Operator entrypoint

[`scripts/rehearse_kaggle_reference.py`](../../scripts/rehearse_kaggle_reference.py) is the concrete reviewed E0 interface. `--generation r3` is the default; `--generation r4` selects the independently built and verified instrumentation candidate. Both identities are compiled in `rehearsal_packages.py`; arbitrary identity overrides are unavailable. Selecting R4 grants no quota, launch, release-quality or submission permission. Its default behavior is read-only: it preflights the immutable package, optionally verifies the Windows-to-WSL package mapping, and atomically writes a reviewable receipt that lists every missing launch input.

```powershell
$py = 'C:\Users\thecl\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py scripts/rehearse_kaggle_reference.py `
  --wsl-distro Ubuntu-24.04 `
  --package-cli-path /mnt/c/Users/thecl/Documents/Biohub-Cell-Tracking/work/e0-reference/package-r3
```

The actual mutation path additionally requires `--launch`, the canonical store and registered run IDs, decision/intent/request/reservation IDs, reviewer identity, a current strict quota-observation JSON file, its matching ledger ID, rate evidence, notebook timeout, final-attempt runtime bound, verified platform limit, WSL distribution, WSL package path, and installed Kaggle CLI path. Supply `--measured-quota-rate` or `--documented-quota-rate-upper-bound` with `--quota-rate-source`. The command supplies no defaults for live quota, rate, runtime, authorization, or run identity. It acquires the OS review lease and passes all inputs to `authorize_and_launch_rehearsal`, whose atomic reservation and single-dispatch checks remain authoritative. An `UNKNOWN` result forbids retry until reconciliation.

## September 8 live execution evidence

The root ran the new production SDK reader against the actual R3 title-derived
notebook at version 1. Its raw and normalized hashes matched the independently
recovered canonicalization proof, and campaign state did not change. A request
for nonexistent version 999999999 was rejected with an empty output directory;
no latest-version fallback occurred. Receipts are under
`reports/campaigns/e0-r3-kaggle-full-20260908-01/sdk-production-*`.
These reads establish the source-client behavior; they do not establish dataset
membership or model quality.

The R3 notebook later completed as exact provider version
`clarkkitchen/biohub-e0-public-reference-reproduction/1`. The provider reported
5,076.3 seconds and the final manifest reports 5,057.298734274 seconds. The
exact-version inventory enumerated 442 files in five pages. Bounded transport
downloaded all 50,681,873 bytes, and independent validation accepted the
241,400-row CSV's release identity, official format, scorer compatibility,
lineage, visible-input coverage, bounds and graph structure. Its SHA-256 is
`a852d1d07ff8c9307d9b10db7f9b4b12e8b1882f14c5dbeb1316d099f0795b3e`,
byte-identical to the pinned upstream final CSV.

The terminal receipt SHA-256 is
`b111ae1752f21c24c287f75116a980ad79af74898ed2c8a7726f60be5d67710e`.
It finalizes canonical R3 state as COMPLETE and settles the reservation at the
1.41-hour account-display change, leaving no retained R3 reservation. The
validator still blocks quality and resource admission because required upstream
evidence is absent. Validation did not approve or perform a submission.

After fresh evidence showed 28.41 quota hours remaining and zero active GPU
jobs, the R4 telemetry differential was admitted with a 6,480-second timeout,
a 1.980-hour reservation and a protected 26.40-hour final-attempt reserve. One
push was confirmed as `clarkkitchen/biohub-e0-instrumented-reference/1`. This is
launch identity only: R4 has no terminal receipt, downloaded-output validation,
R3 parity result, telemetry-completeness result, runtime-overhead result or
quota settlement yet. The focused implementation evidence is 47 passing
release/output tests and 34 passing telemetry tests; provider execution evidence
is still required before R4 can close.
