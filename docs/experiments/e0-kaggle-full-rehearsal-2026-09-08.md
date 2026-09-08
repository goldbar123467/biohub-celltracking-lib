# E0 full private Kaggle rehearsal

Status: **terminal `COMPLETE`, exact-version output independently validated, and
quota reservation settled.** There was one private push and no retry or second
push. The accepted 241,400-row CSV is byte-identical to the frozen upstream
reference.

## Purpose and fixed artifact

This rehearsal executed the complete frozen R3 notebook on Kaggle with internet
disabled, retrieved its exact committed version, and independently validated the
discovered input coverage, CSV, lineage, and manifests. This exercised DeepCenter and the
full postprocessing path beyond the previous eight-frame Vast parity check.
Successful visible-input execution establishes operational reproduction, not a
clean held-out quality score or hidden-test runtime guarantee.

The artifact remains `work/e0-reference/package-r3`, release digest
`e02e7bd80f17ec89fb6217ea86b1591c6ff949befcb655c66988ecb179e23e78`.
Its four file hashes, CLI-normalized notebook hash, private/offline T4 metadata,
and all three numeric dataset versions passed the existing preflight again.
Receipt: `reports/campaigns/e0-r3-preflight-20260908-full.json`, SHA-256
`f33896465097c3b7834c4e0c6082d3ed174d92e7230d611a5d9d4d95e4a568a1`.
No algorithm, model, threshold, or dataset-version changes are proposed.

## Authorization correction

The user explicitly authorized Kaggle cloud GPU use for this project on
September 5. The repository AGENTS.md also names bounded private offline
rehearsals as an intended use of the existing allowance. This authorization
persists and covers this attempt. The initial rehearsal implementation incorrectly
required the broader `routine_runs_and_submissions` flag. The correction records
private rehearsal permission separately, retains its source, and preserves the
existing review lease, exact run binding, quota reservation, and single-dispatch
requirements. It does not grant recurring runs, submissions, purchases, or
scheduler activation. The paused scheduler remains a separate decision.

## Resource admission at launch

At the preparation check, the authenticated CLI reported 0.18 GPU hours used,
29.82 remaining, and 30.00 total, with reset reported as
`2026-09-12T00:00:00`. The authenticated Active Events dialog showed zero active
events. These observations were refreshed before the single dispatch.

[Kaggle staff's T4 announcement](https://www.kaggle.com/discussions/product-feedback/361104)
explicitly states that a T4x2 notebook runtime hour consumes one GPU quota hour.
This is a historical platform statement, not a measured rate for this attempt.
Current [notebook documentation](https://www.kaggle.com/docs/notebooks) and the
[competition code requirements](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/code-requirements)
retain a 12-hour GPU runtime limit. The competition requires internet disabled
and `submission.csv`. Requirements were refreshed on September 8.

The launch plan used a documented admission upper bound of 1.10 quota hours per
runtime hour. The extra 10% was local planning policy for accounting uncertainty,
not a Kaggle claim. The plan protected two 12-hour final attempts at this bound,
or 26.40 quota hours, and reserved another 2.75 quota hours for a 9,000-second
(2.5-hour) platform timeout. At the observed balance, 0.67 hours remained outside
both reservations. A timeout
is an incomplete attempt, not a valid release. The upstream displayed runtime
of 1h 21m 41s motivates this bound but does not predict our runtime.

The run launched through the API without an interactive GPU session. The
operator persisted the run,
decision, exact package binding, authorization source, quota evidence and launch
intent before the single push. Its initially ambiguous title-derived slug was
reconciled before any further action. Quota was captured afterward and the
reservation was settled from the account-display change rather than the
documented admission bound.

## Acceptance and recovery

Accept only a pinned version with independently retrieved matching source and
private/offline metadata, successful exact-version execution evidence, complete
input coverage and a valid CSV plus run manifest. Pull outputs using the existing
R3 operator. Check the input shape contract independently of the notebook's own
manifest. Preserve failure logs and reservations if execution, identity, or quota
accounting is unresolved. No public score or final selection follows from a push
acknowledgment or a valid visible-input CSV alone.

## Launch, provider URL canonicalization, and terminal result

Commit `161b148` contains the scoped permission and documented-rate fixes.
The root independently ran 33 targeted tests before launch; a separate reviewer
confirmed the authorization-source binding and quota changes. One push was
dispatched at approximately 07:10:35 UTC under run
`e0-r3-kaggle-full-20260908-01`. The canonical store reserved 2.75 quota hours
outside a 26.40-hour release reserve. The immediately preceding account quota
remained 29.82 hours, with zero active events.

Kaggle created
`clarkkitchen/biohub-e0-public-reference-reproduction`, version 1, from the
bound title rather than the shorter requested metadata slug. The launcher
correctly retained `UNKNOWN` and its reservation instead of accepting a URL
mismatch. The installed CLI warned about this title/slug mismatch and then
reported version 1 as successfully pushed. Reconstructing that exact warning
and success line produced the original captured stdout SHA-256:
`49bdb98e7730915acfa463891372761b9d98b06e9c848cd17502908d055a919b`.
The reconstructed transcript is preserved beside the original launch receipt.

During execution, the authenticated notebook UI showed a private T4x2 run. The
SDK request using `GetKernel(version_label="v1")` retrieved its source; all 14 cell sources match
the frozen package. Its normalized source SHA-256 is the expected
`c4d12a7d6b1ad3de5b4ef0883329d9fdd8c4cad7c7ee005e17c89e1354ba72d4`.
The provider serializes notebook JSON differently, so the raw returned-file hash
is preserved separately. The installed CLI's version-qualified pull incorrectly
places `/1` in `kernel_slug` and returned HTTP 403; it is not accepted as
exact-version retrieval evidence. The explicit SDK version-label request works.

The SDK returns unversioned dataset names and an inconsistent `last_run_time`;
neither field is treated as proof of pinned dataset versions or launch time.
The exact saved push transcript establishes the accepted version, while the
frozen first notebook cell verifies every mounted artifact against the lock.
The root independently verified the canonicalization receipt and reconciled
the existing launch intent to CONFIRMED under the review lease. The actual
provider identity is the title-derived slug at version 1; the frozen requested
package identity remains unchanged. The original launch receipt and its push
hash remain preserved. The 2.75-hour reservation remained outstanding until
the terminal reconciliation described below.
The receipt SHA-256 is
`c301d5180f1db5ec5eb55482cec6ffd5c304e3c9fb3429710884916e34bfb372`.
The root's focused verification across canonicalization, output transport
integration, release validation and rehearsal admission passed 77 tests with
one Windows symlink-privilege skip. That pre-completion check verified local
behavior only. During the live run, the log recorded four merged prediction
graphs at 906.9 seconds before later notebook postprocessing.

Kaggle subsequently reported exact provider version
`clarkkitchen/biohub-e0-public-reference-reproduction/1` as `COMPLETE`, with a
provider duration of 5,076.3 seconds. The downloaded manifest reports
5,057.298734274 wrapper seconds. The exact-version inventory enumerated all 442
files in five pages, and bounded transport downloaded 50,681,873 bytes.
Independent validation accepted release identity, official format, scorer
compatibility, lineage, complete visible-input coverage, coordinate bounds and
graph structure. The 241,400-row `submission.csv` has SHA-256
`a852d1d07ff8c9307d9b10db7f9b4b12e8b1882f14c5dbeb1316d099f0795b3e`,
exactly matching the pinned upstream final CSV.

The accepted result is
`reports/campaigns/e0-r3-kaggle-full-20260908-01/independent-validation-v1/validation-result.json`.
The earlier `download-validation-v1/validation-result.json` is retained as a
failed validation attempt and must not be cited as the accepted result. The
terminal reconciliation is
`reports/campaigns/e0-r3-kaggle-full-20260908-01/terminal-reconciliation.json`,
SHA-256
`b111ae1752f21c24c287f75116a980ad79af74898ed2c8a7726f60be5d67710e`.
It records canonical state `COMPLETE` and settles the reservation at the
1.41-hour account-display change, leaving no retained R3 reservation. Quality
and resource admission remain blocked by missing upstream fields. This rehearsal
did not approve or perform a competition submission.
