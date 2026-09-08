# Campaign implementation and completion audit

Started 2026-09-08. This is an open audit, not a completion certificate.

Latest checkpoint: the full frozen R3 private Kaggle rehearsal is operationally
**COMPLETE**. Kaggle version
`clarkkitchen/biohub-e0-public-reference-reproduction/1` ran in 5,076.3 provider
seconds; its final manifest reports 5,057.298734274 seconds. Exact-version
inventory found 442 files, and the independent validator accepted the 241,400-row
CSV's identity, format, lineage, visible coverage and graph constraints. The CSV
is byte-identical to the pinned upstream reference. The account display moved
from 0.18 to 1.59 used quota hours, so the R3 reservation was settled at the
display-derived 1.41-hour debit with no retained reservation. This operational
completion does not cure the validator's missing quality and resource evidence,
and it does not authorize a submission or promote E0.

The full R4 telemetry differential has now reached provider `COMPLETE` at
5,706.2 seconds, with 465 exact-version output files independently rehashed.
The CSV is byte-identical to R3, and all 12 detector-coordinate identities and
logical GEFF graphs match exactly. The required telemetry check nevertheless
failed because the frozen timer watched pandas writes while the actual notebook
uses `csv.DictWriter` for submission serialization. The strict failure was
preserved; the canonical run is **FAILED**, with the successful provider state
recorded separately. Its 1.98-hour reservation settled at the 1.58-hour account
display change. Remaining quota is 26.83 hours, including the intact 26.40-hour
protected reserve. See the [R4 differential report](../experiments/e0-r4-telemetry-differential-2026-09-08.md).

The future DictWriter hook and Windows long-path inventory fix are implemented
and locally tested. They do not change the frozen R4 result or prove a corrected
full rehearsal. E1 advancement remains rejected; conditional E2-E5 work remains
unadmitted, and routine campaign authorization and scheduler activation remain
open.

The [readiness refresh](../experiments/campaign-readiness-2026-09-08.md) records
the corrected telemetry package's three identical local builds and executed
embedded-writer smoke, fresh authenticated rules/account limits, and the
remaining conditional science gates. It is still unlaunched and unadmitted.
The original E1 image-inspection requirement is evidenced by the retained eight
AI-reviewed sheets; a human/domain-expert review is not an additional requirement
of the supplied plan.

The corrected full-suite evidence is recorded in the
[campaign suite verification report](../experiments/campaign-suite-verification-2026-09-08.md).
A root-clean native Windows run provisioned independently through `uv` passed
405 tests with 61 explicit optional-dependency/platform skips in 13.20 seconds.
The synchronized Vast Linux run passed 469 tests with 21 skips in 38.30 seconds;
its downloaded artifacts were independently rehashed and its canonical run state
was finalized `COMPLETE`. These suites verify software and campaign contracts on
both platforms. They do not replace the separately recorded real CUDA evidence
or establish model quality.

## Scope and precedence

The user requested reading the markdown plan, implementation by Sol agents, and
independent orchestration/verification. All seven supplied files were read and
copied byte-for-byte; `source-manifest.json` verifies the imports. All existing
project-owned markdown files under `docs/`, the root README/AGENTS, and the local
metric-boundary README were read. Third-party vendor documentation is consulted
at affected dependency boundaries, rather than adopted as project instructions.

The supplied files describe desired behavior. Their template claim that routine
submissions are already authorized is not itself user authorization. The prior
frozen submission had separate explicit authorization. The user subsequently
confirmed the existing Vast 4070 SUPER server may be used. New purchases,
extensions, campaign spending ceilings, and activation of recurring work are
separate from that existing-allocation permission and must not be inferred from
template values. Existing historical spend and the paused prior monitor survive.

The user's earlier explicit Kaggle GPU allowance authorization also persists.
The first private-rehearsal launcher incorrectly required broad routine permission;
this has been corrected with a separate private scope and immutable authorization
source binding. A private rehearsal does not grant recurring jobs or submissions.

Historical July mock reports remain mock reports. Historical September claims
are tied to their recorded revisions; they do not automatically prove current
behavior. Optional/conditional research branches must have an explicit gate
decision, not be silently dropped or run merely to fill an allocation.

## Requirement-to-evidence ledger

Every row remains open until its stated evidence exists and has been inspected.
Passing isolated tests does not close end-to-end operations or model-quality rows.

| ID | Contract and demands | Authoritative completion evidence | Current state |
|---|---|---|---|
| H1 | HOURLY_HELPER purpose, copy-ready prompt, orders 1-9 | One bounded reviewer actually reconciles state, services, costs, jobs, candidates, scores and decisions in order | Live read-only, mutation-enabled launch and downloaded terminal recovery verified; model/release review remains open |
| H2 | Exclusive controller, fencing, stale-dispatch rejection | Contending processes + stale-token rejection using production store/worker | SQLite/OS lock, native Windows lock and stale-token tests pass; exact Vast dispatch token verified |
| H3 | Intent before launch/submission; UNKNOWN retention and reconciliation | Fault injection before/after external acceptance, no duplicate mutation, reservations retained | Fault-injection tests and live one-shot Vast launch/recovery pass; no new Kaggle submission attempted |
| H4 | Approval expires in 60m, digest-bound, consumed once, launched work survives approval expiry | Time-controlled store tests plus actual launch receipt | Store tests pass; live digest-bound intent consumed once and terminal settlement verified |
| H5 | Independently enforced job deadline and minute heartbeat | Actual worker timeout, process identity, graceful stop and forced stop, supervisor death, useful progress distinct from liveness | Linux backstop/ownership tests pass; corrected live `operational-cuda-20260908-02` reached `COMPLETE` for run-spec SHA-256 `a7d3e101545aff6b5ae4864346a0bd11d4c7ce2d7913e61752d9c94225722c59`, fencing token 7, one unit in 3.025262092007324s. The short run did not exercise a minute-scale live heartbeat or cross a deadline |
| H6 | Single iteration under 10m; meaningful-change-only report | Bounded reviewer execution and unchanged-state output test | Read-only review `review-20260908T022206885023Z` finalized run 02 in 33.875s with downloaded completion SHA-256 `d883fd6d2cb3e1d3c3ba296ef7fc6b80f64bc8982b395647adeb9f76a80e89b9`; the immediately following `review-20260908T022313302142Z` returned `NO_CHANGE`, `notify=false`, in 9.030999999959022s. Hard 550s parent limit and unchanged-notification tests pass |
| H7 | Activation steps 1-6; exactly one schedule; first receipt and next run | Verified access + read-only and mutation-enabled iterations + actual scheduler receipt | Read-only inspection found existing heartbeat `biohub-long-run-hourly-check` with status `PAUSED`; `helper-activation.md` binds its replacement prompt to the canonical store and verified entrypoint. No automation was edited or activated, and no scheduled first receipt or next-run time exists; recurring and routine-action authorization remain false and the whole-campaign ceiling remains unresolved |
| G1 | GPU_OPERATIONS live inventory, quota/rules, real CUDA op | Timestamped provider, account, storage and CUDA receipts | `reports/campaign-inventory-20260908.json`, `reports/campaign-kaggle-inventory-20260908.json` verified |
| G2 | Independent provider/quota ledgers; finite worst-case reservation; no double debit | Transactional reservation/settlement tests and live observations | Exact-Decimal ledger, concurrency, protected-reserve, worst-case-plus-grace and atomic terminal-settlement tests pass; one live instance-hour reservation settled from verified elapsed time. USD completion retains its reservation until authenticated actual billing, multiple reservations require explicit allocation, and the whole-campaign ceiling remains unresolved |
| G3 | Two release attempts protected, measured quota debit, fallback reserve policy | Rule/runtime evidence, conservative admission calculation, actual debit reconciliation | R3 started from 29.82h remaining and 0.18h used, protected 26.40h and reserved 2.75h using the documented 1.10 upper bound. The terminal account display showed 28.41h remaining and 1.59h used; the 1.41h display-derived debit settled the R3 reservation. Fresh R4 admission reserves 1.980h for a 6,480-second timeout and leaves 26.430h after the reservation while protecting 26.40h. R4 subsequently settled at a 1.58h display-derived debit, leaving 26.83h remaining, 26.40h protected, and no R4 reservation. |
| G4 | Profile/pilot before expansion; concurrency at most one per platform | Immutable admitted run specs and active-process/provider receipts | The bounded Vast profiles and diagnostics completed through admitted intents. R3 reached COMPLETE and zero active GPU events were verified afterward. One R4 telemetry differential was then pushed exactly once as `clarkkitchen/biohub-e0-instrumented-reference/1`; provider COMPLETE and all 465 downloaded file hashes are verified. Canonical state is FAILED because serialization telemetry is missing; the quota reservation is settled. No new model fit was launched. |
| G5 | Stop/error/OOM policy and ownership; billing distinct from process | Deadline tests plus worker error tests, billing readback, bounded retry history | Ownership, supervisor-death backstop, work-unit finalization, stale PID and deadline tests pass. Historical AMP overflow recovery remains separately documented. No new application OOM/retry was exercised in this pilot; provider billing is separate. |
| G6 | Cache identity includes all numerical/input fields; float32 diagnostic logits | Corruption/invalidation tests + actual cache/reload comparisons | All 16 actual CUDA payloads passed independent model/source/config/frame/precision/transform/TTA identity and strict reload. Replayed native probabilities and frozen control node coordinates matched the contemporaneous production path exactly on all eight frames. |
| G7 | Complete resume state, atomic checkpoints, durable backup/readback | Current trainer resume suite + actual checkpoint hash at both destinations | The immutable R7 full real-dependency suite includes trainer interruption/resume checks. Frozen model and captured source/artifact hashes were independently verified on both controller and Vast. No new training checkpoint was produced by E1. |
| G8 | Stage timing/peak memory/tail runtime; no unmeasured multi-GPU claim | Representative profile with setup, IO, model, extraction, association, solver, CSV, validation | The new real eight-frame dual-model support differential completed in 22.6556 supervised seconds and recorded all eight inner stages plus CUDA allocator/host RSS peaks. Candidate, solved graph and GEFF parity were independently verified. It excludes DeepCenter, later notebook postprocessing, CSV, full-cohort tail time and Kaggle; no complete-runtime bound or speedup follows. The earlier feature-only profile and E1 timings retain their narrower recorded scopes. |
| V1 | VALIDATION_PROTOCOL exact population and provenance classes | Missing/duplicate/overlap test failures + honest historical import | Strict evidence classes, membership checks and exact population tests pass in actual Vast suite; no new complete model evaluation |
| V2 | Official per-sample penalty, edge-union weighting, pooled divisions and exact empty cases | Pinned scorer source + unequal-weight/undefined metrics parity tests | Pinned scorer aggregation parity, unequal candidate-owned weights and undefined-division tests pass; new graph matching still requires actual artifacts |
| V3 | Required detection, localization, caps, edges, divisions, totals and operational fields | Fully instrumented per-clip records + recomputed aggregate | E1 has exact eight-frame count, recall, localization, plateau, cap and timing records for all 56 comparisons, independently recomputed. Estimated-node ratio, per-clip cap rate and temporal adjusted-edge score are explicitly unavailable. |
| V4 | Coordinates through metadata, downsampling, tiles/padding, physical matching and overlays | Known-point round trips, border tests, saved real orthogonal views | Production replay uses corrected stride-scaled physical NMS and raw-coordinate annotation matching. All eight actual XY/XZ/YZ overlay sheets were decoded and inspected by the root AI, with same-slab points, physical scales and hashes; human expert review remains absent. |
| V5 | Development/diagnostic/locked/unknown distinctions; complete paired directions | Frozen manifests, membership proof or explicit unknown label, complete results | Existing 61/71 remains PARTIAL; no clean new holdout claimed |
| V6 | Division all-clips + subset and event review; acceptance preregistered | Pooled event reports, reviewed false/recovered forks, no macro empty-clip substitution | Pooled all-clips and GT-division subset contracts tested; no new real event review or model advancement |
| V7 | Reuse current regression + full real offline path | Integrated suite on actual optional dependencies and completed Kaggle release | Immutable R7 full suite: 265 passed in 25.42s on Vast with real optional dependencies. This precedes the new Kaggle rehearsal module/CLI; their integrated verification is separately recorded below. Offline Kaggle release remains open. |
| E0.1 | Exact score-associated notebook and input versions, hashes/licenses | Pinned source/weights/dependencies + live version/score receipt | Root inspected pinned notebook V1/scriptVersionId 347821442, downloaded archives and local artifact lock; no reproduction score claimed |
| E0.2 | Sequential source audit, transforms, TTA, normalization, solver, overwrites | Effective configuration and source-location audit | Root sequential audit identified effective settings, stale report/print fields and heuristic division proxy; documented in public-reference-e0.md |
| E0.3 | Verify reported constants against exact code, do not substitute | Assertions/source audit of actual selected version | Exact source constants and all 12 unchanged public cells verified; stale summary values explicitly rejected |
| E0.4 | Representative compatibility/timing and unchanged-algorithm private offline package | Actual run + documented adaptation diff | R3 and the additive R4 candidate retain all 12 original cells; the candidate has two byte-identical builds and 18 compiled cells. R3 completed the full private Kaggle notebook in 5,076.3 provider seconds, and its manifest reports 5,057.298734274 seconds. The earlier eight-frame GPU support differential verifies both models, extraction, association, ILP and GEFF on identical input. R4 completed with exact CSV, 12-dataset coordinate-identity and logical GEFF parity. The telemetry acceptance contract failed on missing serialization timing; a future hook fix has local tests but no corrected full rehearsal. |
| E0.5 | Full visible coverage/bounds/graph/runtime/manifest, upstream comparison | Final downloaded CSV validated independently; canonical graph parity on identical inputs | R3 exact-version output was downloaded and independently validated: 241,400 rows across all four visible datasets, valid bounds and graph constraints, and exact identity with the pinned upstream CSV, including byte equality. The validator still blocks quality and resource admission because the upstream manifest omits the required evidence fields. |
| E0.6 | Eligible exact-version submission via helper with overlap label | Operational approval, persisted intent, numeric Kaggle receipt and eventual score | Outstanding; not authorized by template text alone |
| E1.1 | Reproduce frozen control and all three precision variants | Same-frame logits/probabilities/assigned nodes with fixed transforms | COMPLETE for the fixed diagnostic panel: eight cache workers, 16 payloads, all three precision arms, CUDA probability parity and frozen-control extraction parity passed. This is reused diagnostic evidence, not held-out performance. |
| E1.2 | Deterministic connected plateaus/seams and physical NMS | Separate ablation + component geometry/real nucleus inspection | Deterministic connected-plateau and anisotropic physical NMS tests pass. Actual fixed native-AMP overlays show little visual change between legacy and plateau extraction; the AI review is hash-indexed and does not claim exhaustive nucleus annotation. |
| E1.3 | At most 15 initial configurations per variant and one refinement | Preregistered development grid + cached run records | COMPLETE: nine initial configurations per each of six precision/extraction arms, 54 total, followed by exactly two refinements in one stage. No second refinement or grid extension occurred. |
| E1.4 | Full count/recall/localization/caps/score/runtime frontier and visuals | Measured table/plot, decoded overlays, explicit advancement/rejection | Diagnostic frontier and all eight fixed overlay inspections are recorded in e1-pilot-results.md. Best measured predicted-count reduction is 14.7087%, below 20%; 26/26 sparse annotations match. Temporal metrics remain unavailable, so advancement is rejected. |
| E2 | Fixed nodes/features; staged association, harmonic alignment, objective/fork counts, image-supported gaps | Cached identities + paired graph/component ablations on full planned clips | GATE NOT MET on the learned E1 branch: count-reduction proxy failed and temporal quality is unavailable. E0 R3 is now operationally reproduced, but its quality admission remains blocked; no association training/ablation or E2 advancement is claimed. |
| E3 | Training-only teacher QC and matched student supervision experiment | Eligibility/provenance, teacher QC, matched exposure/updates/cost and real development result | E1 leaves the density problem unresolved, which permits the E3 teacher audit without E1 advancement. The September 8 provenance audit pinned official code/model revisions, confirmed gated weights, and found weight terms and pretraining membership unresolved. See e3-teacher-provenance-2026-09-08.md. GPU QC and matched distillation remain unadmitted pending eligibility, training-only scope and finite budget; no teacher/student training occurred. |
| E4 | Optional synthetic graph adapter and matched real comparison | Known-fork/independent-lineage tests plus E2 failure and budget gate | NOT ADMITTED: no completed E2 failure analysis exists to justify the optional synthetic adapter. No synthetic training or derived graph evidence is claimed. |
| E5 | Freeze, conditional refit, exact-artifact rehearsal/release and deadline priorities | Winning candidate evidence, fit membership, exact version receipt | NOT ADMITTED: R3 operational reproduction is complete, but no new winner has passed a complete quality gate. No all-data refit, release promotion or final selection follows from the rehearsal. |
| S1 | STATE_TEMPLATES authoritative transactional store and strict JSON/null/UTC | Atomic DB/event/export tests and no duplicated remote writers | SQLite transactions, canonical finite JSON/digests, OS review lock, monotonic fencing, append-only events and atomic export are exercised on Windows and Linux. Real process-kill tests now prove rollback and complete old/new export recovery; exports reject database/sidecar/default-lock targets. Physical power loss is outside this test scope; see controller-failure-verification.md |
| S2 | Fresh campaign and honest authority/resources/incumbents | Live-backed new campaign export preserving old spend and identities | Live campaign store preserves prior spend and closed-campaign identity. A fresh 04:25 UTC history receipt binds public-score incumbent 54308858 at 0.826; release eligibility is not established and validation/final-selection pointers remain null. E1 has a durable REJECT_ADVANCEMENT decision. Whole-campaign funding and new release evidence remain unresolved. |
| S3 | Immutable run, heartbeat, result, metrics, approval and submission schemas | Schema tests and actual worker/release records bound by hash | Exact run/source/dependency/config/data identity, dispatch token, downloaded completion/log and first live terminal record were hash-bound; recovery rejects altered log/result/artifacts, wrong intent/token and missing coverage. Release/metrics records still require actual model evidence |
| S4 | Explicit state transitions, event receipts, durable handoff | Invalid transition rejection, recovery tests, reconstructable export | Invalid transitions, stale tokens and exact artifact recovery are tested. The corrected operational CUDA run and all admitted E1 workers are COMPLETE with append-only receipts and settled reservations. No active model job remains in canonical state at 03:58 UTC. |
| R1 | SUBMISSION_RUNBOOK refreshed rules/account/existing results | Rule snapshot and actual history, no duplicate old submission | A fresh read-only account history receipt at 08:55 UTC confirms submission 56086171 is COMPLETE with both numeric scores unavailable and 54308858 remains COMPLETE with public score 0.826 and no private score. No new submission was made. The earlier allowance observation was five. |
| R2 | Frozen release source/model/deps/settings/eligibility/runtime; hidden discovery | Immutable manifest and offline checks inside notebook | R3 freezes source/mounted artifacts and exact version-qualified inputs; hidden test discovery and final CSV checks are in the offline notebook. One private E0 R3 push produced exact version 1 under the title-derived provider URL; source and private/offline flags were independently verified. |
| R3 | Rehearsal checks 1-6, exact-version output retrieval and independent full CSV validation | Completed version receipt, hashes, graph/bounds/coverage/timing | COMPLETE for operational rehearsal evidence. Exact v1 inventory enumerated 442 files in five pages; bounded transport downloaded 50,681,873 bytes. Independent validation accepted identity, official format, scorer compatibility, lineage, all four visible shapes and 241,400 rows. The final CSV is byte-identical to the pinned upstream CSV. The terminal receipt settled 1.41 quota hours and left no retained reservation. Quality/resource admission remains blocked and no submission approval follows. |
| R4 | Three submission classes and daily/exploration/reserve accounting | Reviewer eligibility and account-history tests, explicit authorization source | Account history is unioned with intents; unscored first references and unclassified manual entries consume exploratory slots. Final reserve use requires a permitted reason, explanation and hash-verified evidence preserved in the frozen candidate document. Missing/changed evidence and mutation-after-freeze tests pass. Routine authorization remains false; any actual reserve decision still requires evidence review. |
| R5 | Verified installed CLI forms, intent and receipt persistence | Read-only actual CLI tests + ambiguous mutation simulation + actual gated receipt | Installed CLI forms were inspected. The actual single R3 push initially retained UNKNOWN on a slug mismatch, then exact original stdout-hash recovery and a pinned SDK v1 source read supported canonicalization and terminal reconciliation without a retry. R4 was subsequently pushed once with the title-fixed slug and is CONFIRMED as exact version 1. No new competition submission exists. |
| R6 | Public/validation/final pointers distinct; actual final-selection confirmation | Completed comparable scores and actual Kaggle selection receipt | No new final selection; outstanding |
| R7 | Final freeze/recovery time and stop at expiry/deadline | Admission/reviewer tests and saved deadline policy | Per-run deadline/expiration and protected-release-reserve admission are implemented and tested. A complete routine campaign ceiling, schedule and final freeze/recovery allocation are unresolved. |
| ER1 | EVIDENCE_REGISTER distinguish source types and historical limits | Source provenance in every report; no mock numbers used as real evidence | Historical audit read; source pack preserved |
| ER2 | Resolve all seven highest-priority unknowns | Current state/receipt, version/membership, split history, runtime, measured ablations and objective audit | Source/version, canonical runtime state, E1 ablations, and R3 full runtime/display-derived debit are documented. Unresolved: E0 training/evaluation overlap, complete temporal quality and resource telemetry, corrected full R4 serialization telemetry, routine funding/activation and final-release evidence. The audit is not complete. |

## Executed verification at this checkpoint

- WSL Linux worker watchdog: 11 tests passed, including hard deadline, supervisor
  death backstop, stale PID identity, lingering descendants, work-unit bound,
  literal argv, duplicate output directory and artifact-gate failure.
- Kaggle adapter: 11 tests passed, including timeout/unknown without retry,
  exact numeric ID, invalid CSV/NaN/duplicate receipts, quota units and intent gate.
- Live read via the adapter succeeded for history, quota and current submission
  allowance. A CLI parser error may return a misleading shell exit status, so the
  adapter validates the actual response schema.

No row is closed solely because an agent says it is complete. The orchestrator
must inspect the implementation, tests, actual outputs and any uncovered demands.

## Verified checkpoint at 2026-09-08 02:15 UTC

- An immutable source snapshot was uploaded to an isolated directory on the
  existing Vast server and every file hash was checked. The full suite passed
  **198 tests in 22.71 seconds**, including actual CUDA recovery tests. Evidence:
  `reports/jobs/campaign-integration-20260908-04/output.log` on Vast. The snapshot
  archive SHA-256 is `92bcf845f64e139879ad2fc352e407bc9ad762f7f6406c9d4ca957a6b88f66d4`.
- Earlier isolated test attempts failed because the detached launcher used a
  different working directory, the snapshot omitted a notebook fixture, and the
  harness omitted the production CUDA determinism environment. These were setup
  failures and are preserved in jobs 01 and 02. The corrected production setting
  was applied before starting Python; no failed test was relabeled as passing.
- `review-20260908T020530513731Z` reserved and consumed one launch intent for
  `operational-cuda-20260908-01`, with fencing token 5 and exact run digest
  `994bd9a9fa5df2b7eaf81303045a742b486b808a1c0f9adf30939bfd1229ee89`.
  A later read-only iteration downloaded its terminal receipt and log, verified
  hashes, and atomically settled measured supervised wall time.
- That first worker is **STOPPED**, not COMPLETE: the supervisor observed the
  final unit before Python finished shutting down. A new regression test verifies
  bounded finalization grace. The correction will receive a separate run ID and
  source identity; the first result remains unchanged.
- The E0 package preserves all 12 public algorithm cells exactly. Independent
  rebuild and strengthened generated-CSV validation tests pass. The selected
  package is `work/e0-reference/package-r2`; notebook SHA-256 is
  `b0c51948112fd407a848de44d52923f2fe70f432e5520060338f9f518afe5e30`.
  No E0 notebook run or submission has occurred. Its upstream local sweep uses a
  division proxy and is not official held-out evidence.
- The existing helper schedule remains PAUSED. Source implementation, live
  verification, numeric authorization, schedule activation and scientific
  advancement are separate completion requirements.

## Verified operational checkpoint at 2026-09-08 02:26 UTC

R4 source archive SHA-256 is
`f01881cfa4092b150999b05af727314c0ad86023c380527b99271a5d2e057250`.
Its 153 files were verified on Vast; the full suite passed 235 tests in 24.18s.
The durable log is `reports/campaigns/campaign-20260908-01/integration-05/output.log`.
Subsequent controller hardlink and typed-identity fixes passed independent focused
checks. Native Windows controller checks passed 53 tests; one symlink test skipped
because the account lacks that privilege, and the same case executed on Linux.

The new run `operational-cuda-20260908-02` completed with exit 0, one unit, no stop
reason, and 3.025262092007324 seconds supervised wall time. Its actual RTX 4070
SUPER CUDA output was 140.0, independently matching the expected sum of squares.
Run digest: `a7d3e101545aff6b5ae4864346a0bd11d4c7ce2d7913e61752d9c94225722c59`;
fencing token: 7. The downloaded arithmetic artifact SHA-256 is
`010ece34b89e6e91f2f2fcc4ca7e14d3884818349e603f9e560bdc6c642d3bda`.
Recovery review `review-20260908T022206885023Z` independently downloaded and
hashed completion/log/result/output, finalized COMPLETE and settled instance-hours.
Completion SHA-256:
`d883fd6d2cb3e1d3c3ba296ef7fc6b80f64bc8982b395647adeb9f76a80e89b9`.
The next read-only review `review-20260908T022313302142Z` took 9.031s and returned
NO_CHANGE with notify=false. No scientific quality promotion follows from arithmetic.

Integration job 05 runtime was added retrospectively to the verification ledger
with a separate observed-runtime settlement. It is not represented as a prelaunch
approval or provider invoice. Total recorded verification spend is
0.02517642376193931944444444445 instance-hours, with no outstanding reservation
at this checkpoint. The 0.25h cap remains an internal operational-verification
envelope, not a whole-campaign allowance or permission to extend the rental.

## Capture and compatibility checkpoint at 2026-09-08 03:10 UTC

The immutable R5 suite passed 243 tests in 25.00 seconds with real Vast optional
dependencies. Evidence is `reports/campaigns/campaign-20260908-01/integration-06/output.log`.
This result covers R5, not later evaluator or rehearsal edits.

E0 compatibility run `e0-profile-20260908-01` completed with both pinned public
checkpoints, exact upstream transforms, expected feature shapes and finite output.
The profile took 3.947828511 seconds and peaked at 748,850,688 allocated CUDA
bytes. Profile SHA-256 is
`7096d9f08453c23268beeef8396f15fc436e4d8693dd7d8fc6435b6b5a1da4f7`.
This exercised feature inference on one real adjacent-frame input; it did not
execute the complete solver, DeepCenter stage, Kaggle notebook or submission.

All eight preregistered E1 cache runs completed under separate admitted intents,
with serial reserve/settle/re-admit decisions. Every run produced its exact seven
declared artifacts. The controller downloaded and verified terminal result and
artifact hashes. An additional independent reload checked all 16 frame payloads
against their raw typed-array hashes, frozen source identity, float32 storage,
finite logits, positive blend weights, CUDA device identity, and expected native
float16 versus float32 forward dtype. The receipt is
`reports/campaigns/campaign-20260908-01/e1-cache-independent-validation.json`.
The frozen capture plan remains SHA-256
`6f57b1fdac0eb2e611fa8166d0e03c96b1ca6bd447f772130c3c550d0cce7aec`.

The ledger has 0.04239345609556185041666666668 recorded instance-hours spent,
0 outstanding reservations and 0.2076065439044381495833333333 available at this
checkpoint. These are supervised verification durations, not provider billing.
The downstream 54-cell evaluator remains unexecuted pending independent source
and receipt-identity review. No scientific advancement follows from cache capture.

## Current verified checkpoint, 2026-09-08 04:08 UTC

The earlier sections are dated historical checkpoints. The current requirement
table and this section supersede their then-pending states.

The immutable R8 archive has SHA-256
`86c349b977d7aca2332a34d943b42639e67e899161f1959b12d50a44b00929cd`.
All 172 files were rehashed on Vast before execution. The full suite passed
**284 tests in 28.64s**, with one explicit skip because the ignored local R3
package was absent in the isolated source snapshot. Evidence is
`reports/campaigns/campaign-20260908-01/integration-09/`.

After that snapshot, the operator script/test received one bounded change:
protecting the atomic `.partial` receipt destination from overwriting an input.
The final exact script/test hashes are respectively
`0205a7c1a31fc1a93173486ebe91e43c118c9090e98d2414ee10f8cd6e8b1ac1`
and `63a5246e59ba640ec82f479f691f53c4d930cfdad241c2867d11aa2b5a62ab17`.
Root verification of current source passed 65 targeted tests on native Windows
(one Windows symlink-privilege skip), then 20 targeted tests on WSL Linux with
no skips. The final delta is covered by those targeted runs, not attributed to R8.

The root separately rebuilt actual E0 R3 from all pinned archives. All four
package files matched the reviewed hashes exactly. Actual Windows preflight and
WSL mapping passed, with `launch_requested=false`; receipt:
`reports/campaigns/campaign-20260908-01/e0-r3-root-preflight.json`, SHA-256
`3de90b35890a9777c9758037db5639ac124be10bb61041dde8196d4791cab94a`.
This is package evidence, not notebook execution or launch admission.

The E1 evaluator completed in 160.200053624 seconds supervised wall time. All 54
initial rows and two single-stage refinements were independently reaggregated.
The [result report](../experiments/e1-pilot-results.md) and its durable hash index
record the 14.7087% predicted-count reduction, unchanged sparse-annotation recall
and localization, reduced frame cap frequency, and rejected advancement.
All eight actual fixed control/plateau overlay sheets were decoded and inspected
by the root AI. Human expert review and complete temporal metrics remain absent.

Read-only review `review-20260908T035832090842Z` completed in 9.329 seconds and
returned `NO_CHANGE`, `notify=false`. It observed 29.82 Kaggle quota hours
remaining, five submissions available, and submission 56086171 COMPLETE with no
numeric score. Submission 54308858 at 0.826 remains the only observed numeric
account score. No incumbent was promoted and no duplicate submission was made.

All 13 admitted worker records are terminal: the first smoke is STOPPED, its
corrected replacement and all other workers are COMPLETE. After settling the
30.622370885-second supervised R8 verification, the internal 0.25 instance-hour
verification ledger records 0.1151943082772428195833333333 spent, zero outstanding
reservations, and 0.1348056917227571804166666667 available. These are supervised
implementation-verification durations, not total rental billing or a routine
campaign budget. `campaign-current-20260908.json` is a fresh export; SQLite remains
authoritative.

No hourly schedule was resumed, no new rental was purchased, no E0 notebook was
pushed and no conditional fit was started. A finite routine campaign limit and
explicit authorization to activate routine runs/submissions remain required;
measured Kaggle debit and a conservative complete-runtime bound also remain
unresolved. These open requirements prevent declaring the entire plan complete.

## Controller and output-verification checkpoint, 2026-09-08 04:49 UTC

The root reviewed and corrected the Sol agents' exact-version transport and
independent release validator. The integrated targeted suite passes **99 tests
on native Windows in 4.74s** and **99 tests on WSL Linux in 4.40s**. This covers
real child-process death at transaction/export boundaries, protected-store export
paths, account-history/intent submission accounting, immutable reserve decisions,
Kaggle rehearsal contracts, executed output-helper success/path/deadline behavior,
and independent CSV/package/manifest validation. It is not a full-suite rerun of
all training dependencies or an E0 notebook execution.

Logs and the reviewed source/test hashes are in
`reports/campaigns/campaign-20260908-01/integration-10/verification.json`, SHA-256
`29b1a839bab271bee97d7b4aec6131745d6b5526f03840d21210aaee9d55c6e9`.
The new source/test files pass Ruff; the full tracked diff passes whitespace checks.

The authenticated SDK inventory probe used `version_label=v1` and returned one
file and a continuation token. The same exact owner/slug with `v999999999`
returned HTTP 404. Neither call downloaded output bytes. The sanitized receipt,
with its current documentation hash, is
`reports/campaigns/campaign-20260908-01/exact-version-output-probe.json`, SHA-256
`ce091f8bf1b64471b72189044943ccb61d41feffd22eaad7564118af8603fb38`.

Transport success remains distinct from structural validation, quality and
resource admission. A completed manifest or the mere presence of telemetry
fields cannot produce quality/resource PASS. The output helper has a local
hard exit deadline as well as its parent timeout; SDK exception handling cannot
swallow that deadline. No exact-version E0 output exists yet to validate.

The [E0 telemetry audit](e0-telemetry-audit.md) identifies missing measurements
and distinguishes detector-retention fallback from fabricated output. A separate
instrumented package is in preparation, retaining R3 as a frozen control. It must
preserve the twelve public cell sources and pass differential output checks
before operational use. The [E3 provenance audit](../research/e3-teacher-provenance-2026-09-08.md)
pins the official code/model revisions but leaves gated weight terms and training
membership unresolved. No teacher use follows from the code's BSD license.

## Instrumentation and operator integration checkpoint, 2026-09-08

The root reviewed the three Sol contributions and the generated notebook glue.
The final targeted suite passed **163 tests on Windows in 7.93s** and **162 tests
on WSL Linux in 6.33s**, with one optional CPU-Torch parity test skipped on Linux
because that test environment lacks Torch. The CPU-Torch test executed on
Windows. Ruff passed for all newly added Python sources and tests. The final
receipt confirms that tested source bytes did not change during execution and
rehashes all seven original imported markdown files against their manifest.

The receipt is
`reports/campaigns/campaign-20260908-01/integration-12/verification.json`, SHA-256
`7574baa472ecc8e1c3b264b10f581011c4d20b0c217b84053e378d4483192fb5`.
It covers the controller, release/output operator, launcher, support and notebook
telemetry, base and instrumented builders, actual CPU kernels and real telemetry
producer/collector child processes. This is not a full optional-dependency
training-suite rerun.

The independent runtime integration uses the actual standalone support helper
in two child processes. The production command follows the public single-worker
path without `--method`. A validation process has the same dataset name and
inherited arm but a different input root and raw coordinate payload. The
collector selects only the launched production PID, verifies invocation and
dataset outcomes, rehashes the raw bytes, checks bounds/dense frame counts, and
preserves CUDA unavailability instead of manufacturing a measurement.

Generated IPython callback/close tests cover successful restoration, missing
callbacks, repeated callbacks and public-cell failures under an exception-
swallowing event dispatcher. The resource sampler rejects cap exhaustion,
persists partial coverage, and computes its reported maxima only from saved
samples. Sampling can still miss transient peaks; it is distinct from the
support processes' allocator and operating-system peak measurements.

The [instrumented package](instrumented-public-reference-package.md) has two
byte-identical builds against the actual pinned archives and original notebook.
All 12 original cells are unchanged, all 18 packaged cells compile, and the
candidate preflight passes with its explicitly reviewed build identity. The
default R3 gate rejects that identity. A separate R3 rebuild reproduces all four
frozen files. The final candidate build receipt is
`work/e0-reference/telemetry-root-build-verification/candidate-receipt-final.json`,
SHA-256 `205d3b0e3ec303f5a47ff431d1a199c510118f2e30dcc7c935ee564151f854a0`.

The [output operator](kaggle-output-operator.md) now gives a concrete local
validation command and an explicitly bounded exact-version download command.
It accepts only the compiled R3 identity and does not grant approval or mutate
the campaign. The instrumented candidate remains outside that allowlist until
the required model/graph differential and operational review are completed.

No instrumented model run, Kaggle push/submission or schedule activation occurred
in this checkpoint. Real-model/GPU/graph parity, full offline rehearsal, measured
Kaggle debit, complete-runtime bounds, finite routine campaign limits and routine
authorization remain outstanding. The original plan is therefore **not complete**.

## Real-model support-graph parity checkpoint, 2026-09-08 06:31 UTC

The [preregistered eight-frame differential](../experiments/e0-model-graph-parity-2026-09-08.md)
ran once on the existing allocation under the canonical controller. Its
0.131-instance-hour reservation covered the 360-second supervisor cap, 15-second
shutdown grace and 25% planning margin within the remaining implementation
allowance. No campaign ceiling was enlarged. The worker reached `COMPLETE` with
exit code 0 in 22.655628693988547 seconds, run-spec SHA-256
`3ebdada5c421af58f9a247bcdb75200859b777f082bed577d0bdc713855e6b19`, fencing token 48.

Both arms used both real public checkpoints and full raw spatial dimensions.
Their 1,799 detections and 1,527 candidate edges matched exactly, as did the
solved 1,689-node/1,465-edge graphs and logical GEFF contents. Both contained an
eight-frame connected component. The root rehashed all 82 archived files and
used a separate checker to recompute candidate bytes, graph construction and
raw GEFF-to-solved-graph equivalence. A Sol reviewer independently checked
source-chain reconstruction, immutable identities, telemetry lifecycle, stage
counts, resource availability and archive hashes. Four focused comparator tests,
compilation and Ruff checks passed.

The selected clip is a known secondary-model training member, with primary
membership unresolved. This is operational parity evidence, not model quality
or a holdout. The effective public pooling kernel is 3 microns; the checkpoint's
5-micron configuration value is not the notebook launch's extraction setting.

Review `review-20260908T063222554571Z` downloaded and reconciled the completed
attempt. Completion SHA-256 is
`1ab1735c6b32bd3e79efec7cb8a680fb44ded5c49c379b633b069eeac3992574`;
parity receipt SHA-256 is
`bfb52b6d4fc8db1623b4859f30f983583990e6d0e5ca941fb6fff8207a3be90e`.
The independent root receipt is `work/pv9a/root-receipt.json`. The canonical
ledger now records 0.1214875384700174 instance-hours spent, zero reservations and
0.1285124615299826 available, separate from provider billing.

## Full R3 terminal reconciliation and R4 launch checkpoint, 2026-09-08

The frozen R3 notebook completed on Kaggle under the provider-canonicalized
identity `clarkkitchen/biohub-e0-public-reference-reproduction/1`. The provider
reported 5,076.3 seconds; the downloaded final manifest records
5,057.298734274 seconds for the notebook pipeline. The exact-version inventory
passed with 442 files across five pages. Bounded transport downloaded all
50,681,873 bytes, and the independent validator completed against the downloaded
CSV, manifest and exact-version proof.

The 241,400-row final CSV covers all four visible datasets and passes identity,
official-format, scorer-compatibility, lineage, shape, bounds and graph checks.
Its SHA-256 is `a852d1d07ff8c9307d9b10db7f9b4b12e8b1882f14c5dbeb1316d099f0795b3e`,
which is byte-identical to the pinned upstream final CSV. This establishes
operational reproduction, not new model quality. The validator still reports
`BLOCKED_MISSING_UPSTREAM_EVIDENCE` for both quality and resource admission,
including missing cap/fallback/retry telemetry, stage times, memory peaks,
runtime headroom and a directly measured debit.

The R3 terminal reconciliation receipt SHA-256 is
`b111ae1752f21c24c287f75116a980ad79af74898ed2c8a7726f60be5d67710e`.
It records state COMPLETE and settles the reservation at 1.41 quota hours from
the account display change from 0.18 to 1.59 used hours; the underlying display
has 0.01-hour resolution. Canonical state retains no R3 reservation. A fresh
read-only account history at 08:55 UTC still shows only the two earlier completed
competition submissions and no score change.

The subsequent R4 telemetry differential used fresh account evidence of 28.41
remaining hours and zero active GPU jobs. Its immutable admission reserves 1.980
hours for a 6,480-second timeout, leaves 26.430 hours outside that reservation,
and continues to protect 26.40 hours for two final attempts. Exactly one push was
confirmed as `clarkkitchen/biohub-e0-instrumented-reference/1`. At this historical checkpoint, no terminal
status, exact-version output, R3/R4 differential, telemetry validation, runtime
overhead or post-run quota settlement had yet been recorded for R4. The then-current
47-test release/output suite and 34-test telemetry suite pass, but tests cannot
substitute for those provider artifacts.

R4 therefore remained an active operational verification at that checkpoint, and its telemetry
cannot add the instrumented candidate to the release allowlist by itself.
Quality promotion, complete temporal evidence, routine campaign limits,
routine-action authority, schedule activation and final submission evidence
remain unresolved. The full plan remains **not complete**.

## Verified R4 terminal checkpoint at 2026-09-08 10:54 UTC

The [completed differential audit](../experiments/e0-r4-telemetry-differential-2026-09-08.md)
supersedes the R4 pending statements in the preceding historical checkpoint.
Provider execution completed in 5,706.2 seconds; all 465 downloaded files were
rehashed. CSV bytes, 12 detector-coordinate identities, and 12 logical GEFF
graphs match R3 exactly. The strict telemetry acceptance failed on missing
submission serialization timing. This failure is preserved as canonical
`FAILED`, alongside provider `COMPLETE`. The observed 1.58-hour debit is settled,
with zero outstanding reservation and 26.83 hours remaining, of which 26.40 are
protected.

The future DictWriter timer and Windows long-path fix passed local verification:
409 tests with 61 explicit skips in the independently provisioned native Windows
full suite; 73 tests with one Windows-only skip in the focused Linux WSL suite.
The corrected full Kaggle rehearsal remains unexecuted. Routine authorization,
the campaign ceiling, scheduler activation, complete model-quality evidence and
final release/submission gates remain unresolved. These are open requirements,
not completed deliverables.
