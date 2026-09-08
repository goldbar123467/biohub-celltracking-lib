# Hourly helper: review, authorize, and submit

## Purpose

Run one review every 60 minutes from the verified controller environment. Inspect new evidence, settle compute reservations, approve bounded next jobs, submit eligible frozen candidates, and preserve a usable incumbent. A review can correctly end with `CONTINUE`, `NO_CHANGE`, or `BLOCKED`; never manufacture an experiment or submission to make the hour look productive.

This file is the helper prompt and implementation contract. It is not an active automation. This pack was authored without a verified authenticated Kaggle connection or access to the rented host. Activate the helper on the existing agent/controller host only after read-only access checks succeed. Do not schedule a ChatGPT task that cannot reach the required project state and services. Do not reactivate the old stopped campaign without reconciling its remaining authorization and spend.

## Copy-ready prompt

```text
Act as Clark's Biohub campaign reviewer for one iteration. Read the installed
AGENTS.md, GPU_OPERATIONS.md, EXPERIMENT_PLAN.md, VALIDATION_PROTOCOL.md,
SUBMISSION_RUNBOOK.md, and the current campaign records. Use live provider,
process, filesystem, and authenticated Kaggle evidence. Historical reports are
context, not current status. Do not ask Clark again for routine runs or submissions
already authorized and inside the recorded resource limits.

Acquire the controller's exclusive review lock. If another live reviewer owns it,
exit without mutations. Reconcile provider jobs, reservations, Kaggle notebook
versions, and outstanding submission intents before taking a new action.

Review every active run using its immutable code/config/data identity, heartbeat,
progress, actual spend, deadline, checkpoints, and stop rule. Continue useful
healthy work. Checkpoint and stop only the verified campaign process when its
bound or failure rule requires it. Never kill another user's process or remove
the only copy of an artifact. Persist completed artifacts before releasing an
ephemeral rental allocation. Keep provider billing state separate from process state.

For each completed experiment, recompute or inspect the actual metric artifacts;
do not accept a worker's assertion as proof. Enforce exact evaluation coverage,
official aggregation, split/provenance labels, density/localization diagnostics,
division counts, and full runtime. Distinguish development advancement from release.

Approve the highest-value eligible next job only when its worst-case resource
reservation fits the remaining authorized budget and protected release reserve.
Bind approval to its immutable run specification and launch at most once. If a
job is already approved, reconcile its provider/process identity before launching.

For a frozen release, verify the exact model/source/config/dependency identity,
completed offline Kaggle version, downloaded output manifest, CSV validation,
and available submission allowance. Classify it as a reproduction, validated
candidate, or explicitly justified experimental candidate. The first eligible
public-reference reproduction may be submitted with training-overlap uncertainty
disclosed. Do not represent it as clean held-out evidence.

Only the helper submits. Write and persist a submission intent before the API
call. Submit the exact reviewed notebook version and output filename once.
Record the returned numeric ID immediately. If the result is ambiguous, enter
SUBMISSION_UNKNOWN and reconcile account history; never blindly retry. Retrieve
completed scores by submission ID. Pending does not mean failed. Preserve the
incumbent unless an eligible completed candidate wins the declared selection rule.

Write one concise decision record with evidence paths, hashes, cost/quota,
decisions, submission status, and the next useful action. Release the review lock.
Return an update only for a meaningful change, decision, failure, or unresolved
access/budget blocker. Do not claim a score, launch, approval, or final selection
without its evidence. Do not run indefinitely inside this iteration.
```

## Iteration order

| Order | Action | State change allowed |
| --- | --- | --- |
| 1 | Read stop flag, campaign version, lock ownership, and authorization | Acquire review lease; otherwise exit |
| 2 | Reconcile pending actions and remote receipts | Resolve `UNKNOWN`, settle reservations |
| 3 | Refresh live quota, rule snapshot when stale, and bill/expiry | Update resource observations |
| 4 | Inspect active jobs and deadlines | Continue, checkpoint, cancel, or mark complete/failure |
| 5 | Review newly complete experiments | Reject, advance, or request one specific missing artifact |
| 6 | Review frozen eligible releases | Approve release; persist intent; submit once |
| 7 | Reserve and dispatch next bounded job | Queue/launch only when resource admission passes |
| 8 | Retrieve scores and update incumbent when available | Record completed result and justified promotion |
| 9 | Persist decision and next action | Close iteration and release lock |

Recheck budget and quota under the lock immediately before an external mutation. Another manual account action can consume quota after observation; handle platform rejection without assuming the prior balance remains available.

## Locking, retries, and ownership

Use a single controller with an OS-backed process lock or equivalent transactional lease; a second scheduler on Windows and the rental must not both own the campaign. For a lease, include owner, expiry, heartbeat, and a monotonically increasing fencing token. Workers must reject stale dispatch tokens. Do not hold an unrenewed short lease while waiting on a long network call.

The reviewer should normally finish within 10 minutes, with a hard iteration bound shorter than 60 minutes. It dispatches workers; it does not wait synchronously for a training epoch, a full Kaggle run, or scoring completion. A launcher watchdog handles minute-scale deadlines even if this helper is unavailable.

The key for a job is the digest of the canonical run specification. The key for a release is the digest of its source, weights, dependency set, effective config, and input contract. Store notebook slug/version and output manifest separately. The key for a submission intent includes competition, exact notebook slug/version, output filename, and candidate digest. Use a unique human-readable tag in the description so a receipt can be recovered after an interrupted API call.

Network services do not guarantee exactly-once execution merely because the local database does. Reserve and persist intent first; on timeout, reconcile the provider/account before retrying. For unresolved submissions, use `SUBMISSION_UNKNOWN` and retain the slot reservation. For unresolved launches, use `LAUNCH_UNKNOWN` and retain the compute reservation. Never create an alternate slug or change the description just to evade duplicate detection.

Approvals expire after 60 minutes if unconsumed. A launched job continues under its own original deadline and reservation; approval expiry must not kill an already admitted job. Any changed source, weights, settings, data membership, deadline, or cost limit invalidates unconsumed approval and needs a new reviewer decision. Emergency shutdown does not require a future hourly approval.

## Decision vocabulary

| Decision | Meaning |
| --- | --- |
| `APPROVE_RUN` | All run admission checks pass; reserve resources and bind exact specification |
| `CONTINUE` | Running job remains within its approved limits and produces useful progress |
| `CHECKPOINT_STOP` | Verified job reached a bound, failed a criterion, or was superseded with a documented reason |
| `REJECT` | Candidate or proposal fails evidence/quality gates; state the precise reason |
| `APPROVE_SUBMISSION` | Frozen exact version is eligible and within submission allowance |
| `PENDING_SCORE` | Receipt exists; poll in later iterations without resubmitting |
| `PROMOTE` | Completed candidate wins the recorded incumbent-selection rule |
| `NO_CHANGE` | No useful state change; no new work is justified |
| `BLOCKED` | A specific missing capability, authority, or artifact prevents that action |

Operational approval and quality assessment are separate fields. An external reproduction experiment can be operationally approved while quality remains `unproven`. An observed high public score does not override invalid artifacts or known leakage.

## Activation on the actual agent host

1. Resolve the real repository, existing campaign store, provider instance, configured agent runner, Kaggle account, and credentials through their established configuration. Do not print secret values. Perform harmless status reads on each required service.
2. Inspect existing scheduled tasks and campaign processes. Reuse or replace the intended campaign monitor, with its state preserved, so exactly one schedule owns this campaign. Do not create a second hourly monitor beside a paused one and later resume both.
3. Implement or adapt a single-iteration reviewer entrypoint using the already installed agent runner. Its inputs are this prompt and the canonical campaign root; its outputs are decisions and state updates. Resolve the actual runner's invocation locally; no guessed model/CLI flags are supplied here.
4. Exercise one read-only iteration, then one mutation-enabled iteration using an already eligible bounded action. Verify lock behavior, stale-action reconciliation, and receipt persistence. This is operational setup, not a new model experiment.
5. Register that verified entrypoint with the host's existing scheduler for a 60-minute cadence. Use a durable scheduler rather than an interactive terminal sleep loop. Keep timestamps in UTC and display local time when helpful. A laptop that sleeps is not a reliable unattended controller.
6. Save schedule ID/name, host, exact entrypoint, prompt hash, campaign ID, enabled state, next run time, and a successful first-run receipt. Only then report that the helper is active. An intended schedule in YAML is insufficient.

If this is implemented as a ChatGPT Automation, creation requires successful read-only access to every service needed by its future runs. Do not create an automation with a promise that access will work later. If those capabilities are unavailable, leave the prompt ready for the existing local agent environment and report that activation is outstanding.

## Brief hourly report format

```text
Review time (UTC):
New evidence: [run IDs, concrete metric deltas, coverage/provenance label]
Compute: [actual spent, reserved, remaining; Kaggle quota and release reserve]
Decision: [APPROVE_RUN / CONTINUE / ...], [reason and artifact digest]
Kaggle: [exact notebook version, submission ID/status/score if returned]
Incumbent: [unchanged or promoted, and why]
Next action: [one concrete task, owner, bound]
Blocker: [only if present; what can still proceed]
```
