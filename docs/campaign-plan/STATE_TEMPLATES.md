# Machine-readable campaign state and review records

## Implementation contract

The following are templates for the agents to implement in the real project. No corresponding controller software or live state was created by writing this file. Use one transactional state store on the controller, with JSON exports for review. SQLite with transactions on a local durable disk is one reasonable implementation; do not put independent SQLite writers on loosely synchronized filesystem copies.

Keep a canonical `campaign.json`, append-only decision/event records, per-run immutable specifications and results, and a submission registry. Each external mutation needs a persisted intent and recoverable unique identifier. Validate state transitions; never treat a log sentence saying approved as a machine authorization.

All times use ISO 8601 UTC. All monetary amounts carry currency and billing unit. JSON `null` means unknown; it never means zero, unlimited, successful, or approved. Unresolved required fields block the dependent action only. Do not persist credentials in these files.

## Initial campaign template

Create a fresh campaign ID rather than resetting the spent ledger of `longrun-20260906-02`. The historical checkpoint can be imported with its original identity. Fill live fields from evidence before moving to `ACTIVE`.

```json
{
  "schema_version": 1,
  "campaign_id": null,
  "state_version": 1,
  "status": "INVENTORY_REQUIRED",
  "competition": "biohub-cell-tracking-during-development",
  "controller": {
    "host": null,
    "repo_root": null,
    "campaign_root": null,
    "review_runner_argv": null,
    "schedule_id": null,
    "schedule_enabled": false,
    "cadence_minutes": 60,
    "first_successful_review_receipt": null
  },
  "authorization": {
    "routine_runs_and_submissions": true,
    "source": "Clark's request for GPU campaign documents and hourly run/submission approval",
    "budget_evidence_path": null,
    "new_rental_purchase_authorized": false,
    "rental_extension_authorized": false,
    "expires_at": null
  },
  "rules": {
    "verified_at": null,
    "evidence_path": null,
    "deadline_utc": null,
    "daily_submission_limit": null,
    "submission_reset_at": null,
    "final_selection_limit": null,
    "notebook_runtime_limit_seconds": null,
    "inference_internet_enabled": false,
    "external_artifact_requirements": null
  },
  "resources": {
    "rental": {
      "provider": null,
      "instance_id": null,
      "gpu_model": null,
      "gpu_count": null,
      "gpu_vram_gib": null,
      "billing_unit": null,
      "currency": null,
      "unit_rate": null,
      "authorized_cost_total": null,
      "confirmed_spend": null,
      "unreconciled_spend_estimate": null,
      "outstanding_reservations": [],
      "prepaid_hours_remaining": null,
      "paid_until": null,
      "max_additional_billable_hours": null,
      "observation_time": null
    },
    "kaggle": {
      "quota_hours_remaining": null,
      "quota_reservations": [],
      "quota_reset_at": null,
      "quota_hours_per_wall_hour": null,
      "release_reserve_hours": null,
      "available_accelerators": [],
      "observation_time": null
    }
  },
  "policy": {
    "max_rented_gpu_jobs": 1,
    "max_kaggle_gpu_notebooks": 1,
    "default_submissions_per_day": 3,
    "default_exploratory_submissions_per_day": 2,
    "reserve_daily_slots_when_limit_is_five": 2,
    "release_attempts_reserved": 2,
    "approval_valid_minutes": 60,
    "review_iteration_limit_minutes": 10,
    "heartbeat_target_seconds": 60
  },
  "historical_submission_to_reconcile": {
    "id": 56086171,
    "notebook": "clarkkitchen/biohub-frozen-learned-submission",
    "version": 1,
    "last_reported_status": "PENDING",
    "accepted_at": "2026-09-07T23:37:06Z",
    "last_status_observation_at": null,
    "live_status": null,
    "public_score": null
  },
  "incumbents": {
    "best_completed_public": null,
    "best_validation_evidence": null,
    "final_selected_submission_ids": []
  },
  "stop_requested": false,
  "next_action": "Reconcile existing provider allocation, account status, and project artifacts"
}
```

The 23:37:06 timestamp is the supplied acceptance time. Recover a precise status-observation time from original receipts when possible. Store policy changes with prior value, new value, rationale, reviewer, and timestamp; never change budget authorization in the policy section.

## Immutable run proposal

```yaml
schema_version: 1
run_id: null
experiment_id: E0
state: PROPOSED
hypothesis: null
control_run_id: null
changed_factors: []
work_kind: inference
source:
  git_commit: null
  source_bundle_sha256: null
  dependency_manifest_sha256: null
  effective_config_sha256: null
model:
  weight_sha256: null
  training_membership: unknown
data:
  input_manifest_sha256: null
  split_manifest_sha256: null
  training_ids: []
  development_ids: []
  evaluation_ids: []
  evidence_class: overlap_unknown
execution:
  host: null
  provider_instance_id: null
  working_directory: null
  argv: null
  nonsecret_environment: {}
  gpu_devices: []
  random_seed: null
  max_wall_seconds: null
  max_steps_or_clips: null
  max_incremental_cost: null
  max_quota_hours: null
  deadline_utc: null
  checkpoint_interval_seconds: null
stop_rules: []
success_rules: []
expected_artifacts: []
reviewer_decision_id: null
run_spec_sha256: null
```

Hash the normalized immutable fields using canonical UTF-8 JSON, sorted keys, deterministic separators, finite numeric values, and explicit nulls. Exclude the digest itself and mutable execution/status fields from its preimage. Define this field set once in the controller. Treat paths as argv values; never construct a shell command by interpolating model-produced text. Reject unresolved paths/placeholders and mismatched source/config hashes at launch.

## Worker heartbeat and completion

```json
{
  "run_id": null,
  "run_spec_sha256": null,
  "fencing_token": null,
  "host": null,
  "provider_job_id": null,
  "pid": null,
  "process_start_time": null,
  "observed_at": null,
  "stage": null,
  "completed_units": 0,
  "total_units": null,
  "last_unit_duration_seconds": null,
  "elapsed_wall_seconds": null,
  "gpu_memory_peak_bytes": null,
  "rss_peak_bytes": null,
  "last_checkpoint_path": null,
  "last_checkpoint_sha256": null,
  "checkpoint_persisted_and_verified": false,
  "error": null
}
```

The completion manifest additionally records actual exit status, intended and actual clip IDs, metrics/artifact paths and SHA-256, stage timing, actual/bounded cost, fallback counts, and exact resume history. Write `COMPLETE` only after expected artifacts pass their checks. Incomplete evaluation is `PARTIAL`, not a lower-quality synonym for complete.

## Metrics record

Use per-clip records and a separate aggregate file. Required per-clip fields:

```text
run_id, clip_id, embryo, frame_ids, evidence_class, scorer_sha256,
predicted_nodes, estimated_nodes, annotated_nodes, matched_annotated_nodes,
localization_um_quantiles, pre_cap_candidates, capped_frames, total_frames,
edge_tp, edge_fp, edge_fn, raw_edge_jaccard, adjusted_edge_jaccard,
division_tp, division_fp, division_fn,
predicted_forks_after_solver, predicted_forks_after_postprocess,
runtime_seconds_by_stage, peak_ram_bytes, peak_vram_bytes,
fallback_count, graph_sha256, evaluation_complete
```

Aggregates include the exact planned and evaluated ID lists, missing IDs, pooled counts, official score components, per-direction results, and which metrics are per-clip means. Assertions must recompute aggregates from raw records and verify the planned population. Preserve undefined metrics as null plus a reason; never emit invalid JSON NaN or silently map it to a favorable value.

## Approval and submission records

```yaml
decision_id: null
reviewed_at: null
reviewer: hourly_helper
subject_id: null
subject_sha256: null
decision: null
operational_approval: false
quality_class: unproven
evidence_paths: []
reason: null
budget_reservation_id: null
submission_slot_reservation_id: null
valid_until: null
consumed_at: null
superseded_by: null
```

```yaml
candidate_id: null
candidate_sha256: null
competition: biohub-cell-tracking-during-development
submission_class: reproduction
notebook_slug: null
notebook_version: null
notebook_complete_receipt: null
rehearsal_manifest_sha256: null
visible_csv_sha256: null
output_filename: submission.csv
eligibility_evidence_path: null
approval_decision_id: null
intent_id: null
unique_description_tag: null
intent_persisted_at: null
submitted_at: null
submission_id: null
status: NOT_SUBMITTED
last_status_checked_at: null
public_score: null
private_score: null
score_receipt_path: null
selected_for_final: false
selection_receipt_path: null
```

Every successful mutation appends an event with the state version before/after, request identity, exact response/receipt path, and timestamp. Redact secrets from raw provider receipts. Keep `SUBMISSION_UNKNOWN` until a remote receipt or confirmed absence resolves it; a local retry counter cannot establish absence.

## State transitions

Jobs: `PROPOSED → APPROVED → LAUNCH_INTENT → RUNNING → COMPLETE/PARTIAL/FAILED/STOPPED`. An ambiguous launch moves to `LAUNCH_UNKNOWN` and must be reconciled before another launch. Completed experimental outputs move through review before any scale-up or release.

Releases: `DRAFT → FROZEN → REHEARSAL_RUNNING → REHEARSAL_COMPLETE → OUTPUT_VALIDATED → APPROVED → SUBMIT_INTENT → ACCEPTED → SCORED/FAILED`. `SUBMISSION_UNKNOWN` branches from `SUBMIT_INTENT`. A changed frozen artifact becomes a new candidate; it cannot retain old approval. Only a scored eligible candidate can replace the best-completed-public pointer. Final selection is an additional recorded action.

## Durable handoff

At each review, update a short `handoff.md`: current incumbent identities, active job and deadline, settled and reserved resources, new evidence, unresolved intents, next task and exact input artifacts, and outstanding blockers. This is a human/agent summary of the state store, not an alternative source of authority. A newly started helper must reconstruct decisions from records rather than trusting chat memory.
