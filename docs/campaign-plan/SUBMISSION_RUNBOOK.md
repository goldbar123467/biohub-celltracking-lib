# Kaggle release and submission runbook

## Live constraints and identity

Competition: `biohub-cell-tracking-during-development`.

The supplied September 7 report recorded five submissions per day, two final selections, a 12-hour CPU/GPU runtime limit, internet disabled, and a September 29 final deadline. A September 8 search of the official overview corroborated September 29, 2026 at 23:59 UTC; full rules content and authenticated account limits were not retrievable in this authoring session. Refresh all limits, reset semantics, external-data eligibility, and account quota on the actual Kaggle account before launch/submission. Store the observed values and time in `campaign.json`.

The existing frozen notebook is `clarkkitchen/biohub-frozen-learned-submission`, version 1, submission `56086171`. Retrieve its current processed state. Do not resubmit it merely because the supplied report said pending. Locate all other completed account submissions before choosing an incumbent; do not assume this particular learned model is the best.

## Release contract

A release directory is immutable after helper approval. It contains the notebook/source bundle, inference-only model weights, dataset version references, offline dependencies, effective settings, licensing/provenance record, source/config/weight hashes, local metrics with coverage labels, runtime estimate, and build manifest. Preserve old candidates and source tags. A final all-data fit must have its own release identity.

Notebook inputs must provide all imports and weights offline. Verify attached input versions and their mounted content hashes inside the notebook. Kaggle may resolve dataset attachments differently than local assumptions; content identity is mandatory. Discover actual hidden/test inputs at runtime, stream the volumes, and do not hardcode the four visible IDs or a presumed hidden clip count.

Use the existing hash-verifying export/build path where compatible. The report identifies `scripts/package_learned_kaggle.py`, `biohub_ct.submission.validator`, and `scripts/make_submission.py`; inspect actual source/help before invoking. Adapt the public reference deliberately instead of assuming the existing point-detector exporter understands its architecture.

Fail closed on missing weights, corrupt input, incomplete dataset coverage, invalid coordinates, or invalid graphs. Do not disguise a deployment failure by emitting arbitrary fallback nodes. A genuinely empty prediction for an input, if schema-permitted, is distinct from an exception being hidden by fabricated output. Save diagnostics and leave the prior incumbent intact.

## Rehearsal checks

1. Confirm exact source, model, support code, dependency hashes, and effective settings. Compile notebook code cells and run the focused export/inference compatibility checks.
2. Push a private, committed Kaggle notebook with internet off and an account-supported accelerator. Record returned version, metadata, and runtime start. Do not change its slug/version while release verification is in progress.
3. Wait by checking status in later helper iterations. A successful push is not a completed run. Inspect logs for retries, fallbacks, silent truncation, capped frames, and stage timing.
4. Retrieve the completed version's actual output and run manifest. Beware that ordinary CLI output/status commands target the latest run; freeze the slug until retrieval, or use a supported version-specific route and verify the embedded release ID. Never assume downloaded latest output belongs to the version about to be submitted.
5. Verify raw CSV SHA-256 against the notebook manifest. Independently check exact input-ID coverage, schema, unique node identifiers in the required scope, finite coordinates, discovered shape bounds, valid edge endpoints, time direction, permitted frame gaps, incoming/outgoing degrees, and division topology. Read the actual competition validator for exact constraints.
6. Require sufficient wall-time and memory headroom on representative workloads. Include setup/install time and CPU graph/CSV costs. Report capped frames and predicted counts even for a structurally valid release. A density problem can fail quality review while the CSV remains valid.

The historical visible CSV hash is `db2448f055047364f4a9229bd09b4c6215d0d02d8671630b1b13f57bcbd58a8d`. It identifies the old visible output, not a future hidden-test output. A code submission reruns inference on hidden inputs; matching the visible CSV hash proves rehearsal identity, not hidden-output equality.

## Three allowed submission classes

| Class | Sufficient evidence for operational approval |
| --- | --- |
| Public-reference reproduction | Verified external-artifact eligibility, pinned implementation, unchanged effective algorithm, complete offline rehearsal, valid output, resource fit; training overlap labeled honestly |
| Validated improvement | Above operational checks plus preregistered paired component/full-score comparison and completed available evaluation coverage |
| Exploratory candidate | Above operational checks plus a specific testable hypothesis, declared evidence limitation, material difference from prior entries, and available exploration allowance |

A failed CSV or incomplete rehearsal cannot be excused as exploratory. The helper can submit an honest exploratory candidate without repeatedly asking Clark, within the allowance below; its approval record must explain what result would change the next decision. Do not call it the incumbent merely because it is newer.

## Submission allocation

The hourly cadence is for review. Default to at most three submissions per platform-defined day and reserve two slots when the live daily limit is five. Within those three, default to at most two exploratory entries. Count the first reference reproduction against the exploratory allowance until it has our own completed result. Recompute allowance from actual account history, including manual submissions and unresolved intents; use Kaggle's returned remaining allowance where available rather than guessing how failed attempts count.

Reserve slots are for a justified repair, a clearly superior frozen candidate, or deadline recovery. The helper may release a reserve for one of those reasons and record why. These are campaign policies, not additional Kaggle restrictions. Never exceed the actual limit. Submit no duplicate candidate hash without a documented operational need such as a failed prior execution that has been reconciled and repaired.

A useful sequence is the unchanged reference, one causally interpretable improvement, and one justified alternative. Do not sweep threshold values on the public leaderboard. Once a public score influenced selection, record that information path. Preserve a candidate chosen using broader validation or different errors for a possible second final selection.

## Verified CLI forms and local adaptation

Kaggle's official CLI documentation was consulted for these forms. Check the installed CLI's help before use; these commands are instructions for the connected project host, and were not executed here. Use structured subprocess argument arrays in automation, not string interpolation. Replace uppercase placeholders with resolved values and reject unresolved placeholders before execution.

```bash
kaggle competitions submission 56086171
kaggle competitions submissions biohub-cell-tracking-during-development
kaggle kernels pull redoctopusk/biohub-942tta/VERIFIED_VERSION -p PINNED_SOURCE_DIR -m
kaggle kernels push -p RELEASE_DIR --accelerator VERIFIED_ACCELERATOR --timeout VERIFIED_SECONDS
kaggle kernels status OWNER/RELEASE_SLUG
kaggle kernels output OWNER/RELEASE_SLUG -p VERIFIED_OUTPUT_DIR
kaggle competitions submit biohub-cell-tracking-during-development -k OWNER/RELEASE_SLUG -v VERIFIED_VERSION -f submission.csv -m UNIQUE_RELEASE_TAG
kaggle competitions submission RETURNED_SUBMISSION_ID
```

The old public-reference version number is not supplied; resolve it rather than copying `1` from our different frozen notebook. Dataset and model version pins also require actual discovery. A supported command form is not evidence that this account has access to the artifact or accelerator.

## Commit, submit, reconcile

Under the controller lock, bind approval to the exact release digest and notebook version; confirm it is still eligible and quota remains. Persist a unique submission intent and slot reservation before the API call. Record the numeric submission ID and raw status receipt immediately after success. If the call times out or the receipt is missing, mark `SUBMISSION_UNKNOWN` and inspect account history for the unique release tag/version/time. Never issue another submission until that ambiguity is resolved.

Statuses are separate: build complete, notebook queued/running, notebook complete, output validated, approved, submission accepted, scoring pending, scored, and failed. Record the public score only after Kaggle returns it. Leave unavailable private score fields null. On a lower score, preserve the incumbent and investigate coverage, generalization, numerical changes, and component tradeoffs before investing in another fit.

## Final selection

Refresh the actual number of final selections and the deadline. If two are permitted, normally retain the strongest completed eligible candidate and one meaningful alternative supported by complementary local evidence or errors. Two forks of the same model with indistinguishable output are not meaningful diversity. A claimed July 0.826, the public notebook's reported 0.946, and our partial 0.221 are not comparable entries in one selection table without their identities and evaluation populations.

Keep separate pointers for best completed public score, best available validation evidence, and selected final entries. They can disagree. The helper may choose final entries within Clark's submission authorization, but it must record actual Kaggle selection confirmation. Do not invent a CLI final-selection flag; use the installed supported interface. Keep exact versions and hashes restorable through the deadline.

Freeze the last operational candidate with enough time for a conservative full execution plus recovery. If an upgrade is still pending near the deadline, preserve a completed valid selection. Stop scheduled compute/submissions at campaign expiry or the final deadline; leave a final artifact and decision record.

Sources: [official competition overview](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview), [rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules), [Kaggle kernel CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [Kaggle competition CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md). Historical account observations come from the supplied September 7 reports, not a fresh account query.
