# Controller failure and submission accounting verification

This follow-up closes two implementation gaps found while auditing the supplied state and submission contracts. It does not activate the helper or authorize more compute.

## Process death and canonical state

`CampaignStore.export_json` now rejects destinations equal to the authoritative database, its SQLite `-wal`, `-shm` or `-journal` sidecar, or the default review-lock file. Previously, an accidentally selected export destination could replace one of those files with JSON. The path is resolved before comparison, and rejection occurs before any export write.

The new crash-recovery tests spawn a real child process against an isolated temporary campaign store. A pipe identifies the exact boundary before the parent kills that child:

- After the production ledger/event writes but before their transaction commits: the earlier committed ledger survives; the interrupted ledger, event and state-version increment all roll back.
- After a complete temporary export is flushed and synced, before atomic replacement: the prior JSON remains byte-identical and valid.
- Immediately after atomic replacement: the complete new JSON is visible and its state version agrees with its event tail.

Every case independently reopens SQLite, checks `PRAGMA integrity_check`, reacquires the OS review lock with a strictly larger fencing token, and confirms contiguous committed event versions. A subsequent normal export succeeds even if process death left an unreferenced temporary file.

This verifies abrupt process death on the tested Windows and WSL filesystems. It is not a physical power-loss or storage-controller durability experiment. SQLite remains authoritative; an old but internally consistent JSON export cannot grant authorization.

## Daily submission policy

The first unscored public-reference reproduction now consumes exploratory allowance. It stops consuming that classification only when its own account result is COMPLETE with a numeric score; a COMPLETE row with blank scores remains unscored. A new reference attempt still needs an available exploratory slot.

The reviewer now counts the union of actual account history and unobserved submission intents, rather than counting only helper-owned intents. A confirmed intent and its numeric account row count once. Unresolved intents retain their slots; proven-ABSENT intents do not. Account acceptance time determines the day for observed entries, including acceptance across midnight. The installed Kaggle CSV's unzoned API dates are treated as UTC.

Manual account entries with no established submission class consume daily allowance and conservatively consume exploratory allowance. They are reported as unclassified in the derived accounting result; the code does not infer a class from their free-text descriptions. Duplicate numeric history rows or malformed timestamps reject admission. Live platform allowance and protected final slots are still checked separately.

Regression cases cover reference-plus-manual exhaustion, blank versus zero-valued scores, history/intent deduplication, retained UNKNOWN reservations, proven absence, and midnight acceptance. No submission was made during these tests.

## Protected submission reserve

A `final_release=true` candidate now requires `reserve_release_justification`
with one permitted category (`justified_repair`, `superior_frozen_candidate`, or
`deadline_recovery`), a nonempty explanation, and an evidence file included in
`verified_files`. The frozen candidate document preserves that decision, and
admission rehashes the evidence before using protected slots. A regression
checks missing evidence, a rejected change after freeze, and an altered evidence
file. This enforces a recorded, immutable decision; the reviewer still has to
assess whether the evidence supports the stated reason.

Such a justified reserve decision can exceed the default three-entry allocation
and its two exploratory entries. It always remains bounded by the actual
platform allowance. The slot function's `final_release` exception is reached by
the production reviewer only after the release-artifact justification check.

## Executed checks

On native Windows, 17 state/crash tests passed and 20 accounting/admission/reviewer tests passed before the reserve-justification change. On WSL Linux, the updated combined state, crash, budget, accounting, admission and reviewer suite passed **44 tests in 3.16 seconds**. The newly added test files pass Ruff. These are controller CPU tests against isolated temporary files; they did not use the Vast GPUs or alter the live campaign ledger.
