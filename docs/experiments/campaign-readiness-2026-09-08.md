# Campaign readiness refresh

Observed 2026-09-08 at 11:15-11:17 UTC. This records local preparation and current
account observations. It does not admit another experiment or declare the
campaign complete.

## Durable handoff

The reviewer now writes the required controller-root `handoff.md` atomically
under its existing review lock, after recording the decision and exporting
current SQLite state. It includes distinct incumbents, active jobs/deadlines,
resource ledger values and observation times, unresolved intents, evidence,
current approvals, exact next-task inputs, and blockers. It explicitly grants no
authority. Ledger snapshots are not presented as additive account balances.

The independently provisioned root suite passed 41 focused tests with six
explicit optional-dependency skips in 2.88 seconds. The final presentation change
then passed all 14 reviewer tests in 1.53 seconds. Ruff and the diff check passed.
Regressions cover current state/receipt binding, replacing stale summaries,
terminal-job reporting, controller-root placement, database-path conflict, and
visible write failure.

An actual read-only review at 11:24 UTC generated the handoff from canonical state
version 276, returned `NO_CHANGE`, and kept notifications quiet. Its local
receipt is under
`reports/campaigns/campaign-20260908-01/reviews/review-20260908T112438145544Z`.
The final renderer repeated that result at 11:26 UTC, state version 279, review
`review-20260908T112610785525Z`. Root independently rehashed its decision, state
snapshot, handoff, and all seven configured next-task inputs. The verification
receipt is `reports/campaigns/campaign-20260908-01/handoff-root-verification-20260908.json`.
No active job or unresolved intent remained. The controller-root `campaign.json`
is a derived export of SQLite, and the original initial export is preserved.

## Corrected telemetry candidate

The candidate embeds the reviewed `csv.DictWriter` timing fix. The orchestrator
reviewed the agent's preparation script and independently rebuilt a third package
from the pinned source and hash-verified dependency archives. All four files
match the agent's two builds byte for byte. All 12 public algorithm cells remain
unchanged; all 18 packaged cells compile. Frozen R4 retains its original identity
and failed telemetry result.

| Identity | SHA-256 |
| --- | --- |
| Release digest | `9dc0a201ea15c25a4de7481badaefe52a1ad5b26b81a5dd168619f2d72dca77d` |
| Notebook | `f0ba48c3d846f51ccc6ad534e590dfd0f5c2313568afc8b5da597f1f2a4e0e7b` |
| Package manifest | `18e7f5e40ef501f38cc49d7c643ec40a140c8a10779604ec601839fe8768d006` |
| Artifact lock | `e5e934c9a3ab7b8b71158c53ddd509adbd79f26eca9f4d4e5b7abd3cf4e21692` |

An executed synthetic smoke extracted the runtime from the actual packaged
notebook. Target writes produced `MEASURED` serialization records, identical CSV
bytes to the unwrapped control, no records for another filename, and restored the
global writer after cleanup. Timing covers writer calls, excludes graph work
between calls and final close/fsync, and includes lazy iterable production inside
`writerows`. This is local packaging and writer evidence only. Full notebook
execution, model quality, and corrected whole-run telemetry remain unverified.

Ignored local evidence: `work/e0-dictwriter-candidate-review-receipt.json`,
`work/e0-dictwriter-candidate-root-review.json`, and the two directories
`work/e0-reference/package-dictwriter-candidate-{a,b}`. The root independently
verified the agent receipt hash
`cb090169a6c8806f760e977c9cda8c0b52e520625ba29da747b5688138d3d239`.
Status: **NOT_LAUNCHED / NOT_ADMITTED / NO_QUALITY_CLAIM**.

## Live competition and account observations

The authenticated competition pages were read in the browser on September 8.
The [code requirements](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/code-requirements)
require notebook submissions, at most 12 CPU or GPU runtime hours, internet
disabled, and output named `submission.csv`. Public external data and pretrained
models are permitted, subject to the full rules. The same overview lists the
entry and merger deadlines as September 22 and the final deadline as September
29, 2026, all at 23:59 UTC.

The [competition rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)
list five submissions per day and at most two final selections. Their external
data/tool provisions include access and licensing conditions; these do not
establish the eligibility of an individual weight file or its training data.
Rules were already accepted on the account; this check made no account changes.

The authenticated installed CLI returned five submissions currently allowed and
GPU quota of 3.17 hours used, 26.83 remaining, and 30.00 total at 11:15:53 UTC.
Its raw quota reset field is `2026-09-12T00:00:00`, without an explicit timezone.
The response does not establish the daily submission-reset timezone or failed
submission accounting. `numTotal=2` is retained as a raw account field, not
interpreted as today's usage or the final-selection limit. Before any admission,
refresh account allowance and history; do not infer reset semantics from these
fields. The ignored receipt is
`reports/campaigns/campaign-20260908-01/account-limits-refresh-20260908.json`.

The campaign ledger retains 26.40 GPU hours as protected reserve and no R4
reservation. Its currently unprotected remainder is 0.43 hours, below R4's
observed 1.58-hour debit and the prior 1.98-hour worst-case reservation. The
prepared corrected candidate therefore has no current full-rehearsal admission.
Changing the reserve is a resource-policy decision, not a consequence of passing
local tests. No new provider job, competition submission, final selection, or
scheduler activation occurred in this refresh.

## Requirement audit boundary

The original E1 plan requires inspection of matched/unmatched bright regions and
overlays or orthogonal slices. It does not require a human or domain expert. The
retained eight-sheet root AI inspection covers four fixed clips at frames 0 and
50 with XY/XZ/YZ views, fixed coordinates/intensity geometry, hashes, and per-sheet
observations. See [E1 results](e1-pilot-results.md). The separate visual-review
artifact preserves sparse-label and baseline-setting limitations; its lack of
human review is provenance, not an additional approval gate. The earlier
experiment-specific preregistration's stricter wording does not change that
original requirement or retroactively modify its immutable evaluator output.

E1's 14.7087% reduction failed the preregistered 20% advancement threshold. Further
training or temporal sweeps are therefore conditional, not required merely to
close a rejected branch. Clean-holdout claims still require appropriate training
membership evidence; none is made. E2-E5 remain subject to their specified
scientific admission gates. A completed implementation is distinct from an
admitted experiment, successful rehearsal, or eligible final submission.
