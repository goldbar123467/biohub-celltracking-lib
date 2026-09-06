# First learned-detector campaign

Run: `longrun-20260905-01`. Platform: existing Vast RTX 4070 SUPER.
This is an experimental model-quality campaign, not a released competition model.

The launcher freezes a committed source snapshot, fixes the CUDA determinism
environment, and starts a logged tmux job. A shared 27,000-second wall-clock
allocation covers two sequential 9,000-second fits, internal threshold selection,
paired complete outer-embryo evaluations, and a conditional all-data refit of at
most 3,600 seconds. A separate GNU timeout ends the process after 27,120 seconds,
with a further 30-second termination grace. Smoke tests fit inside the remaining
part of the authorized eight-GPU-hour ceiling. The allocation is a maximum, not
a promise that every stage will fit.

The internal development clips are every tenth sorted training-embryo clip.
Periodic selection uses negative masked development loss on eight fixed patch
batches. Final threshold selection tests `0.3, 0.5, 0.7` on the first two internal
development clips, reusing inference across thresholds. Both fold models finish
fitting before outer evaluation. Outer labels never feed back into either fit.
The selected checkpoint and threshold are frozen before that fold is scored.

Refitting requires complete paired outer evaluation, improved official score in
both directions, and no learned fallback or capped frames. Its threshold is the
median of the two internal-development-selected thresholds, frozen before refit.
The runtime manifest records the actual data/model/optimizer/inference settings;
the research note contains proposed alternatives, not additional experiments.

## Verified preflight

- All 24,886 expected files passed the complete SHA256 receipt audit with zero
  mismatches or extra files. A Kaggle CLI bookkeeping marker was moved to ignored
  scratch space before the final audit. Receipts establish local integrity,
  not organizer-signed provenance.
- The integrated server suite passed 74 tests. This includes exact CPU and CUDA
  AMP interrupted/resumed model, optimizer, scheduler, scaler and RNG comparisons.
- A real-data 20-step smoke resumed to step 30 and loaded the selected checkpoint
  for inference. A `64x256x256` uint16 frame became `64x64x64` probabilities in
  0.110 seconds; combined smoke peak CUDA allocation was 666,428,928 bytes.
  One-frame speed is not a complete hidden-test runtime measurement.
- The learned full-clip path also completed 100 frames, graph linking, CSV
  validation/round-trip and pinned official scoring in a development-clip smoke.
  Inference and CSV processing took 12.26 seconds. This used a 30-step checkpoint
  and an internal development clip, so it establishes execution, not transfer.

## Launch and monitoring

On Vast, after synchronizing a reviewed commit with a clean working tree:

```bash
bash scripts/start-training-campaign.sh longrun-20260905-01
```

The source snapshot remains under `work/campaign-sources/<run-id>`. Logs and exit
codes are under `reports/jobs/<run-id>`. The machine-readable campaign manifest,
stage, per-fold training status, JSONL metrics, evaluation rows and checkpoints
are under `reports/campaigns/<run-id>`. Five-minute checkpoint intervals are an
upper target, also triggered at every 1,000 steps and validation improvements.

From Windows, retrieve status plus hash-verified latest/previous/best checkpoints:

```powershell
& .\scripts\check-campaign.ps1 -RunId longrun-20260905-01
```

Backups are private ignored files under `reports/campaign-backups/<run-id>`.
Checkpoint payloads are immutable; the downloader checks SHA256 before accepting
them and preserves the matching pointer snapshot. Do not commit data or weights
to the public source repository.

The hourly Codex check depends on the local computer and desktop app remaining
available. Tmux training survives an SSH disconnect; it does not survive server
loss. A missing exit-code file is not evidence of success. Inspect process state,
step advancement, checkpoint timestamps, GPU state and free disk together.

The campaign records `budget_exhausted` if evaluation cannot finish within its
deadline. Partial results cannot qualify the model for promotion. Do not extend
the accepted budget. Recovery is an explicit resume
from a hash-verified checkpoint with the same source/config/split/dependencies;
use `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python on this CUDA stack.

After successful validation, learned offline packaging, private checkpoint
attachment, target-environment inference and CSV validation still precede any
scored submission. The existing classical notebook remains the tested fallback.

## Authorized repair and restart monitoring, 2026-09-06

The user authorized the hourly monitor to diagnose and repair failed runs,
restart them, verify progress, and then leave training running on the server.
This supersedes the previous instruction to stop after every failure.

Find the current attempt in the server's `reports/campaign-active.json`. Before
a restart, confirm the current job has exited and back up its checkpoints and
failure logs with verified hashes. Diagnose the actual error, make a bounded
repair in the working repository, run relevant regression tests and a real-data
recovery check, then synchronize a reviewed commit. Preserve the previous source
snapshot and attempt records. Do not weaken integrity or finite-value checks to
hide a failure, change validation splits, or reset the compute budget.

Restart from the failed attempt using a new job ID:

```bash
bash scripts/start-training-campaign.sh longrun-20260906-02 longrun-20260905-01 900
```

The launcher refuses a parent without an exit record. The recovery plan deducts
the parent's wall runtime, all earlier charged attempts, and a conservative
600-second repair/testing reserve from the shared 27,000-second allocation.
The optional third argument charges a larger repair/testing duration; the first
repair uses 900 seconds to cover reproduction and the extended real-data check.
If diagnostics take longer than the reserve, charge the excess before launch.
Completed fits and verified latest/previous/best checkpoints are retained.
Explicit checkpoint migration permits source identity changes while preserving
data identity, split, numerical configuration and dependency checks. The new
checkpoint records its parent identity. Optimizer/scheduler/scaler/RNG state
continues from the checkpoint; failed work after that checkpoint is replayed.

If a repair succeeds, update the active-attempt pointer, verify advancing steps
and a newly saved checkpoint, notify the user of the recovery, and keep the
hourly monitor active. Continue autonomously while the remaining authorized
budget supports meaningful work. If the budget is exhausted or the problem
cannot be safely repaired, preserve artifacts, report the concrete blocker, and
pause the monitor. On successful campaign completion, back up the results and
report actual validation and remaining release gates, then pause the monitor.

The first failure reproduced after 2,398 completed updates. The FP16 scaler had
grown from 65,536 to 131,072, and nonfinite gradients caused clipping to abort
before GradScaler could skip the invalid update and lower its scale. The repair
uses GradScaler's recorded overflow to skip that optimizer update, lowers the
scale, restores RNG, and retries the same batch. Scheduler and data step advance
only after a finite update. Persistent failures stop after eight retries;
non-AMP nonfinite gradients and nonfinite losses still fail immediately.
See the [PyTorch 2.9 AMP contract](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html).

Repair verification: the original code reproduced the exact failure at step
2,398. The repaired same-stack run continued from checkpoint 2,000 through
7,000, recovered three AMP overflows, and completed a later scale-growth cycle.
The integrated suite passed 84 tests, including CPU/CUDA exact overflow-update
semantics, checkpoint resume, source migration, and cumulative restart budgets.
These are numerical/recovery checks, not held-out model-quality results.
