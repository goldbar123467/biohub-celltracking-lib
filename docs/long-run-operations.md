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
deadline. Partial results cannot qualify the model for promotion. Do not silently
restart a failed job or extend the accepted budget. Recovery is an explicit resume
from a hash-verified checkpoint with the same source/config/split/dependencies;
use `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python on this CUDA stack.

After successful validation, learned offline packaging, private checkpoint
attachment, target-environment inference and CSV validation still precede any
scored submission. The existing classical notebook remains the tested fallback.
