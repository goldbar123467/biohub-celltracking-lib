# Frozen learned-model submission

On 2026-09-07 the user explicitly requested a Kaggle submission after being told
that the training campaign exhausted its budget, whole-embryo evaluation was
incomplete, and detection caps failed the promotion criterion. This authorizes
an experimental submission; it does not convert those results into a validated
model. The training campaign and its hourly monitor remain stopped/paused.

## Candidate and limitations

Use the existing fold0 development-selected checkpoint at step 32,000 and its
already frozen threshold 0.3. This is the only fold with a completed threshold
selection record. Do not retune from the partial outer results or train again.
The model was fit on the `6bba` embryo, with internal development clips excluded
from gradient updates; `44b6` supplied the independent outer evaluation.

- Training source: `4587708564490ea582ba3bfdc28ea9b730ef2f52`.
- Original checkpoint SHA256:
  `e83396022fdb2452fbc875fcf134c2426fef464b4e3218e1d50c73c74f03beda`.
- Exported tensor-only weights SHA256:
  `2cdce85448f833c80fe0be6d5245ea860124ac508965bb7f4dac1d78c8eb2ad5`.
- Model: `PointDetector3D(base_channels=12)`.
- Inference: XY stride 4, tile size 64, overlap 16, NMS radius 3 micrometers,
  link distance 8 micrometers, maximum 2,000 nodes per frame, threshold 0.3.
- Both fits completed, at 63,564 and 62,577 updates. The budget ended after only
  61/71 paired classical/learned outer clips for `44b6`; the other direction was
  not evaluated. Detection caps occurred in 43/61 learned clips. There is no
  complete paired model-quality result and no all-data refit.

## Packaging and checks

`scripts/package_learned_kaggle.py` authenticates the original checkpoint against
the saved selection, exports only finite model tensors, and checks exact tensor
round-trip equality. The notebook verifies the model manifest, weights, embedded
source archive and offline wheel hashes before inference. It discovers the actual
test IDs, reads one frame at a time, and retains the existing detector/linker.
The streaming writer checks graph structure, bounds, exact dataset coverage and
CSV schema before atomically publishing `submission.csv`. The notebook saves
diagnostics including capped frames, runtime, environment and artifact hashes.

The integrated Vast suite passed 87 tests in 8.79 seconds. The initial invocation
failed one existing CUDA determinism test because a variable set outside the tmux
launcher was not inherited; invoking the job with `env CUBLAS_WORKSPACE_CONFIG`
fixed that invocation. The final suite includes exact exported-model prediction
parity, wrong-checkpoint rejection, corrupted-weight rejection, deterministic
notebook generation and compiled code cells. These tests do not establish Kaggle
execution or generalization. The Kaggle rehearsal and submission receipts are
retained separately in private generated reports.

From the project root, export on the configured Vast environment:

```bash
.venv/bin/python scripts/package_learned_kaggle.py export \
  --checkpoint reports/campaigns/longrun-20260906-02/fold0/checkpoints/step-000032000-002d9eec30.pt \
  --snapshot reports/submission-source-snapshot-20260907.json \
  --output-dir work/kaggle-point-model-20260907 \
  --dataset-id clarkkitchen/biohub-point-detector-20260907
```

Retrieve that private export to Windows and verify its hashes. Build and run:

```bash
python scripts/package_learned_kaggle.py build \
  --output-dir work/kaggle-learned-release-20260907 \
  --kernel-id clarkkitchen/biohub-frozen-learned-submission \
  --model-dir work/kaggle-point-model-20260907 \
  --dataset-id clarkkitchen/biohub-point-detector-20260907
kaggle datasets create -p work/kaggle-point-model-20260907
kaggle kernels push -p work/kaggle-learned-release-20260907 \
  --accelerator NvidiaTeslaT4 --timeout 43200
```

The model dataset and notebook are private. The inference alarm is nine hours;
the submission must satisfy Kaggle's current 12-hour notebook limit and internet
must remain disabled. Quota checked before this run was 29.98/30 GPU hours
remaining. This is a dated balance, not a reservation or permission to consume it.

Before scoring, verify the committed notebook completes, retrieve its CSV and
manifest, match the raw CSV SHA256, and independently validate discovered dataset
coverage and graph constraints. Submit the exact verified notebook version:

```bash
kaggle competitions submit biohub-cell-tracking-during-development \
  -k clarkkitchen/biohub-frozen-learned-submission -v <verified-version> \
  -f submission.csv -m "Frozen fold0 learned detector; incomplete outer validation"
```

Record the returned submission ID and processed status. Do not equate notebook
push, rehearsal completion, or submission acceptance with a completed score.
Public score and unavailable private performance must remain separate.

Current [rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules)
and [notebook requirements](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview)
were rechecked on 2026-09-07: five submissions per day, two final selections,
12-hour CPU/GPU limit, internet off, final deadline September 29. The authenticated
submission listing showed no submissions on September 7 before this attempt.
