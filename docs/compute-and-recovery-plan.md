# Compute and recovery plan

Prepared 2026-09-05. Quotas and accelerator availability are snapshots, not guarantees.

## Verified resources

The signed-in Kaggle notebook editor showed **00:00 used / 30 hours** before the readiness runs, with GPU T4 x2, P100 and TPU v5e-8 choices. There were no active events before setup. We selected T4 x2. Count the budget in the quota units Kaggle actually displays; do not assume this means 60 freely interchangeable GPU-hours. Refresh the available balance before each batch and check the reset date in the account UI.

Vast has an RTX 4070 SUPER with about 12 GB VRAM, a 9.6-CPU quota, about 85.45 GiB container memory and a 130 GiB disk. The full dataset is still downloading under Supervisor. Kaggle mounts the competition data directly, so its runs need not wait for the Vast copy or download another 87 GB into `/kaggle/working`.

The frozen file inventory contains **199 training samples from two embryos**: 71 for `44b6`, 128 for `6bba`. The validation design is therefore two directions: fit `44b6`, evaluate `6bba`; then fit `6bba`, evaluate `44b6`. Keep all fields of view and timepoints from an embryo together. Two held-out embryos provide limited evidence about broader biological variation; report both directions and metric components, not just a pooled number.

## Work allocation

| Work | Location | Reason |
|---|---|---|
| Code editing, source research, manifests and experiment ledger | Windows project | Durable project record and fast iteration |
| Data audit, metric checks, preprocessing experiments, short GPU debugging | Vast | Persistent shell state and interactive debugging |
| Independent validation runs, detector training, bounded ablations | Kaggle GPU notebooks | Use the account's unused quota and directly mounted data |
| Dependency-wheel preparation | Kaggle CPU notebook | Avoid consuming GPU time on package downloads |
| Final inference rehearsal and submission packaging | Kaggle GPU notebook, internet off | Exercise the actual target environment |

First 30-hour allocation, provisional ceilings rather than launched jobs:

| GPU quota budget | Purpose |
|---:|---|
| 1 hour | Readiness checks and representative throughput/memory benchmarks |
| 12 hours | Two embryo-held-out baseline directions, up to 6 hours each |
| 6 hours | A small number of controlled changes selected from baseline errors |
| 5 hours | Refit the selected configuration on all training data |
| 6 hours | Offline inference rehearsal, packaging failures and reserve |

Do not spend the whole baseline budget until a small representative training run measures steps/second, peak allocated VRAM and data-loader time. Reallocate when evidence warrants it. If projected hidden-test inference needs more of the quota, shrink ablations first. The competition deadline is 2026-09-29 23:59 UTC; no recurring launches or account upgrades have been scheduled.

Use one notebook per reproducible experiment. For T4 x2, each GPU has separate memory. Two cards do not make one 32 GB allocation. Initially benchmark one device and consider independent work on the second; use distributed data parallel only if measured end-to-end speed justifies the implementation and synchronization overhead. Do not attempt training distributed across Vast and Kaggle over the internet.

## Tmux operation on Vast

Tmux 3.4 was already installed. Project configuration uses its own server socket (`-L biohub`) so it does not alter other sessions.

The Windows `scripts/connect.ps1` now attaches or creates session `biohub`. Its windows are `work`, `download` (Supervisor log), and `gpu` (nvidia-smi). The environment activates in the work window.

```bash
# Reattach after an SSH disconnect.
bash /workspace/biohub-cell-tracking/scripts/tmux-session.sh

# Launch future commands in their own detached, logged window.
bash /workspace/biohub-cell-tracking/scripts/run-job.sh my-unique-run python path/to/train.py --config path/to/config

# View sessions and job evidence.
tmux -L biohub list-windows -t biohub
cat /workspace/biohub-cell-tracking/reports/jobs/my-unique-run/exit-code.txt
```

Press **Ctrl+B, then D** to detach. **Ctrl+B, then W** selects a window. Reusing a run ID is rejected so prior logs cannot be overwritten silently. Each job has output, start/finish timestamps and an exit code. Dead panes remain available for inspection. A missing exit-code file means unfinished or interrupted, not success.

Verified: a ten-second job survived closure of its launching SSH connection and exited 0; a deliberately failing job retained exit code 7. These tests establish disconnect survival and failure recording, not immunity to crashes.

## Failure and checkpoint policy

| Failure | Protection or required response |
|---|---|
| SSH disconnect or closing the local terminal | Tmux keeps interactive jobs alive; Supervisor keeps the download alive |
| Python exception or CUDA out-of-memory | Record failure; inspect logs and fix the cause. Do not blindly retry training |
| Server reboot or container stop/start | Processes stop; relaunch explicitly from a valid checkpoint |
| Instance recycle/destruction | Current Vast filesystem is lost; recover from external backups |
| Kaggle timeout, preemption, or worker failure | Use bounded runs and checkpoints saved as version outputs; an uncommitted interactive file is not a durable backup |

Before a substantial training run, implement and test checkpoints containing model, optimizer, scheduler, scaler, RNG states, global step/epoch, split identity, configuration and dependency/source versions. Save at least every ten minutes and at epoch boundaries with atomic replacement, keeping latest, previous and best-by-validation separate. For resume verification, compare the next update from an uninterrupted run with a resumed update on the same hardware; do not promise bitwise equivalence between the 4070 and T4.

This checkpoint policy is a requirement for the upcoming trainer, **not an already-implemented model recovery feature**. The current project has no competition-trained model yet.

Prefer committed Kaggle batch versions for long runs. Stop an idle interactive GPU session after saving a batch version so both do not consume quota. Keep output artifacts compact, provisionally below 10 GiB per run; do not export a copy of the input volumes. Retrieve successful run outputs into the Windows project with the authenticated CLI and record their hashes before relying on them as backups.

## Kaggle readiness work

- Private `clarkkitchen/biohub-gpu-readiness`: bounded GPU/data/checkpoint infrastructure check, internet disabled, 300-second run timeout; it does not train a competition model or submit predictions.
- Its first run failed quickly because the default Kaggle image does not contain Zarr. This is a verified dependency mismatch, not a GPU failure.
- Private `clarkkitchen/biohub-offline-dependencies`: CPU-only wheel preparation for Zarr and codecs. The GPU notebook verifies the bundle's hashes and installs from attached files with `--no-index`.
- Per-version results are kept under `work/` and accepted readiness results under `reports/`. A pushed version is not considered successful until its status and output report agree.

Verified outcome: dependency notebook version 1 and GPU readiness version 2 both completed. The GPU notebook remained private with internet disabled and used two Tesla T4 devices (15,636,037,632 bytes VRAM each). Forward/backward passed on both cards. It decoded a real `(100,64,256,256)` uint16 training sample and passed a tensor-checkpoint save/reload test. The 199 training samples were mounted directly. Results and the smoke checkpoint were downloaded into `reports/kaggle-readiness-v2/`.

Kaggle ran Torch `2.10.0+cu128`, whereas Vast currently has `2.9.1+cu128`. Before resuming a real training checkpoint across services, align the trainer dependencies or test the intended interchange path explicitly. The smoke checkpoint is only a serialization check, not a useful cell-tracking model. The readiness code reported 1.955 seconds for its GPU/data/serialization checks; that excludes package installation, notebook startup and shutdown, so it is not the GPU quota charge.

## Sources

- Live account editor, inspected 2026-09-05: https://www.kaggle.com/code
- Official quota guidance: https://www.kaggle.com/docs/efficient-gpu-usage
- Official notebook execution/accelerator metadata: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- Tmux persistence and session documentation: https://github.com/tmux/tmux/wiki/Getting-Started
- Competition runtime/metric requirements: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview
