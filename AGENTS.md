# AGENTS.md

Repo-local instructions for AI coding agents.

Canonical source repository: `https://github.com/goldbar123467/biohub-celltracking-lib.git`.
Use it for this project and synchronize reviewed commits between Windows and Vast.
Preserve existing work and history; never force-push or discard unrelated changes.
This repository is public. Keep actual SSH settings, credentials, competition data,
checkpoints and generated run reports in ignored local paths. See
`docs/infrastructure-setup.md` and `configs/ssh_config.example` for setup.

## Setup

```bash
python -m pytest -q
python scripts/smoke_test_kaggle_path.py
python scripts/make_submission.py --data-dir <test_dir> --output submission.csv --debug
python -m biohub_ct.submission.validator submission.csv
```

Use Python 3.11+. The verified core path is stdlib plus NumPy. Treat `zarr`,
`scipy`, `scikit-image`, `geff`, `polars`, `tracksdata`, and `torch` as optional
unless a task explicitly requires them.

## Architecture Map

- `src/biohub_ct/data/`: dataset discovery, lazy Zarr metadata, GEFF adapters, graph schema.
- `src/biohub_ct/metrics/`: local metric probes and official metric adapter.
- `src/biohub_ct/detection/`: classical detector and learned-detector scaffold.
- `src/biohub_ct/linking/`: greedy/LAP/ILP linking interfaces.
- `src/biohub_ct/pipelines/`: runnable baseline and submission pipeline.
- `src/biohub_ct/submission/`: writer, validator, repair helpers.
- `docs/`: competition facts, research summaries, agent operations, decisions, experiments.

## Hard Rules

- Kaggle inference must not require runtime internet.
- Submission columns stay exactly `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`.
- Node rows use `source_id=-1,target_id=-1`; edge rows use `node_id=t=z=y=x=-1`.
- `id` values are consecutive integers starting at 0.
- Every test dataset must appear; use the documented fallback node only when detection returns none.
- Never load a full 4D Zarr video unless the command explicitly opts into that.
- Validate `submission.csv` before calling a task complete.

## Work Standards

- Inspect official metric/source docs before changing scoring code.
- Add or update focused tests for behavior changes.
- Keep learned-model and optional-solver code behind graceful imports.
- Record non-obvious choices in `docs/decisions/`.
- Record experiments using `docs/agent_ops/experiment_protocol.md`.


## Compute infrastructure and project operations

## Two testing platforms are available

Use both **Vast.ai** and **Kaggle cloud** for authorized project work. The user explicitly wants to use the existing Kaggle GPU allowance alongside Vast. Do not plan as though Vast is the only compute option or treat Kaggle as only a submission destination.

The following setup was verified on **2026-09-05**. Hardware availability, quota, running jobs, data-download progress and installed versions must be checked when relevant before a new run.

| Platform | Verified setup | Intended work |
|---|---|---|
| Vast.ai | RTX 4070 SUPER, 12,282 MiB VRAM; 9.6 CPU quota; about 85.45 GiB container RAM; 130 GiB disk | Interactive debugging, data audits, official metric tests, preprocessing and short GPU experiments |
| Kaggle cloud | Private GPU readiness notebook completed with internet off on two Tesla T4 devices, each with 15,636,037,632 bytes VRAM | Bounded training experiments, independent embryo-held-out validation, controlled ablations and final offline inference rehearsals |

Windows at `C:\Users\thecl\Documents\Biohub-Cell-Tracking` is the durable project record for source, configurations, experiment notes and retrieved reports. The Vast project is `/workspace/biohub-cell-tracking`.

Read `docs/compute-and-recovery-plan.md` for allocation and recovery details, `docs/competition-contract.md` for data/evaluation constraints, and `README.md` for setup commands. Preserve the user's applicable engineering and ML instructions; read relevant skills before substantial implementation or review.

## Choose and budget the platform deliberately

- Use Vast for fast diagnosis and bounded local smoke runs; use Kaggle for reproducible cloud experiments and target-environment verification. Record which platform produced each result.
- Kaggle mounts competition data directly. Its work need not wait for the Vast download, and must not copy the entire input dataset into `/kaggle/working`.
- The account displayed 30 hours of GPU allowance before readiness testing. This is a dated snapshot, not a current balance or permission to consume it all. Refresh the live quota and accelerator choices before each batch. Follow the provisional allocation in the compute plan and benchmark throughput, peak VRAM and loader time before costly training.
- Two T4 cards have separate memory; they do not provide a single pooled allocation. Benchmark one device first. Use multiple devices only when the method and measured benefit justify it. Do not distribute training across Vast and Kaggle over the internet.
- Prepare offline dependency wheels on a Kaggle CPU notebook. Prefer committed batch versions for long GPU runs and stop idle interactive GPU sessions after saving work to avoid duplicate quota consumption.
- Existing infrastructure may be used within the user's authorized project scope without redundant permission requests. New rentals, upgrades, purchases, recurring jobs or substantial expansion beyond that scope require user authorization.

## Vast connection, jobs and storage

Read the provider instructions at `/etc/vast-agents-guide.md` before acting on the server; `/workspace/AGENTS.md` is the provider-level guide. This project file belongs at `/workspace/biohub-cell-tracking/AGENTS.md`; do not replace the provider guide.

Connect from PowerShell:

```powershell
& 'C:\Users\thecl\Documents\Biohub-Cell-Tracking\scripts\connect.ps1'
```

This attaches or creates tmux session `biohub` on its dedicated `-L biohub` socket and activates the project environment. Detach with Ctrl+B, then D. For noninteractive SSH, use the `vast-biohub` alias in `configs/ssh_config`.

Run substantial commands through the logged job launcher after activating the project environment:

```bash
source /workspace/biohub-cell-tracking/scripts/activate.sh
bash /workspace/biohub-cell-tracking/scripts/run-job.sh unique-run-id command arguments
```

Replace the example ID and command with the actual experiment. Inspect `reports/jobs/<run-id>/output.log` and `exit-code.txt`. Run IDs cannot be reused; a missing exit code means unfinished or interrupted, not success. Tmux disconnect survival and failure-code recording were tested. Tmux does not protect against Python errors, reboot, instance loss or CUDA OOM, and is not automatic checkpoint recovery.

The dataset download has its own Supervisor process, `biohub-download`. Check `supervisorctl status biohub-download`, `reports/download-status.json` and `reports/download.log` before using or restarting it. Treat the complete dataset as available only after completion and verification with `scripts/verify_data.py --sha256`. Restarting the streaming download replays the ZIP; inspect failures before explicitly restarting it. Keep the downloader's 15 GiB reserve and check actual disk space before caches or training outputs grow.

This instance has no persistent volume. Stop/start preserves files but ends processes; recycle/destruction loses the filesystem. Retrieve valuable reports and checkpoints to durable storage and verify their hashes. Do not alter drivers or provider management services for routine project work.

## Kaggle execution and evidence

The authenticated account is `clarkkitchen`. Existing private notebooks and local sources:

- `clarkkitchen/biohub-offline-dependencies`, verified version 1: CPU wheel preparation; source in `notebooks/kaggle-dependencies/`.
- `clarkkitchen/biohub-gpu-readiness`, verified version 2: offline GPU/data/serialization smoke run; source in `notebooks/kaggle-readiness/`.
- Accepted readiness evidence: `reports/kaggle-readiness-v2/` and `reports/tmux-and-cloud-readiness.json`.

From Windows, the configured CLI is available through WSL:

```powershell
wsl -d Ubuntu-24.04 -u thecl -- /home/thecl/.local/bin/kaggle kernels status clarkkitchen/biohub-gpu-readiness
```

Reuse the established authenticated setup. Credential locations are documented in the ignored `docs/local-setup.md` on the configured workstation; never print, commit or embed tokens/private keys in notebooks, logs or artifacts.

The first GPU readiness version failed because Zarr was absent from the base image. Version 2 succeeded using the attached dependency notebook output, verified wheel hashes and `--no-index` installation. Preserve this offline dependency path. A successful push is not a successful run: inspect the completed version's status, logs and output report, retrieve required artifacts and verify their hashes.

Vast was tested with Torch `2.9.1+cu128`; Kaggle used `2.10.0+cu128`. Align dependencies or explicitly test checkpoint interchange before resuming real training across platforms. Record environment, source revision, config, seeds and split IDs with each run. Do not infer matching numerical behavior from matching CUDA version strings.

## Evaluation and recovery requirements

- Use whole-embryo validation: fit `44b6`, evaluate `6bba`, then reverse. Keep all views and timepoints from an embryo together. Do not randomly split frames. Report both directions and the metric components; two embryos provide limited evidence of generalization.
- GEFF labels are sparse; unlabeled cells are not automatically negatives. Follow the frozen data and metric contracts before building a training target or scorer.
- Before substantial training, implement and test atomic checkpoints containing model, optimizer, scheduler, scaler, RNG states, epoch/step, config, split identity and dependency/source versions. Save at least every ten minutes and at epoch boundaries, retaining latest, previous and validation-best separately. Test resumed versus uninterrupted updates on the same hardware.
- Kaggle checkpoints must become saved version outputs and be retrieved when needed as backups. Unsaved interactive files are not durable recovery artifacts. Inspect failures before relaunching; do not silently retry or assume a resume feature exists.
- Final inference must be tested in Kaggle with internet off, within the competition's runtime limit, discovering actual hidden-test sample IDs and producing `submission.csv`. The four public example tests are copied from training and are not held-out validation. Recheck current competition rules before submission.
- Readiness checks established infrastructure operation, including real-data decoding and GPU forward/backward. The saved smoke checkpoint is a synthetic serialization test, not a trained cell-tracking model. No competition model training or submission was completed by setup. Update the experiment record as work advances and distinguish executed results from proposed work.

Keep this project-scoped file synchronized between Windows and Vast when changing instructions. Preserve provider-level instructions and do not place credentials in either copy.
