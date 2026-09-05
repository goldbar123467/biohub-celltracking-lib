# Infrastructure setup

Recorded 2026-09-05. The source repository is
`https://github.com/goldbar123467/biohub-celltracking-lib.git`.
Windows and Vast should use the same reviewed commit. Do not overwrite uncommitted
changes when synchronizing; preserve provider-level `/workspace/AGENTS.md`.

## Connection and environment

The configured Vast checkout lives at `/workspace/biohub-cell-tracking`.
Existing connection scripts use ignored `configs/ssh_config`. A new workstation
can copy `configs/ssh_config.example` and fill in its own host, SSH port and key
path after verifying the server host key. Existing Windows connection details and
credential locations are retained in ignored `docs/local-setup.md`.
Never commit credentials or copy a private SSH key onto the GPU server.

Use `scripts/connect.ps1` from Windows to attach to tmux; use
`source /workspace/biohub-cell-tracking/scripts/activate.sh` on Vast.
The scripts assume that server project path. They are operational helpers, not
a general-purpose provisioning system.

The existing server environment uses Python 3.12.14 and Torch 2.9.1+cu128.
`requirements-setup.txt` records setup dependencies;
`requirements-server.lock.txt` records the installed snapshot, including an
editable local organizer checkout. It is not a portable, standalone lockfile.
Reconstruction requires the pinned sources:

- Organizer baseline: `https://github.com/royerlab/kaggle-cell-tracking-competition`,
  commit `075fc5f5a52d11077f9dc2b074644618f26939e2`, under `vendor/official-baseline`.
- Tracksdata: `https://github.com/royerlab/tracksdata`,
  commit `63a1912f3b6ebd1536a2e8a8adfdf7f5eb84efa4`.

The dependency-light library remains configured by `pyproject.toml`; these server
requirements supplement it for infrastructure and organizer checks. Do not install
optional dependencies from an unpinned branch merely to recreate the verified server.

## Data and checks

The authenticated inventory is generated with `python scripts/inventory_api.py`.
It contains 24,886 files totaling 87,609,892,618 bytes. Inventory and extraction
receipts remain in ignored `reports/`. Inspect the existing Supervisor download
and its status before launching another download. Full verification requires
completion, then `python scripts/verify_data.py --sha256`.

On the configured server, bounded infrastructure checks are:

```bash
source /workspace/biohub-cell-tracking/scripts/activate.sh
python scripts/smoke_environment.py
python -m pytest vendor/official-baseline/tests/test_metrics.py vendor/official-baseline/tests/test_division_metrics.py vendor/official-baseline/tests/test_division_sandbox_examples.py -q
```

The setup run passed CUDA forward/backward, real image/GEFF decoding and 102
organizer metric tests. These are separate from this library's own tests and do
not establish full metric parity or real-data baseline quality.

When integrating this infrastructure with repository baseline
`dfd932d6665615a567af5c7aa3b58e6efb91a685` on 2026-09-05, the existing Vast
environment passed all 22 library tests, the Kaggle-path smoke script, CLI
submission generation and validation on a one-dataset mock input, and the local
versus official synthetic metric probe. Shell syntax checks and compilation of
the infrastructure Python scripts also passed. No existing library implementation
was changed by that integration. These checks establish integration and a small
synthetic parity case, not real-data accuracy or comprehensive metric equivalence.

## Kaggle and recovery

Private notebook sources are in `notebooks/kaggle-dependencies/` and
`notebooks/kaggle-readiness/`. Dependency version 1 and GPU readiness version 2
completed. Readiness installed hash-verified attached wheels offline, decoded a
real sample, exercised both T4 GPUs and round-tripped a synthetic checkpoint.
The exact downloaded outputs remain in local `reports/kaggle-readiness-v2/`.

Kaggle used Torch 2.10.0+cu128, so test or align dependencies before exchanging a
real training checkpoint with Vast. Internet must be disabled for final inference.

Use the existing tmux job launcher for work that must survive SSH disconnects.
The server has no persistent volume: tmux cannot preserve a process across reboot
or files across instance destruction. The
[compute and recovery plan](compute-and-recovery-plan.md) defines GPU allocation,
logging, checkpoint requirements and backups. Those checkpoint requirements still
need to be implemented in a future trainer.
