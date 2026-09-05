# Biohub Cell Tracking Library

Competition source for [goldbar123467/biohub-celltracking-lib](https://github.com/goldbar123467/biohub-celltracking-lib).

The implemented path streams real Zarr frames through a classical point detector
and constrained adjacent-frame linker, checkpoints each completed dataset, and
atomically writes a validated submission CSV. Validation reloads that CSV and
calls a pinned organizer metric. A deterministic builder packages the same source
for offline Kaggle execution.

## Run

```bash
python -m pip install -e '.[dev,kaggle,official-metric]'
python -m pytest -q
python scripts/make_submission.py --data-dir /path/to/test \
  --config configs/classical.json --output reports/rehearsal/submission.csv
```

Read [testing, evaluation and submission commands](docs/testing-and-submission.md)
for whole-embryo folds, resume behavior, source/input identity, optional test
requirements and Kaggle packaging. `--debug` is a metadata-only fixture smoke
mode. Use the normal path for real image inference.

## Current scope

- Real uint16 TZYX Zarr loading with strict competition chunk checks.
- GEFF graph reading that preserves the estimated total cell count.
- Frozen 71/128-clip embryo splits, with overlap and missing-ID checks.
- Exact spatial-bin NMS and greedy association; detection uses original intensity
  maxima to avoid clipped plateaus.
- Graph/schema/bounds validation, atomic output and per-dataset recovery.
- CSV-round-tripped official evaluation with component metrics and provenance.
- Generated self-contained, internet-off Kaggle notebook with verified wheels.

Learned 3D detectors, temporal linkers, division models and ensembles remain
scaffolded. The classical baseline has measured engineering results, not a claim
of competitive model quality. See [executed evidence](docs/experiments/2026-09-05-architecture.md)
and [current research](docs/research/alpha-2026-09-05.md).

## Compute and operations

Vast.ai provides debugging and real-data evaluation; Kaggle provides final offline
execution. Read [AGENTS.md](AGENTS.md), [infrastructure setup](docs/infrastructure-setup.md)
and the [compute and recovery plan](docs/compute-and-recovery-plan.md) before runs.
The full dataset remains unavailable until the downloader and SHA256 verification
both complete. Source Git does not back up model checkpoints or local datasets.
Credentials, local connection settings and generated artifacts remain outside Git.
