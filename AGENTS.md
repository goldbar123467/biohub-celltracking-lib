# AGENTS.md

Repo-local instructions for AI coding agents.

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

