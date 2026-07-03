# Biohub Celltracking Lib

Competition library for Kaggle's Biohub - Cell Tracking During Development.

This repo is built around four priorities:

- keep the Kaggle inference path offline and reproducible;
- match the organizer metric semantics before optimizing models;
- iterate with a dependency-light classical baseline first;
- document source facts and experiment decisions for future agents.

## Quickstart

```bash
python -m pip install -e .
python -m pytest -q
python scripts/smoke_test_kaggle_path.py
python scripts/make_submission.py --data-dir /path/to/test --output submission.csv --debug
python -m biohub_ct.submission.validator submission.csv
```

The default verified path uses Python 3.11+ plus NumPy. Optional packages such as
`zarr`, `scipy`, `scikit-image`, `pandas`, `geff`, `polars`, and `tracksdata`
unlock richer local data access and official-metric integration.

## What Works Now

- Discovers `.zarr` datasets and paired `.geff` folders.
- Opens Zarr metadata lazily without loading whole videos.
- Reads synthetic `.geff/graph.json` fallback graphs.
- Represents tracking graphs as nodes and directed temporal edges.
- Runs synthetic edge/division metric probes matching the organizer semantics covered by tests.
- Runs a conservative no-training baseline with safe fallback output.
- Writes and validates exact `submission.csv` schema.
- Provides Kaggle-offline scripts and a notebook scaffold.

## What Is Scaffolded

- Real GEFF reading through the optional `geff` package.
- Official organizer metric calls through `tracking_cellmot` objects.
- Learned 3D U-Net detector, transformer/GNN linker, ILP solver, and ensembles.

Start with `docs/competition/metric_deep_dive.md`, then run the smoke tests before
touching the Kaggle path.
