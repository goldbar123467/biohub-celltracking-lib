# Status

Date: 2026-07-03

## Implemented

- Repo scaffold and concise `AGENTS.md`.
- Dependency-light graph schema.
- Lazy Zarr metadata reader with default physical scale.
- Synthetic GEFF JSON fallback reader.
- Local edge metric probes for sparse-GT semantics and node-count penalty.
- Local division metric probes for component/fork behavior.
- Greedy adjacent-frame linker.
- Classical detector scaffold and safe fallback node.
- Submission writer and validator CLI.
- Smoke Kaggle-path script.

## Scaffolded

- Real GEFF reading through optional `geff`.
- Official metric adapter through optional organizer dependencies.
- Learned 3D U-Net detector.
- Transformer/GNN linker.
- ILP/Motile integration.
- Notebook packaging workflow.

## Verified

- `python -m pip install -e .` completed successfully after package metadata fixes.
- `python -m pytest -q` passed: 19 tests.
- `python scripts/smoke_test_kaggle_path.py` passed on a synthetic metadata-only Zarr dataset.
- `python scripts/make_submission.py --data-dir /tmp/biohub_ct_mock_test --output submission.csv --debug` wrote a mock submission.
- `python -m biohub_ct.submission.validator submission.csv` passed on that mock submission.
- `python -m biohub_ct.metrics.probes` printed `edge_tp=1 edge_fp=0 edge_fn=0`.

Generated `.pytest_cache`, `__pycache__`, and mock `submission.csv` artifacts were removed after verification.

## Risks

- Local metric has not yet been cross-run against installed organizer metric dependencies.
- Kaggle pages should be rechecked in the live UI before final submission.
- The baseline is valid but intentionally weak; it is a path smoke test, not a leaderboard model.
