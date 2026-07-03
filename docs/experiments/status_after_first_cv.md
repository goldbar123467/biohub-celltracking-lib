# Status After First CV

Date: 2026-07-03

## Scope

Stage 1 work was limited to proving the local score loop. No U-Net, transformer, GNN, ILP, or ensemble implementation was added.

Real Biohub train data were not present under `/home/clark` or `/kaggle` in this workspace. The EDA/split/evaluation commands were implemented, tested, and executed on `/tmp/biohub_ct_mock_train`, a tiny metadata-only mock train directory. Real-data EDA and real fold scoring remain blocked until a real train path is provided.

## Commands Run

```bash
git status
git init
git add .
git commit -m "valid scaffold with classical baseline and submission validator"
git tag scaffold-valid-v0
python -m pip install polars scipy 'tracksdata @ git+https://github.com/royerlab/tracksdata@main'
python -m pip install --no-deps 'tracking-cellmot @ git+https://github.com/royerlab/kaggle-cell-tracking-competition@main'
python -m pytest tests/test_metric_synthetic.py tests/test_division_cases.py -q
python -m biohub_ct.metrics.probes --official
python scripts/eda.py --data-dir /tmp/biohub_ct_mock_train --output docs/experiments/eda_train.md
python scripts/make_splits.py --data-dir /tmp/biohub_ct_mock_train --output splits.json
python scripts/evaluate_local.py --data-dir /tmp/biohub_ct_mock_train --split splits.json --fold fold0 --pipeline classical --output docs/experiments/baseline_classical_fold0.md
python -m pytest -q
```

## Metric Parity Result

Focused parity tests passed:

```text
10 passed
```

Official probe output matched local probe:

```text
local:    edge_tp=1 edge_fp=0 edge_fn=0 edge_jaccard=1.000000 adjusted=1.000000
official: edge_tp=1 edge_fp=0 edge_fn=0 edge_jaccard=1.000000 adjusted=1.000000
```

## Mock Fold0 Result

This is a machinery check, not a meaningful leaderboard estimate.

| metric | value |
| --- | ---: |
| datasets | 1 |
| adjusted edge Jaccard | 0 |
| division Jaccard | nan |
| final score | 0 |
| predicted nodes / target nodes | 0.25 |
| edge TP/FP/FN | 0/0/1 |
| division TP/FP/FN | 0/0/0 |
| runtime | 0.212509 s |

## Top 5 Observed Failure Modes

These are from the mock metadata-only run:

1. Metadata-only Zarr forced the safe fallback node path.
2. The fallback produced one node and missed the GT edge.
3. Edge recall failed: false negatives exceeded true positives.
4. Predicted node density was below the target node count.
5. Real image data and visual artifacts are required to classify true detection/linking errors.

## Next 5 Highest-ROI Experiments

1. Run `scripts/eda.py` on the actual Kaggle `train/` directory and inspect displacement/NMS/link-gate distributions.
2. Run `scripts/make_splits.py` on actual train data and confirm embryo-disjoint folds.
3. Run unchanged classical baseline on `fold0` with real image chunks to establish the first meaningful score.
4. Sweep node density controls: threshold, NMS radius, and per-frame cap.
5. Sweep link distance from measured parent-child displacement percentiles before adding gap recovery or divisions.

## Blocker

Real stage-1 scoring is blocked by missing local competition data. Provide a path containing paired train `{name}.zarr` and `{name}.geff` folders, then rerun:

```bash
python scripts/eda.py --data-dir /path/to/train --output docs/experiments/eda_train.md
python scripts/make_splits.py --data-dir /path/to/train --output splits.json
python scripts/evaluate_local.py --data-dir /path/to/train --split splits.json --fold fold0 --pipeline classical --output docs/experiments/baseline_classical_fold0.md
```
