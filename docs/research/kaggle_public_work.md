# Kaggle Public Work Snapshot

Retrieved by web search on 2026-07-03. Treat this as a triage list, not validated code.

## Notebooks To Inspect

- `Biohub Cell Tracking - Classical Baseline` by Xiaolei Lian: search snippet says it turns each 3D+time movie into a lineage graph and writes `submission.csv`.
  URL: https://www.kaggle.com/code/xiaoleilian/biohub-cell-tracking-classical-baseline
- `Biohub Cell Tracking: Data Model, EDA, Baseline` by Pilkwang Kim: snippet mentions a self-contained rule-based tracker with multi-scale DoG detection and physical linking.
  URL: https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-data-model-eda-baseline
- `LB839 Learned Graph Tracker` by yusuketogashi: snippet describes an inference-only learned 3D U-Net detector plus lightweight physical-distance tracking.
  URL: https://www.kaggle.com/code/yusuketogashi/lb839-learned-graph-tracker-micro-safe-divisi
- `Biohub Cell Tracking: Learned Graph w Gap Recovery` by Pilkwang Kim: snippet mentions learned detections, two-pass physical-motion relinking, and one-frame gap recovery.
  URL: https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery

## Extracted Public Ideas

- Classical baselines are competitive enough to deserve careful calibration before training large models.
- Physical-distance gates and node-count calibration likely matter because the official metric penalizes total predicted node count.
- Gap recovery appears to be a common improvement path, but non-adjacent edge behavior must be metric-tested before enabling.
- Conservative division proposals are safer than broad division recall until local probes quantify FP cost.
- Learned detectors can help, but the training path must respect sparse annotations.

## Rules For Using Public Work

- Record notebook URL, version, author, and exact idea before implementing.
- Re-implement ideas from understanding; do not paste code without license/rule review.
- Run local metric probes before using any public-LB-driven heuristic.

