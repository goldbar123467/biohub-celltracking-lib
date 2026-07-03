# Leaderboard Strategy

## First-Pass ROI

1. Validate the metric against organizer code with synthetic graphs.
2. Calibrate node count per dataset before chasing division recall.
3. Tune classical peak threshold, NMS radius, and link distance.
4. Add one-frame recovery only after metric probes for non-adjacent edges.
5. Add learned detector/linker once the classical path is reproducible.

## Precision vs Recall

Favor edge precision in annotated regions. Extra unannotated nodes are tolerated only until the node-count penalty dominates. Use `estimated_number_of_nodes` where available to tune output density.

## Divisions

Default to conservative division proposals. Division score has low weight but spurious annotated-region forks can hurt. Add divisions only after edge linking is stable.

## Public LB Risk

The crawled leaderboard snapshot says public LB is about 29% of test data. Use it as a sanity signal, not as the primary optimizer. Group local validation by embryo prefix to reduce leakage risk.

## Runtime

Read frame-by-frame, avoid full video loads, and keep optional learned inference behind explicit config. The hidden test set is approximately train-sized in the data-page snapshot, so debug performance on multiple local samples before submitting.

