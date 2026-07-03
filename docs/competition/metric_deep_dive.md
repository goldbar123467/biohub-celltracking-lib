# Metric Deep Dive

## Edge Matching

Nodes are matched per timepoint by optimal bipartite centroid assignment in physical units. The max match distance is 7 microns. The scale is anisotropic, so voxel-distance thresholds are wrong unless converted through `(z,y,x)=(1.625,0.40625,0.40625)` microns.

A predicted edge is a true positive when both predicted endpoints match GT nodes and those GT nodes have the same directed edge. Duplicate predicted source-target edges must not inflate TP.

## Sparse Ground Truth

The GT graph is sparse. Predicted nodes that do not match GT nodes are not direct node false positives. Predicted edges in unannotated regions can be ignored. However, predicted edges touching annotated in/out regions become false positives when they do not match a GT edge.

Practical implication: overpredicting plausible unannotated cells is not automatically fatal, but overpredicting nodes still affects the adjusted edge score.

## Node-Count Penalty

The organizer code applies:

```text
adjusted = max(0, edge_jaccard * (1 - 0.1 * (num_pred_nodes - total_true_nodes) / total_true_nodes))
```

The local synthetic tests intentionally preserve the observed behavior that underpredicting relative to `total_true_nodes` can raise the adjusted value above raw edge Jaccard. Do not cap this locally unless the organizer code changes.

## Division Scoring

Metric prose says a cell division is a node with exactly two outgoing edges. The organizer code uses graph dividing-node behavior and stage/component checks that behave like `out_degree >= 2` for predicted forks. Document this as a prose-vs-code detail before changing division logic.

A GT division is recovered only when matched predicted nodes:

- cover a one-node pre-split stage;
- touch both daughter lineages;
- lie in one weakly connected predicted component;
- include at least one predicted fork.

Predicted divisions in unannotated regions are ignored. Predicted divisions matched to annotated GT regions count as FP if they are not paired to a GT division.

## Final Score

The final score is adjusted edge Jaccard plus `0.1 * division_jaccard` when divisions are present. Edge agreement and node-count calibration are the main optimization targets.

## Official Parity Status

Status on 2026-07-03: synthetic parity is passing against the installed organizer package from `royerlab/kaggle-cell-tracking-competition@016845f`.

Commands:

```bash
python -m pytest tests/test_metric_synthetic.py tests/test_division_cases.py -q
python -m biohub_ct.metrics.probes --official
```

Coverage:

- edge TP/FP/FN and Jaccard;
- duplicate edge dedupe;
- 7 um node matching inside/outside threshold;
- sparse-GT ignored unannotated edges;
- spurious edges touching annotated regions;
- adjusted edge Jaccard node-count penalty;
- division TP/FN/FP behavior for component coverage and annotated-region false forks.

Known mismatches: none in the current synthetic cases. Remaining risk: parity has not yet been checked on full real GEFF/Zarr train datasets in this workspace because real competition data are not present locally.

## Pathological Cases

- Duplicate predicted edges can fake high TP unless deduped.
- Dense detections can preserve edge Jaccard while losing adjusted score through node count.
- A naive exact-time division scorer can reject valid temporally shifted forks.
- Spurious forks near annotated continuing tracks can create division FP.
