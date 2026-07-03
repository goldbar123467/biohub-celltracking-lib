# Classical Baseline Fold Evaluation

Data dir: `/tmp/biohub_ct_mock_train`
Splits: `splits.json`
Fold: `fold0`
Pipeline: `classical`

## Summary

- datasets: 1
- adjusted_edge_jaccard: 0
- division_jaccard: nan
- final_score: 0
- edge_tp/edge_fp/edge_fn: 0/0/1
- division_tp/division_fp/division_fn: 0/0/0

## Per Dataset

| dataset | runtime_s | pred_nodes | target_nodes | node_count_ratio | edge_tp | edge_fp | edge_fn | division_tp | division_fp | division_fn | adjusted_edge_jaccard | division_jaccard | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| embA_0001 | 0.212509 | 1 | 4 | 0.25 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | nan | 0 |

## Observed Failure Modes

- Metadata-only or no-detection fallback produced one node and missed GT edges.
- Edge recall is the dominant failure: false negatives exceed true positives.
- Predicted node density is below target node count.
- Need real image data and visual failure artifacts to classify remaining errors.
- Need real image data and visual failure artifacts to classify remaining errors.
