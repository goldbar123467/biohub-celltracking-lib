# Segmentation And Detection

## Current Baseline

The implemented baseline is intentionally conservative:

1. read one frame at a time when chunks are available;
2. robust quantile normalize;
3. find 3D local maxima;
4. apply anisotropic non-max suppression in physical units;
5. refine peaks by local intensity moments;
6. fall back to one central node if no data can be read.

This is not expected to be leaderboard-strong yet. It exists to validate the full offline submission path and provide an ablation anchor.

## High-ROI Improvements

- Tune threshold and NMS radius against local edge/division probes.
- Add multi-scale DoG/LoG when SciPy/scikit-image are available.
- Estimate per-dataset node count from training GEFF metadata and public data statistics.
- Add crop/intensity features for link scoring.
- Train a sparse-label heatmap detector only after the metric adapter is validated against organizer code.

## Sparse Label Warning

The organizer baseline explicitly uses sparse supervision: annotated edges train edge scores, while background detections and unannotated cells are ignored. Dense background loss from sparse labels is likely harmful.

