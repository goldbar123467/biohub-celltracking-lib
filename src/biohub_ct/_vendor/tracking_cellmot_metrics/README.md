# Organizer Metric Boundary

Organizer repository: https://github.com/royerlab/kaggle-cell-tracking-competition

License in the organizer repo snapshot: BSD-3-Clause.

This repo does not copy the organizer metric source into the active import path because it depends on `tracksdata` and `polars`, which are optional here. Instead:

- `biohub_ct.metrics.official_adapter` detects whether `tracking_cellmot` is importable.
- `biohub_ct.metrics.edge` and `biohub_ct.metrics.division` implement dependency-light synthetic probes based on the official prose and code.
- Future work should install organizer dependencies and add parity tests before using local CV scores for model selection.

