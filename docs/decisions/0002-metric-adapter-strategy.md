# Decision 0002: Metric Adapter Strategy

Use a dependency-light local metric for synthetic probes and keep an official adapter boundary for organizer code.

Reason: organizer metrics depend on `tracksdata` and `polars`. The local tests encode verified semantics from the organizer prose and code: 7 um matching, sparse GT FP rules, duplicate-edge dedupe, node-count penalty, and division component coverage.

Next step: install official dependencies in a controlled environment and compare every synthetic probe against `tracking_cellmot.metrics.evaluate`.

