# E1 durable preregistration and result index

`capture-plan-r5.json` is the immutable plan frozen before capture. `configs/` contains the authored inference settings. Original Windows/Linux paths are historical provenance, not portable installation paths.

The evaluator plan with completion receipts, admitted-run and annotation identity indexes, result tables/plots, visual-review report and their `SHA256SUMS` are retained at their original paths in the local checkout and explicitly ignored. They are generated experiment evidence and are not distributed with the public source. The authored experiment notes summarize the results and their limits; local evidence is required to independently reproduce the recorded audit.

`e1-diagnostic-summary.json` and `e1-diagnostic-frontier.csv` contain independently recomputed aggregates for 54 initial and two refinement rows. PNG and SVG figures are rendered from those values. `visual-review.json` records subsequent AI inspection of all eight fixed overlay sheets and their hashes. It does not alter the immutable renderer's original `manual_review_performed=false` value or claim human expert review.

Raw logits, the full evaluator report, downloaded controller receipts, and overlay PNGs remain in the local ignored reports and Vast snapshot directories identified by their source manifests. This index does not contain the complete raw experiment bundle. The ignored `SHA256SUMS` records the original snapshot, including the earlier README. Its original README bytes and the preservation receipt are retained in `reports/campaigns/public-sync-preservation-20260908/`; the current README documents the later public-source separation.

The selected result reduced predicted count by 14.7087%, below 20%. All 26 sparse annotations were matched, but no complete temporal clip, official adjusted-edge score, estimated-node count ratio, or per-clip cap rate is available. Advancement remains false.
