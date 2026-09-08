# E1 frozen-detector pilot results

The preregistered pilot is complete and does **not** meet its advancement gate. The lowest-count eligible diagnostic setting retained 12,444 predictions versus 14,590 in the frozen control, a 14.7087% reduction. Reaching 20% required at most 11,672 predictions. No new fit or model promotion follows from this result.

The bounded CUDA evaluator completed all 54 initial configurations and the two configurations admitted by its single refinement stage. Independent aggregation found no mismatch across all 56 rows. Every row covers the same eight previously inspected frames from four clips and all 26 sparse annotations. This population is `diagnostic_reuse_not_heldout`.

## Measured comparison

The table holds threshold and radius fixed within each pair of columns. It therefore separates extraction and precision effects from the selected radius change.

| Forward and activation | Extraction | Predictions at p=0.3, r=3 um | Predictions at p=0.7, r=4 um |
| --- | --- | ---: | ---: |
| AMP, native sigmoid | Legacy maxima | 14,590 | 13,247 |
| AMP, native sigmoid | Connected plateau | 14,466 | 13,121 |
| AMP, float32 sigmoid | Legacy maxima | 14,084 | 12,683 |
| AMP, float32 sigmoid | Connected plateau | 14,071 | 12,657 |
| Full float32 | Legacy maxima | 13,872 | 12,444 |
| Full float32 | Connected plateau | 13,872 | 12,444 |

At the original threshold and radius, plateau extraction alone removed 124 predictions, 0.85% of the control. Full float32 with plateau extraction removed 718, 4.92%. Increasing radius accounts for additional reduction in the selected setting. Threshold changes had negligible effect over the frozen grid. Precision and radius matter here, but these data do not support plateau handling as a sufficient remedy for excess detections.

Every configuration matched 26/26 sparse annotations using the explicitly diagnostic local matcher. Matched localization was unchanged: median 2.182489 um and p95 4.365179 um. These values do not establish prediction precision because most true nuclei are not annotated. The selected setting reduced cap frequency from 5/8 to 2/8 frames and clipped-candidate fraction from 14.4432% to 1.36335%; no candidate-pool truncation was recorded.

The separately frozen high-cap check retained 1,966 nodes with both the 2,000 and 262,144 caps on `44b6_0113de3b` frame 0. That frame was already uncapped. This cannot establish that other frames are uncapped.

![Preregistered detection frontier](e1-preregistration/e1-diagnostic-frontier.png)

## Image inspection and limitations

The root agent decoded and inspected all eight fixed overlay sheets, including XY, XZ and YZ full projections and physically scaled thin-slab crops. These compare native AMP at p=0.3 and r=3 um with legacy versus connected-plateau extraction. They do not depict the selected full-float32 setting. Both arms retain markers near bright centers and in dim interstitial locations; the extraction-only change has little visible effect in these crops. Depth-collapsed projections cannot resolve whether every marker belongs to a distinct nucleus.

The [visual review](e1-preregistration/visual-review.json) records every image hash, fixed selection, observation and limitation. This is AI image inspection, not human expert review or exhaustive annotation. The immutable renderer index still records that no manual review had occurred at render time; it was not rewritten after inspection.

Estimated-node count ratio, per-clip cap rate and fixed-linker adjusted edge score remain unavailable because the panel contains isolated frames rather than complete temporal clips. No complete temporal graph or division review was performed. The 20% predicted-count proxy also failed. Both the incomplete scientific evidence and the failed measured proxy independently prevent advancement.

## Evidence identity

- Frozen capture plan: `6f57b1fdac0eb2e611fa8166d0e03c96b1ca6bd447f772130c3c550d0cce7aec`.
- Evaluator plan: `f3bc2e225eeb56120288210afaea0d41f568a33da60350b0bd0e1dc274e77631`.
- Completed evaluator report: `ab9962044642a2af0ffd168ade417ccf4cbf5ea6b587164f0332c7e46385aae5`.
- [Durable index and limitations](e1-preregistration/README.md), [aggregate JSON](e1-preregistration/e1-diagnostic-summary.json), [all-row CSV](e1-preregistration/e1-diagnostic-frontier.csv), and [checksums](e1-preregistration/SHA256SUMS).

Raw NPZ payloads, full reports, PNG overlays and worker receipts are retained under the local campaign download directories and original Vast snapshots. The durable index identifies them but does not substitute for the raw bundle. The recorded immutable R7 source suite passed 265 tests on Vast with real optional dependencies before subsequent Kaggle rehearsal additions; this count must not be attributed to later source.
