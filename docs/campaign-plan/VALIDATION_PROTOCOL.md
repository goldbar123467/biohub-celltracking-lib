# Validation and promotion

## Freeze the evidence population

Every metrics artifact must include exact clip IDs, embryo/direction, frame coverage, data hash/version, training membership, development-selection history, scorer commit/wheel hash, model/config hash, and missing/failed clips. A scorer that evaluates only the intersection of prediction and ground-truth directories can silently omit failed clips; explicitly require equality with the planned evaluation set before declaring a complete result.

The supplied learned model trained on `6bba`, selected its fold0 checkpoint/threshold internally, and evaluated 61 of 71 `44b6` clips. The reverse fit existed but reverse outer evaluation was not completed. Do not fill missing results with zero, drop them silently, or call this two-direction OOF. The classical comparator in this panel is not established as the July public-score model.

Separate four labels:

| Label | Meaning | Valid use |
| --- | --- | --- |
| Development | Used to select code/settings/model | Optimize and diagnose |
| Previously inspected diagnostic | Outer or other data examined during iterative decisions | Explain behavior, with contamination disclosed |
| Locked evaluation | Excluded from fitting, teacher generation, calibration, and selection until freeze | Paired generalization evidence to the extent membership is known |
| Overlap unknown | Pretrained model membership cannot be established | Descriptive behavior and eligible reproduction, no clean held-out claim |

Group by whole embryo for outer directions and prevent overlapping spatiotemporal crops from crossing internal development splits. Record source-volume relationships; disjoint clip filenames are not proof of independent inputs. With only two embryos, clip or frame resampling cannot establish broad biological-population confidence. Report each direction and their score components. If no genuinely untouched data remain, say so and use a frozen diagnostic protocol rather than inventing a holdout.

## Metric contract

Use the pinned organizer implementation as the authority. The supplied September audit and organizer documentation specify, for sample `i`:

```text
U_i = edge_TP_i + edge_FP_i + edge_FN_i
J_i = edge_TP_i / U_i
r_i = N_pred_i / N_est_i
A_i = max(0, J_i * (1 - 0.1 * (r_i - 1)))
AdjustedEdge = sum(U_i * A_i) / sum(U_i)
Division = sum(div_TP_i) / sum(div_TP_i + div_FP_i + div_FN_i)
Score = AdjustedEdge + 0.1 * Division
```

Handle empty denominators, absent or sparse estimated counts, and malformed inputs exactly as the pinned scorer specifies. The formula above describes ordinary valid samples; do not invent edge-case conventions or cap values differently. Keep `N_est` evaluation-only. Inference must not require ground-truth-derived estimated counts unavailable on hidden inputs.

Raw pooled edge Jaccard is `sum(TP)/sum(U)`. It is a separate diagnostic. Never substitute the mean count ratio into the penalty or macro-average clip scores as the official aggregate. Sample weights can change when predictions change: compute both candidates from their own complete official raw results, then pair comparisons by identical clip coverage.

For intuition only, the multiplier for one sample is 1 at ratio 1, 0.9 at ratio 2, 0.7 at ratio 4, 0.5 at ratio 6, and 0 at ratio 11. This does not reconstruct the score from the reported mean ratio 6.39. Reducing nodes changes matching and linking as well as this multiplier; count alone cannot select a candidate.

Sparse annotations do not support dense precision estimates. Unmatched predictions can be real cells. Report annotated-node recall and matched localization error, not an invented detector precision. Preserve the official sparse edge and division matching behavior instead of penalizing every unmatched edge or fork. Verify the exact matching units and tolerance from the pinned scorer.

## Required output per candidate

| Component | Required measurements |
| --- | --- |
| Detection | Annotated recall, matched count, median/p90/p95 localization error in micrometers, count ratio distribution, predicted totals |
| Peak extraction | Raw maxima, plateau statistics, pre-cap candidates, capped-frame count and denominator, capped-clip count and denominator |
| Association | Pooled edge TP/FP/FN, raw Jaccard, official adjusted contribution, edge counts and displacement distribution |
| Divisions | Pooled TP/FP/FN, division Jaccard, predicted forks by processing stage; all-clips and division-bearing-subset reports |
| Full score | Official combined score, each embryo/direction, complete evaluated ID set and missing IDs |
| Operational | End-to-end time, stage times, peak RAM/VRAM, actual quota/bill debit, fallback/retry count |

Store per-clip raw statistics so aggregates can be independently recomputed. Show mean and quantiles of count ratios rather than only a pooled ratio. Pooled annotated recall and mean per-clip recall have different meanings; report their denominators and do not compare one to the other's historical number.

## Numerical and coordinate diagnostics

Compare original inference, AMP logits with float32 sigmoid, and full float32 inference on identical frames. Record model output before activation, probabilities, ties, extraction order, and retained nodes. The supplied 1,340 to 1,241 observation proves a local numerical effect only; the persistent 2,000-node caps show it is not a complete explanation.

Track coordinate conventions through OME axes, anisotropic physical scale, downsampling, tile origin, overlap blending, and output conversion. Test a known point through the complete transform and round trip. Read actual metadata, including uppercase T/Z/Y/X, rather than hardcoding the earlier default spacing. Check borders and half-voxel offsets. Exclude padded regions from exported detections.

When comparing two detectors with different centers, use the official assignment or an explicitly named diagnostic matching rule; row order and nearest-neighbor reuse are insufficient. Save overlays or orthogonal slices for several high-impact failures with identical coordinates and intensity windows.

## Development advancement versus release

Development gates in `EXPERIMENT_PLAN.md` allocate the next experiment. They do not establish final generalization. Before looking at a new candidate's locked results, record the primary control, the minimum useful development change, and permitted regression. Suggested release-review defaults for our own model changes are positive paired development score change and no embryo-direction score loss greater than 0.005 absolute on completed available evaluation directions. This is an operational tolerance, not a significance threshold or a guarantee. A zero tolerance may be appropriate for a pure correctness fix; preregister the choice.

When locked evaluation is possible, perform one frozen evaluation in both directions, with identical complete coverage for candidate and control. If a change is selected because of those results, that panel becomes selection evidence for subsequent work. Do not repeatedly tune on the same set and preserve its held-out label. Exploratory submissions remain possible with the helper's explicit experimental classification, bounded submission allowance, and honest evidence limits.

For the first public reference reproduction, verified eligibility, artifact identity, full offline execution, valid output, and adequate runtime are sufficient to submit as an external reproduction experiment even if a clean local comparison is unavailable. Do not force an overlap-unknown pretrained model through a falsely independent CV gate. Any claim of beating our incumbent requires actual comparable results or completed public scores, clearly labeled.

## Division evidence

A predicted fork is not itself a recovered division. Use the official local topology, timing tolerance, matching, and false-positive accounting from the exact scorer revision. Include zero-GT-division clips when pooling assessable FPs; evaluate the division-bearing subset additionally, not exclusively. Do not assign division Jaccard zero or one to empty clips and macro-average it.

A simple sensitivity calculation for a fully pooled event set is `0.1 * TP / (TP + FP + FN)`, while graph edits can also change the edge term. If there are 22 evaluated GT divisions, one isolated recovered event with no introduced FP and no edge change would contribute about 0.00455 to the division term. This is a conditional arithmetic example, not an expected gain. Inspect every newly matched or newly false fork when counts are small.

## Reuse the existing correctness gates

The September architecture reports establish useful real-IO, graph, count, serialization, deterministic-build, and official-scorer tests. Reuse them after confirming the current checkout and environment. Run the relevant regression plus one full release-path rehearsal. CI success on an environment without optional organizer dependencies does not establish official-metric parity. Do not spend GPU time on mock-only score evaluations.
