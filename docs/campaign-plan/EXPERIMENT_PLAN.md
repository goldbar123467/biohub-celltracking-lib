# Ordered experiments

## Decision objective

Optimize the pinned official final score subject to runtime, artifact validity, and resource limits. The local learned detector's near-perfect annotated recall is already paired with severe count inflation. Further training is warranted only when an experiment identifies a better supervision target or a measurable learning trend that survives the count adjustment.

The public reference is the first deployment candidate. Repairing our detector is a parallel research question, not a prerequisite to submitting an eligible reproduction. Clean generalization claims require known training membership; public-model deployment eligibility is a separate question.

## Queue and dependencies

| ID | Experiment | Starts after | Compute | Decision artifact |
| --- | --- | --- | --- | --- |
| E0 | Pin and reproduce public reference | Live inventory and budget | GPU inference plus CPU audit | Frozen baseline, Kaggle receipt, provenance classification |
| E1 | Diagnose our detector without fitting | Development manifest and existing weights | One logit-cache pass, then mostly CPU | Recall/count/localization frontier |
| E2 | Fixed-node association and divisions | E0 or a frozen viable node set | Cached features and CPU graph work | Paired component ablations |
| E3 | Teacher audit and matched distillation | E1 identifies unresolved density failure; budget remains | Rented GPU | Teacher QC and matched real development comparison |
| E4 | Targeted synthetic pretraining | Valid graph adapter and a specific E2 failure | Small bounded training pilot | Equal-budget real-only versus synthetic-plus-real comparison |
| E5 | Scale, refit, and release | A candidate earns advancement | Reserved compute and Kaggle | Final exact-version submission |

E2 can run on cached outputs while E3 uses a GPU. E4 is optional. A better working reference can make E3/E4 unnecessary. Do not wait for all experiments before releasing E0.

## E0 — Reproduce the public learned reference

Hypothesis: the reported public pipeline provides a much stronger and more practical starting point than our current sparse detector. This remains untested in our environment.

1. Resolve the exact version of `redoctopusk/biohub-942tta` associated with the September 7 report. Download the notebook and attached model/support artifacts without editing them. Save notebook version, dataset versions, source and weight SHA-256, dependency versions, licenses, and effective runtime settings. If the score-associated version cannot be established, call the result a pinned reproduction candidate, not an exact reproduction of 0.946.
2. Inspect sequentially executed notebook cells and support code. The reported family uses two temporal U-Nets, a separate center-evidence model, learned association, graph optimization, gap repair, and division postprocessing. Resolve preprocessing, coordinate transforms, effective TTA, association normalization, solver options, random seeds, and every overwritten constant.
3. The audit reports threshold 0.965, reverse weight 0.15, harmonic fusion, appearance cost 0, disappearance cost 2, division cost 1.2, and a gap-repair branch effectively clamped to one missing frame. Verify each against the pinned code. Do not silently substitute these values if the selected version differs.
4. Run representative development inference for compatibility and timing, clearly flagging unknown training overlap. Build a private Kaggle reproduction with unchanged algorithmic settings and all dependencies available offline. Only environment/path adaptations belong in this first version; document each diff.
5. Validate complete visible input coverage, graph structure, bounds, runtime, and output manifest. Store stage-level counts and results. If upstream outputs are available for identical inputs, compare canonicalized nodes/edges and stage tensors, with predetermined numerical tolerance where bitwise equality is impossible. Without upstream output, do not claim output equivalence.
6. Submit the frozen eligible reference through the helper once the operational gates pass. Unknown training membership prevents a clean local validation claim; it does not automatically disqualify a publicly permitted pretrained model. Unresolved rule eligibility does block its use.

Advance when the pipeline is reproducible under the exact Kaggle environment and has a completed submission result or a processing receipt. Preserve it while later experiments proceed. If it fails, identify the failing stage before changing model architecture. Stop repeated reproduction attempts after the reserved pilot budget; save a precise blocker and continue E1 or packaging work.

## E1 — Frozen detector diagnosis

Hypothesis: numerical ties, postprocessing, calibration, or localization error explain a meaningful part of our count inflation, and can be corrected without another fit.

Choose a fixed development panel stratified by embryo, density, intensity, frame position, and clip geometry. Explicitly label the already inspected eight-frame set and the 61 partial outer clips as diagnostic if reused. Include uncapped and capped cases. Derive selection thresholds on development only.

Perform these stages in order:

1. Reproduce the frozen baseline exactly. Compare AMP logits with the original sigmoid, AMP logits with float32 sigmoid, and full float32 inference. Keep tiling, threshold, NMS, and tie handling fixed. Compare logits/probabilities and resulting matched detections; a cast after low-precision sigmoid does not restore lost values.
2. Inspect plateau connected components and tile seams. Use deterministic connected-plateau representatives and deterministic distance suppression in physical coordinates as a separate ablation. Avoid arbitrary top-k ordering of identical scores. Inspect whether a component contains multiple real nearby nuclei before collapsing it. Record plateau volume, distinct peak counts, and spatial displacement.
3. For each chosen precision/tie variant, cache compatible scores. Start a small threshold-by-NMS search, for example five thresholds selected from observed development score distributions and three radii derived from training spacing. Include the frozen control and cap the first grid at 15 configurations per selected variant. One local refinement around a useful frontier is sufficient before freeze; do not run an unbounded search.
4. Compute annotated recall, matched localization distances, predicted/estimated count distribution, pre-cap candidate count, clipped candidate fraction, per-frame and per-clip cap rates, adjusted edge score with the fixed linker, and total runtime. Save a plot/table and inspect matched and unmatched bright regions.

Do not transfer 0.965 from the public model. Do not tune the maximum-node cap as a substitute for a detector; retain it as a disclosed runtime safeguard. Run an uncapped or sufficiently high-cap diagnostic on a small feasible panel to expose truncation, without making the whole inference job unsafe.

Suggested *pilot* advancement rule: at least 20% relative reduction in count ratio against the frozen control, no more than one percentage point absolute annotated-recall loss, improved development adjusted edge score, no unexplained rise in localization error, and no increase in cap frequency. These are conservative campaign defaults, not biological truths or sufficient release evidence. Write any replacement criteria before evaluating candidates. A model can pass this pilot and still be too dense to release.

A candidate with slight recall loss can be better under the actual metric; show that tradeoff rather than optimizing recall alone. If all calibration candidates reduce count only by losing useful edges, stop the sweep and use that result to justify E3. Do not narrate sparse-loss underconstraint as established causation until a matched supervision ablation tests it.

## E2 — Association first, then divisions

Hypothesis: better temporal evidence and valid division handling improve graph quality once node quality is fixed.

Cache one immutable set of detections and node features. Its manifest includes ordered node IDs, coordinates, frame, confidence, model, and preprocessing. Compare a small planned sequence: current greedy linker; physically gated assignment with appearance/disappearance options; forward learned association; forward/reverse fusion; bounded gap repair. Reuse the public implementation before replacing it. Tune on development and evaluate paired clips.

For a harmonic-fusion ablation, after reproducing the reference's normalization and candidate alignment, the schematic combination is `p = 1 / ((1-w)/max(p_f, eps) + w/max(p_r, eps))`. The pinned implementation defines alignment, unmatched mass, epsilon, renormalization, and conversion back to logits. A reverse probability over a different candidate set cannot be fused by array position. Retain missing-edge behavior from the control. Compare the verified reference weight with a small predeclared set including zero; keep detections and solver settings fixed.

Instrument objective terms and graph constraints. In the reported objective, `division_cost - p_second - appearance_cost` is positive when cost is 1.2, appearance is 0, and `p_second <= 1`; an isolated second daughter can therefore be disfavored under those assumptions. Confirm the actual solver revision and all coupled terms before generalizing. Count divisions immediately after the solver and after every postprocessing stage.

Then compare no-division output, existing public division repair, and one targeted alternative. Evaluate parent evidence, two distinct daughters, sister separation, physical displacement, temporal persistence, and competing continuation edges. A fork must not create multiple parents, duplicate daughters, an impossible jump, or later branch merging. Use event-level TP/FP/FN across *all* evaluation clips, including those with no annotated divisions where false positives may still be assessable. Also report the division-bearing subset.

Gap repair must use image-supported candidate nodes and output only temporal structures accepted by the exact competition validator. An advertised gap length is not proof that the branch executes it. Measure interpolation/localization errors; do not invent bridge nodes merely to join components. Inserting nodes makes this a joint detection/association change, so evaluate it separately from fixed-node linking.

Advance an association candidate only with paired official-score improvement and acceptable runtime; report raw edge changes and the count-adjusted contribution separately. Advance a division candidate on a positive total-score delta with pooled event evidence, not simply more forks. A two-point external ablation is motivation, not our expected gain. Preserve worst-case clips and any embryo-specific regression.

## E3 — Training-only teacher and matched distillation

Hypothesis: dense, quality-controlled center supervision reduces the sparse detector's bright-tissue excess more efficiently than extending the same sparse loss.

First confirm external-data/model eligibility and teacher provenance. Apply FOCUS-3D only to the gradient-training partition in each evaluation direction. Do not pseudo-label held-out inputs for a supposedly clean experiment. A teacher with unknown pretraining overlap makes that experiment overlap-unknown even when student labels are generated correctly.

Before bulk labeling, run a small stratified training-only pilot and inspect axial as well as XY views. Measure merged nuclei, fragmented objects, missed faint centers, border truncation, centroid-to-annotated-center offsets, and density differences. Estimate coordinate corrections only from training data, with a separate training-internal QC subset. A segmentation centroid is not automatically the annotated nuclear center.

Convert trusted objects into center targets. Preserve observed annotations. Ignore uncertain/conflicting regions; do not treat all teacher omissions or all unlabeled bright tissue as negative. Keep soft/confidence-weighted targets where supported. Separate the effect of better positives from newly assigned background supervision with a small ablation if affordable. Store teacher weights, preprocessing, object filtering, confidence mask, correction parameters, and all training IDs.

Compare the existing sparse-loss baseline and distilled detector from matched initial weights/seeds, the same real training partition, augmentation schedule, effective batch size, patch/voxel exposure, and optimizer update budget. Also report wall/GPU time and the full teacher-labeling cost. If synthetic or teacher sampling changes exposure, name that difference. Equal epochs alone do not establish equal work. A single paired seed is a pilot; use a second paired seed for promising scale-up if the allocation allows it.

Use E1's count/recall pilot rule plus real development adjusted edge improvement to decide whether to expand. Loss reduction or teacher agreement alone is insufficient. Stop if gains occur only on teacher-like labels, count inflation persists, or labeling cost makes the remaining release budget infeasible. Training and inference must use compatible voxel resampling and coordinate transforms.

## E4 — Optional synthetic association/division pilot

Use synthetic data only for an identified shortfall, such as few mitotic events or rare displacement patterns. The supplied source audit warns that `track_id` is a clone/lineage ID shared by daughters. Build continuation and division targets from graph edges, never equality of `track_id`. Validate the adapter on a known fork and two independent lineages before training.

Fit generator parameters, spacing, motion, image scaling, and event priors on the training partition only. The reported 165,267 events are not a real-data class prior. Compare equal-total-budget real-only training with synthetic pretraining plus real fine-tuning. Report the real-only exposure difference explicitly. Reject a synthetic improvement without corresponding real development gain. Do not replace the project's block-mean preprocessing with the generator's stride sampling without an isolated reason and paired validation.

## E5 — Promote, refit, release

Freeze preprocessing, thresholds, graph settings, and training duration before any final evaluation. Apply `VALIDATION_PROTOCOL.md` and `SUBMISSION_RUNBOOK.md`. If a new model wins, a final all-training-data fit is a separate artifact, not a held-out model; choose its schedule from earlier development results and log its membership. Rehearse that exact artifact. Retain both the best known completed submission and the strongest independently justified alternative when final-selection limits permit.

During the final 72 hours before the verified deadline, prioritize completed candidates and unresolved operational failures. During the final 24 hours, avoid new architecture or expensive training branches unless there is no runnable eligible submission. Schedule the last submission early enough for a conservative full scoring run and recovery attempt; do not aim for the deadline minute.
