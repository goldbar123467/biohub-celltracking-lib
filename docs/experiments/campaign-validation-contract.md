# Campaign validation and evidence contract

This contract governs campaign metric artifacts and paired candidate decisions. It does
not run the organizer matcher, train a model, start GPU work, or establish that a model
generalizes. Matching must come from the pinned organizer scorer through the existing
official adapter. This layer validates the returned per-clip counts, provenance,
coverage, derived metrics, aggregation, and artifact bytes.

## Authority and scorer behavior

The implementation was checked against all seven September campaign documents and the
current repository adapter plus vendored organizer source. The scorer identity is:

```text
organizer commit: 075fc5f5a52d11077f9dc2b074644618f26939e2
normalized metrics.py SHA-256: cfdd596e3f8909cca14db0682889738b19ff75c3808b3773175aba9367ca7444
```

These values match `biohub_ct.pipelines.evaluate.official_module`. The raw Windows file
has CRLF line endings, so the raw-file digest differs. The official adapter normalizes
CRLF to LF before checking the pinned source digest, and this contract records that
normalized digest.

For each clip, the pinned scorer uses:

```text
U_i = edge_tp_i + edge_fp_i + edge_fn_i
J_i = edge_tp_i / U_i
d_i = (predicted_nodes_i - estimated_nodes_i) / estimated_nodes_i
A_i = max(0, J_i * (1 - 0.1 * d_i))
```

At aggregate level it computes raw edge Jaccard from pooled edge counts and computes
adjusted edge Jaccard as `sum(U_i * A_i) / sum(U_i)` over rows with defined adjustment.
It pools division TP, FP, and FN before computing division Jaccard. If the pooled
division denominator is zero, the division metric is undefined and the scorer drops
the division term from the combined score. The contract preserves that behavior. It
does not use a macro mean of clip scores, apply a count ratio averaged across clips, cap
an underprediction reward, assign a value to an empty Jaccard, or reinterpret sparse
matching.

## Per-clip record

`biohub_ct.campaign.evidence.PerClipEvidence` accepts an exact schema. Missing and extra
keys are rejected. `build_per_clip_evidence` accepts raw fields and derives annotated
recall, predicted-to-estimated count ratio, edge Jaccard, adjusted edge Jaccard, and
division Jaccard. A worker cannot supply those five values to that constructor.

The required identity fields are run and candidate IDs, clip ID, embryo, direction,
frame IDs, evidence class, graph path and digest, and the following provenance:

- candidate, source bundle, dependency manifest, effective configuration, model weight,
  input manifest, split manifest, scorer source, and Git identities;
- exact training and development ID lists;
- training membership, prior-inspection state, and whether the data influenced
  development selection.

A model-free candidate uses `model_weight_sha256: null` plus a nonempty
`model_weight_sha256_reason`. A model artifact uses a lowercase SHA-256 and a null
reason. The four evidence classes are `development`,
`previously_inspected_diagnostic`, `locked_evaluation`, and `overlap_unknown`.
Unknown training membership must use `overlap_unknown`. `locked_evaluation` requires
known exclusion from the training IDs, no development-selection use, and no prior
inspection. A public pretrained model with unresolved membership therefore cannot be
labeled a clean or locked holdout.

Every raw count is a present, nonnegative integer. The record includes predicted,
estimated, annotated, and matched-annotated nodes; pre-cap candidates; capped and total
frames; edge and division TP/FP/FN; forks after solver and postprocess; stage runtimes;
RAM and VRAM peaks; fallback count; and graph artifact identity. Frame IDs are unique
and increasing, `total_frames` equals their count, matched annotations cannot exceed
annotations, and cap/count relations are validated.

Metric JSON always has this shape:

```json
{"value": 0.25, "reason": null}
```

An undefined metric uses JSON null and an explicit reason:

```json
{"value": null, "reason": "division_union_zero"}
```

NaN and infinity are rejected during record parsing and recursive JSON serialization.
When no annotated nodes, estimated nodes, edge union, division union, or matched nodes
exist, the relevant metric stays undefined. This is evidence about the denominator; it
is not silently converted to zero or one.

## Coverage and completion

The aggregation entry point receives the exact planned clip IDs grouped by direction.
It rejects duplicate planned IDs, duplicate evaluated IDs, unexpected IDs, wrong
direction labels, incomplete per-clip records, mixed candidate identities, and
nonuniform provenance within a direction.

`COMPLETE` requires all of the following:

1. Every planned direction is present in the output.
2. Planned and evaluated clip IDs are exactly equal in every direction.
3. Every evaluated record has `evaluation_complete: true`.
4. Every graph artifact exists and its current SHA-256 matches the record.
5. Per-clip identities and metrics pass strict validation.

`PARTIAL` may have missing planned clips and reports them exactly. It still cannot
contain duplicate, unexpected, wrong-direction, or malformed records. An artifact root
supplied to the CLI also verifies available partial artifacts. A partial result is a
different status, not a low-quality synonym for complete.

The aggregate contains overall and per-direction blocks. Each block reports:

- pooled edge TP/FP/FN and raw edge Jaccard;
- official adjusted edge Jaccard with its candidate-owned edge-union denominator;
- pooled division counts and score behavior;
- all-clips division evidence, including false positives on zero-GT-division clips;
- an additional subset whose clips have at least one GT division event, defined by
  `division_tp + division_fn > 0`;
- mean per-clip annotated recall with clip denominator and pooled annotated recall with
  matched-node numerator plus annotated-node denominator;
- per-clip predicted-to-estimated count ratios, mean, min, p50, p90, p95, max, defined
  and undefined clip counts, and a separately labeled pooled ratio;
- cap rates with both frame and clip denominators, prediction/fork totals, stage time,
  resource peaks, fallbacks, and per-clip localization quantiles.

Quantiles cannot be pooled from other quantiles, so localization remains per clip until
raw matched distances are available.

## Paired candidate comparisons

`compare_paired_evidence` requires identical clip coverage, direction, embryo, frame
IDs, count denominators, input and split manifests, scorer identity, evidence class,
training membership, and training/development populations. It rejects a comparison
when any required diagnostic field is absent or malformed because each side must first
pass the per-clip schema.

Control and candidate are expected to have different candidate, source, configuration,
or model-weight identities. Each side must be internally uniform within a direction.
The function recomputes the two aggregates independently, so the candidate uses its own
`TP + FP + FN` sample weights and never inherits the control's weights. Returned deltas
are candidate minus control. An undefined component yields an explicit undefined delta.

## Bundle and CLI

A bundle contains:

```json
{
  "schema_version": 1,
  "status": "COMPLETE",
  "planned_clip_ids_by_direction": {
    "fit_6bba_eval_44b6": ["44b6_..."],
    "fit_44b6_eval_6bba": ["6bba_..."]
  },
  "records": [],
  "aggregate": {}
}
```

`aggregate` is optional. If supplied, the auditor recomputes it from the raw records and
rejects any difference. For a complete bundle, graph hashes are always read back. For a
partial bundle, pass `--artifact-root` to verify the available graph artifacts.

```powershell
python scripts/audit_campaign_evidence.py bundle path\to\evidence.json `
  --artifact-root path\to\campaign-root --output path\to\aggregate.json
```

The loader rejects duplicate JSON object keys and nonstandard NaN/Infinity tokens. An
output file is written through a flushed temporary file and atomic replace, and the CLI
will not overwrite either input file.

## Historical 61-clip audit

The legacy report at
`reports/campaign-backups/longrun-20260906-02/fold0/outer-progress.json` was audited
against `configs/embryo-splits.json` with:

```powershell
python scripts/audit_campaign_evidence.py historical-outer `
  reports\campaign-backups\longrun-20260906-02\fold0\outer-progress.json `
  --split configs\embryo-splits.json --fold fold0
```

The source report SHA-256 was
`92185ac32141c01f5c5bd9596c478de36695cfab97f5046c7fa13cd82bf14fc4`; the split
manifest SHA-256 was
`a24789795d9d21e313074e4918f461388543f2ee42de041035e21f7b331362bd`.
The original fold trained the learned candidate on all 128 `6bba` IDs and planned
evaluation on all 71 `44b6` IDs. Both learned and classical rows cover the same first 61
planned `44b6` clips. Ten planned `44b6` clips are missing, the report's own complete
flag is false, and no reverse `fit_44b6_eval_6bba` evaluation exists.

Recomputation from each candidate's own raw rows gives:

| Measurement | Frozen learned | Local classical comparator |
| --- | ---: | ---: |
| Evaluated/planned clips | 61/71 | 61/71 |
| Edge TP/FP/FN | 11030/10916/6499 | 8322/6996/9207 |
| Pooled raw edge Jaccard | 0.387765864 | 0.339327217 |
| Official weighted adjusted edge Jaccard | 0.221064759 | 0.304967678 |
| Own edge-union weight denominator | 28445 | 24525 |
| Mean per-clip annotated recall, 61-clip denominator | 0.997497292 | 0.699390527 |
| Mean per-clip predicted/estimated ratio, 61-clip denominator | 6.390973134 | 2.167472893 |
| Division TP/FP/FN | 0/0/22 | 0/0/22 |
| Clips with a detection cap | 43 | 0 |

The audit classifies this as `PARTIAL` and
`previously_inspected_diagnostic`. The learned model's original split excludes these
`44b6` IDs from fitting, but the results have since been inspected and influenced the
campaign, so they are not currently locked evidence. This is one whole-embryo direction
and is not two-direction OOF.

Legacy rows lack estimated and annotated count denominators, matched counts,
localization distances, pre-cap candidates, total-frame denominators, stage fork counts,
stage runtimes, resource peaks, graph paths for hash readback, and per-clip binding of
the split to the model artifact. Their annotated recall can only be reported as the
historical mean of ratios. Pooled recall cannot be recovered. Count-ratio quantiles can
be reported from the saved ratios, while a pooled predicted/estimated ratio cannot be
recovered. Per-clip division counts support all-clips and GT-division-bearing subset
summaries, although postprocessing fork diagnostics remain unavailable.

The historical audit therefore does not support `COMPLETE`, a reverse-direction result,
two-direction OOF, a newly locked holdout claim, or full compliance with this new
contract. The local environment used here also lacks the optional organizer package, so
this work does not rerun graph matching. It validates aggregation against the pinned
vendored source and existing official raw outputs. Official matching parity and any new
candidate quality claim require an environment with the pinned dependencies and actual
per-clip artifacts.
