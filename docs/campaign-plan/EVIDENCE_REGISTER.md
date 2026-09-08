# Evidence register and open questions

Prepared 2026-09-08 from eight supplied Markdown reports and a limited public-source check. No training, new Kaggle run, scoring query, or rented-host inspection occurred while authoring this pack. All numerical experiment budgets, tolerances, and advancement thresholds elsewhere in the pack are proposed campaign policies unless explicitly identified as measurements.

## Source precedence

Current official rules/account receipts govern execution. A pinned executable scorer governs metric behavior. Actual artifact manifests and complete raw measurements govern experiment conclusions. Dated reports provide historical evidence. Public notebook display values and participant statements motivate experiments but do not replace our reproduction.

The September documents supersede the July documents for implemented capabilities and available real-data evidence. They do not make the July mock measurements real-data measurements.

| Supplied report | Evidence type | Use in this pack |
| --- | --- | --- |
| `status.md` | July 3 scaffold and mock smoke tests | Historical commands only |
| `status_after_first_cv.md` | July 3 mock fold and limited metric probes | Proof of machinery, not model quality |
| `eda_train.md` | Metadata-only `/tmp/biohub_ct_mock_train` | Excluded from biological statistics and split design |
| `baseline_classical_fold0.md` | One mock sample, score zero | Excluded from candidate ranking |
| `leaderboard_hypotheses.md` | Unrun hypotheses | Ideas requiring current evidence |
| `2026-09-05-architecture.md` | Real IO, detector bug, offline rehearsals | Reuse working packaging and specific correctness checks |
| `2026-09-07-learned-submission.md` | Frozen weights, incomplete evaluation, accepted submission | Exact historical candidate identity and limitations |
| `2026-09-07-kaggle-research(1).md` | Six-notebook source audit and recomputed partial real panel | Main experimental starting point |

## Historical measurements to preserve exactly

The following were recomputed in the supplied September 7 audit on the same 61 evaluated clips. They have not been recomputed in this authoring session.

| Measure | Frozen learned | Local classical comparator |
| --- | ---: | ---: |
| Pooled raw edge Jaccard | 0.387766 | 0.339327 |
| Officially weighted adjusted edge Jaccard | 0.221065 | 0.304968 |
| Mean per-clip annotated-node recall | 0.997497 | 0.699391 |
| Mean per-clip predicted/estimated node ratio | 6.390973 | 2.167473 |
| Pooled division TP / FP / FN | 0 / 0 / 22 | 0 / 0 / 22 |

Scope: 61/71 clips in one embryo direction, incomplete OOF; learned model trained on `6bba` and outer evaluation used `44b6`. Detection caps occurred in 43/61 learned clips. Two fits reached 63,564 and 62,577 updates, but only fold0 had a completed threshold-selection record and the reverse outer direction was not evaluated. The frozen checkpoint is step 32,000 with threshold 0.3.

Exact historical identities:

```text
Training source commit: 4587708564490ea582ba3bfdc28ea9b730ef2f52
Original checkpoint SHA256: e83396022fdb2452fbc875fcf134c2426fef464b4e3218e1d50c73c74f03beda
Inference weight SHA256: 2cdce85448f833c80fe0be6d5245ea860124ac508965bb7f4dac1d78c8eb2ad5
Model: PointDetector3D(base_channels=12)
Settings: XY stride 4; tile 64; overlap 16; NMS 3 um; link distance 8 um; cap 2000/frame
Notebook: clarkkitchen/biohub-frozen-learned-submission, version 1
Submission: 56086171, accepted 2026-09-07 23:37:06 UTC
Last supplied processing state: pending
Visible CSV SHA256: db2448f055047364f4a9229bd09b4c6215d0d02d8671630b1b13f57bcbd58a8d
Visible run: 548.80 seconds including setup; four datasets; 1,268,175 node/edge rows
Visible fallback count: zero; three of four clips reached detection caps
```

The September 5 CPU rehearsal and earlier 199-clip projection belong to a different configuration. Do not transfer that runtime to the learned/public pipeline. The historical July public score 0.826 must be reconciled to its actual notebook and source before treating it as the same classical comparator.

The eight-frame precision diagnostic compared original half-precision sigmoid, float32 sigmoid on AMP logits, and full float32. For `44b6_0db75fae` frame 0 it reported 1,340 / 1,241 / 1,204 nodes; frame 50 reported 1,284 / 1,180 / 1,128. Other frames still reached 2,000. This supports a partial numerical explanation, not a complete fix or evidence that dense supervised retraining will succeed.

## Public-method leads from the supplied audit

| Lead | Reported evidence | Unresolved requirement |
| --- | --- | --- |
| [biohub-942tta](https://www.kaggle.com/code/redoctopusk/biohub-942tta) | Code tab displayed 0.946 during September 7 audit | Exact score-associated version, execution parity, own completed score |
| [Pilkwang origin](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-two-seeds-logit-blend) | Two temporal U-Nets and a separate center-evidence artifact shared across forks | Complete checkpoint training membership and effective dependency versions |
| [Division ablation](https://www.kaggle.com/code/rogerrogerroger3r/what-each-piece-of-this-pipeline-is-worth) | Participant reports 0.924 with divisions and 0.904 without | Our paired event/edge ablation |
| [Proxy/ILP report](https://www.kaggle.com/code/rogerrogerroger3r/biohub-the-proxy-is-inverted-the-ilp-can-t-fork) | Three configurations were misranked by a small proxy; division objective issue alleged | Pinned solver objective, broader component evidence |
| [FOCUS discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217) | Dense labeling/distillation lead; direct inference timeout reported by a participant | Teacher QC, cost, overlap, matched student experiment |
| [Synthetic-data discussion](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103) | `track_id` is lineage/clone identity; daughters share it | Edge-derived targets, training-only calibration |

Those notebook sources were inspected in the supplied audit, not downloaded again here. Its three model datasets reportedly declared CC0-1.0, while full training membership was unresolved. Do not infer all dependency/weight permissions from a dataset license or from an origin notebook's title. The support slug/manifest epoch discrepancy is another reason to hash actual artifacts. Forks sharing the same checkpoints do not constitute independent model diversity.

The reported proxy/public pairs were 0.9434/0.937, 0.9418/0.941, and 0.9414/0.942. They demonstrate misranking of those configurations, not a general inversion law. The earlier FOCUS paper could not be fully retrieved in the supplied audit; this pack claims no paper-specific performance result.

## Public-source verification in this authoring session

- [Official competition overview](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview): a fresh indexed official result corroborated the September 29, 2026 final deadline at 23:59 UTC. The opened page yielded no readable body. The [rules page](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules) also yielded no readable body. Daily allowances, runtime caps, and account quota remain runtime checks.
- [Organizer repository](https://github.com/royerlab/kaggle-cell-tracking-competition) and [metric description](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md): read successfully. The documentation corroborates sparse graph scoring, count adjustment, edge-union weighting, pooled divisions, and the additive 0.1 division term. Pin the executable revision in the project; a current `main` page does not establish the competition wheel revision.
- [Kaggle notebook CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) and [competition CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md): read successfully for versioned source retrieval, push, status/output behavior, exact-version submission, and submission-ID status lookup. Installed CLI compatibility still needs checking.
- The reference notebook was found in fresh search results, but its executable content and live score were not revalidated. The FOCUS repository and raw metric URL had retrieval failures; use the supplied audit for historical teacher claims.

## Highest-priority unknowns

1. Current result of submission 56086171 and identity of the best completed account submission.
2. Current rental allocation, numerical remaining authority, provider billing state, and Kaggle quota/reset time.
3. Exact notebook/model/support versions and training membership for the reference.
4. Which truly untouched evaluation data, if any, remain after prior analysis and overlapping crops.
5. Hidden-run feasibility from representative workload measurements, beyond four visible clips.
6. Whether calibration can lower density while retaining useful edges; whether dense supervision adds a further matched-budget gain.
7. Which division-producing stage is effective under the pinned solver objective.

Agents should resolve these from existing records and authorized read-only service access before asking Clark for information already present in the project.

## Supplied-source integrity

The following SHA-256 values identify the attachment bytes used to prepare this pack. They are source-document hashes, not model or dataset hashes.

| Attachment | SHA-256 |
| --- | --- |
| `2026-09-05-architecture.md` | `7180a0d00ec6fc4413dfd7c5f676a68b694e386e726b7e275d18b56677bebfe0` |
| `2026-09-07-kaggle-research(1).md` | `b393bd09efa2c179d59f3ca5b47476690790bf5903b3c68152efa0b3ea37ab2c` |
| `2026-09-07-learned-submission.md` | `f57e875b10dcc88f59222a54a9cac33e8a2b6ee3798b822147576f1d7e7acc7b` |
| `baseline_classical_fold0.md` | `d483a57bda70193ee0c06caafb84aa5e17620227a6b2920abf0c1e316f76898e` |
| `eda_train.md` | `93583d59ac16dc6a0c765d1259bedb2bd8c16673dc2661e3bdfad8d6fe8ac7f8` |
| `leaderboard_hypotheses.md` | `309716cf68232eeacafa213a7d3cbb398466d3f4cf1277a80bcd9b1c94ce97a8` |
| `status.md` | `662e031408bb09cdbd498c547cbbe521ef27cecff5135f5d96c841e32b65aac4` |
| `status_after_first_cv.md` | `4c0dea62c312e3e7db00a6b9f56a43b54047f49954680dbaa1a95cf8a4718bb5` |
