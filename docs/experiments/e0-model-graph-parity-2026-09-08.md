# E0 real-model support-graph parity

Preregistered 2026-09-08. Status: executed and independently verified PASS for
the bounded support-path comparison described here.

## Question and boundary

Does the additive support telemetry preserve the actual dual-model detector,
association, candidate graph, ILP result, and GEFF content on identical visible
input? This is implementation verification on the existing RTX 4070 SUPER. It
does not evaluate model quality or admit the instrumented notebook for release.

The input is `44b6_0113de3b`, frames 0 through 7, with full raw spatial dimensions
64 x 256 x 256. The public downsampling, normalization, TTA, detection threshold,
secondary and bidirectional fusion, and ILP settings remain those of the pinned
public support source. Both comparison arms each use both public checkpoints.
The effective pooling kernel size is the support `PredictConfig` default of 3 microns;
the notebook launch does not override it. The checkpoint configuration's 5-micron
value, recorded by the earlier feature-only profiler, is not this runtime setting.

The secondary model's pinned split manifest lists this clip in both `train` and
`test`. Its membership is therefore known training overlap, regardless of the
test label. Primary membership remains unresolved. This comparison cannot be
used as a clean holdout, OOF result, or selection experiment. The secondary
manifest SHA-256 is
`cbe8ace34ffc157172280538441454b60250f0188faa063d1a9eadfb1ac55c0b`.

## Immutable inputs and admission

The control support source SHA-256 is
`49613ad0b50ac90c3e07e3f8a803f2adf0926203be3f4a97d577c60b8f756178`.
The second arm applies the current additive telemetry patch to those same bytes.
The staged run specification binds the entrypoint and supervisor source,
checkpoint/configuration files, dependency metadata, raw frame hashes and Zarr
metadata, and the operational-only provenance statement. Input hashes must
match before and after the comparison.

One shared application deadline is 300 seconds. The independent supervisor
deadline is 360 seconds with 15 seconds of shutdown grace. The reservation is
0.131 instance-hours, or 471.6 seconds, covering `(360 + 15) * 1.25 = 468.75`
seconds. This fits the observed 485.30049 seconds remaining in the existing
implementation-verification ledger. It does not enlarge that ledger or grant
routine campaign spending. The complete successful runtime is unknown.

The existing provider environment is recorded as used. Its tracksdata, GEFF,
NumPy and Zarr versions differ from the pinned Kaggle wheels; no exact Kaggle
environment claim follows from this test. No dependency installation is part
of the attempt.

## Predetermined pass criteria

- Both fresh comparison arms execute the two real models and all support-path
  detection and association stages on the same eight frames.
- Every frame has detections; each candidate graph has edges; an actual ILP
  solve runs and returns. At least one solved connected component spans six
  frames.
- Ordered detector coordinates and candidate edges, including probability and
  distance bits, match exactly. No tolerance is widened after observation.
- Canonical pre-ILP and solved graphs match in nodes, coordinates, edge
  endpoints, attributes, and topology.
- Semantic GEFF rereads match each other and their corresponding solved
  in-memory graph. Serialized directory bytes alone are insufficient evidence.
  For the in-memory-to-GEFF comparison only, storage dtypes and internal
  `edge_id` values are presentation details when numeric values survive exactly;
  node identities, time, coordinates,
  edge endpoints, and all substantive attributes must survive. Full graph schemas
  and GEFF array dtype inventories are still compared between the two arms.
- Downloaded evidence permits independent recomputation of the comparisons.

A timeout, empty graph, skipped solver, missing evidence, input change, or
insufficient temporal span cannot produce a parity PASS. Failed or incomplete
comparisons remain explicit; there is no automatic retry or scope reduction.

## Remaining release scope

This support-path comparison excludes DeepCenter and the later notebook gap,
division and minimum-track-length postprocessing, complete visible competition
coverage, CSV generation, full notebook telemetry, Kaggle execution, quota-debit
measurement, and submission/score receipts. These requirements remain open even
if the bounded support-path comparison passes.

## Executed result

The single admitted run `e0-model-graph-parity-20260908-01` completed at
06:31:40 UTC with exit code 0. Its shared application took 21.2565 seconds;
the independently supervised worker took 22.6556 seconds. The frozen run-spec
SHA-256 is `3ebdada5c421af58f9a247bcdb75200859b777f082bed577d0bdc713855e6b19`,
with fencing token 48. The reviewed verifier source SHA-256 is
`94d76798b158a038ba764247b3a749a7997458975bc47edfff7f2ffe05392ee4`.

| Quantity | Control | Instrumented |
| --- | ---: | ---: |
| Detected nodes across eight frames | 1,799 | 1,799 |
| Candidate edges before ILP | 1,527 | 1,527 |
| Solved graph nodes | 1,689 | 1,689 |
| Solved graph edges | 1,465 | 1,465 |
| Maximum connected unique frames | 8 | 8 |

Coordinates and all ordered candidate arrays matched byte-for-byte. Canonical
pre-ILP, solved and reloaded graphs matched between arms. An independent local
checker, which did not import the production comparator, verified all 82
archived files, loaded the NPZ candidates, checked graph construction against
them, decoded the raw GEFF arrays with Zarr, and matched every serialized node,
edge and substantive attribute to the solved graph. All logical GEFF arrays and
group attributes also matched between arms.

Every frame contained detections: 224, 224, 226, 223, 224, 222, 226 and 230.
Both solved graphs satisfied the checked temporal adjacency, parent and child
limits. The three input snapshots agreed in all eight uint16 frame hashes and
both Zarr metadata hashes. Each full raw frame contained 8,388,608 bytes.

The instrumented invocation recorded all eight inner stages: data read 0.5326s,
encode/TTA 3.2763s, detector extraction 0.0131s, pair scoring 0.3900s, thresholding
0.0704s, graph construction 0.0047s, ILP 0.8204s, and GEFF serialization 0.2477s.
CUDA allocator peaks since telemetry attachment were 821,379,584 allocated bytes
and 1,080,033,280 reserved bytes. The process-wide host RSS high-water mark was
1,852,940,288 bytes and can include the preceding control arm. These are distinct
measurement scopes. The sequential, single-observation arm timings do not prove
instrumentation overhead or a performance gain and cannot be extrapolated to a
full Kaggle release duration.

The controller reconciled the terminal result in review
`review-20260908T063222554571Z`, verified downloaded artifacts and settled the
reservation. The implementation ledger now records 0.1214875384700174
instance-hours spent, zero outstanding reservations, and 0.1285124615299826
instance-hours available. These are supervised implementation-worker times,
separate from whole-instance provider billing and any future campaign allowance.

Evidence is preserved under
`reports/campaign-downloads/e0-model-graph-parity-20260908-01/reports/campaign-workers/e0-model-graph-parity-20260908-01/`.
The parity receipt SHA-256 is
`bfb52b6d4fc8db1623b4859f30f983583990e6d0e5ca941fb6fff8207a3be90e`;
the 10,332,160-byte artifact archive SHA-256 is
`56a54ba073c898b948eecdbf4b4b5d4d71ddcd9a44fcee86f90b80e0c9fa4f10`.
The root's independent recomputation is `work/pv9a/root-receipt.json`, produced
by `work/verify_e0_parity_download.py`. Four focused comparator tests passed on
native Windows; compilation and Ruff checks passed. No additional model fit,
Kaggle push/submission, schedule activation or notebook release admission was
performed in this checkpoint.
