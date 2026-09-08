# E0 full instrumented Kaggle differential

Date: 2026-09-08

The private R4 provider execution completed successfully, and its 241,400-row
submission is byte-identical to the frozen R3/reference CSV. The required
telemetry contract **failed**: submission serialization was not measured.
Successful execution and output parity do not turn this into an accepted
telemetry rehearsal or authorize a submission.

## Exact run and retrieved evidence

The single private, offline push was
`clarkkitchen/biohub-e0-instrumented-reference/1`, provider run ID `348176862`,
using the frozen `work/e0-reference/package-r4-title-fixed-a` release digest
`41b2810b88edd3b61a4bed8b3f32d1a0df3454af463f20520473d96bb6244353`.
The provider reported success after **5,706.2 seconds**, with 465 output files.
The subsequent authenticated active-events view showed zero active events.

All evidence below is under
`reports/campaigns/e0-r4-kaggle-telemetry-differential-20260908-01/`.
The exact-version transport retrieved **465 files, 54,932,337 bytes**. Its
successful receipt is
`download-validation-v1/transport/kaggle-output-transport-receipt.json`, SHA-256
`8f3edc21be11dd07d8bd2f1d82b639a613167b4e45a907cf20748d4a7f656f15`.

The first outer validation attempt failed while constructing the proof because
ordinary Windows filesystem access could not resolve a downloaded path longer
than 260 characters. The original failed result remains unchanged at
`download-validation-v1/validation-result.json`, SHA-256
`f384fa49badc5134c111653a349319bf4087ddd1a8ce68d9c78df690ae3c1644`.
A bounded access fix uses extended absolute Windows paths consistently while
preserving relative inventory identities and containment checks. The root
rehashed every original downloaded file and recovered the proof locally without
another provider request or download. The recovery receipt is
`output-proof-recovery-v1.json`; the recovered exact-version proof has SHA-256
`c9b3c3891e1761e13baa37028a097586f798ce3c51ef79bad1189ebe5c84a5f3`.

A fresh independent local validation completed at
`independent-validation-v1/validation-result.json`, SHA-256
`b7e89965e911a752c501535ecd57007766121812c3c29c03038c7deb1d3e7a3c`.
Identity, official format, scorer compatibility, and E0 lineage checks passed.
The inherited upstream release manifest still leaves quality and resource
admission blocked; the separate telemetry evidence does not silently rewrite it.

## Output comparison

`r3-vs-r4-reference-comparison.json` reports exact dataset sets, node identities,
node values, edge identities, and byte equality. Both CSVs have SHA-256
`a852d1d07ff8c9307d9b10db7f9b4b12e8b1882f14c5dbeb1316d099f0795b3e`.

| Dataset | Final nodes | Final edges |
|---|---:|---:|
| `44b6_0113de3b` | 25,636 | 24,927 |
| `44b6_0b24845f` | 20,754 | 19,464 |
| `6bba_05b6850b` | 6,150 | 5,956 |
| `6bba_05db0fb1` | 70,301 | 68,212 |

The separate detector-coordinate and logical GEFF comparison also passed:
`r3-r4-graph-evidence-comparison-v1.json`, SHA-256
`01c29f1e2103d5cb923aee28419d591260a66caf1f2460cbf00074f49c0438ce`.
All 12 datasets (four production and eight validation) have identical coordinate
hashes, row counts, and dense 100-frame counts. All 12 GEFF graphs have exact
logical array/attribute equality, including node IDs, edge endpoints and the
exact floating-point bits of stored probabilities. No numerical tolerance was
used. The root independently exercised a mutated real coordinate artifact and
verified that its inconsistent frame counts were rejected.

R4's enriched coordinate manifests include raw-artifact identities. Every raw
artifact was checked against the exact-version proof, rehashed, decoded as
little-endian int16 TZYX, and used to recompute frame counts. R3 retained only
coordinate hashes/counts; its solved GEFF graphs have fewer nodes than the
pre-ILP coordinate manifests and cannot reconstruct those full raw coordinates.
Thus the R3 side of pre-ILP comparison remains hash/count evidence. Raw manifest
file bytes differ because of R4's added artifact metadata; their normalized
coordinate records match exactly.

## Failed telemetry acceptance

The unchanged strict downloaded-telemetry verifier was executed against the
frozen package and the original output tree. It completed package identity,
offline re-harvest equality, manifest binding, cell/source coverage, callback
cleanup, and resource checks, then raised
`TelemetryVerificationError: submission serialization timing was not observed`.
It did not write a success report. The root reproduced and persisted the failure
in `telemetry-failure-audit-v1.json`, SHA-256
`491aa8ae481fcdad6f54eec8e17aa58f977bb3d43d40d60b5eedde38dcec63a6`.
All 24 relevant telemetry input files were rehashed before and after that check.

The frozen notebook writes `submission.csv` with `csv.DictWriter`, while the
frozen timer wraps only `pandas.DataFrame.to_csv`. Consequently the serialization
record has `status: UNAVAILABLE` and an empty record list. Timing the entire
submission function retrospectively would mix graph postprocessing with CSV
writing and would not recover the missing measurement. The raw notebook's
top-level `PASS` does not include this acceptance requirement.

A separate invocation of the verifier's strict stage/process check passed all
eight recorded stages, all four production datasets, and both production shard
processes. This result is preserved inside the failure audit and independently
cross-checked. It does not change the failed overall verdict.

## Measured scopes and limits

| Measurement | Observed value | Scope |
|---|---:|---|
| Provider elapsed | 5,706.2 s | External successful version display |
| Wrapper elapsed | 5,686.044763595 s | Prepended telemetry cell through final harvest/cleanup boundary |
| Existing manifest elapsed | 5,684.728857719 s | Original notebook timing boundary |
| Resource samples | 1,116 | Five-second whole-wrapper sampling; no dropped samples or write errors |
| Sampled host RSS peak | 8,135,196,672 bytes | Sampled process tree |
| Sampled GPU 0 used peak | 1,120,927,744 bytes | Device-wide sampling |
| Sampled GPU 1 used peak | 1,471,152,128 bytes | Device-wide sampling |
| Submission serialization | Unavailable | Required measurement missing |

The 16 observed cells comprise 12 public cells and four instrumentation cells;
all completed with matching pre/post source hashes and no coverage violations.
Both support launches and callback cleanup passed. Process-emitted CUDA
allocation/reservation and `ru_maxrss` records are distinct from device-wide
sampled maxima. Samples are not continuous high-water marks and may include
unrelated device allocations. Stage durations exclude model loading, gaps
between stages, and notebook overhead.

R4's provider time exceeded the single R3 execution by 629.9 seconds, or 12.41%.
This is an observed difference between one pair of executions, **not an estimate
of causal instrumentation overhead**. Provider/kernel boundary differences,
execution variability, and asynchronous work are not controlled by this pair.
No hidden-test runtime guarantee or held-out model-quality result follows.

## Quota and next gate

The authenticated account display moved from 1.59 to 3.17 used hours and from
28.41 to 26.83 remaining hours, with the same reset and 30.00-hour total. The
observed debit is therefore **1.58 quota hours**, rounded to the display's 0.01h
precision. The 26.40-hour reserve for two final attempts remains intact.
At 10:53:43 UTC the canonical run was finalized **FAILED** for the missing
telemetry requirement while retaining `provider_state: COMPLETE`. The 1.98-hour
reservation was settled at 1.58 hours; `settlement_needed` is false and no run
reservation remains. `terminal-reconciliation.json` has SHA-256
`0cb5ec9a1e3a5ca95a0b8225671037634f557deaa70a353350d0d6f7659e6c18`.
An independent store read confirmed 3.17 hours spent, zero outstanding
reservation, 26.40 protected hours, and 0.43 available hours.

The future runtime fix measures actual `csv.DictWriter` serialization/write
calls, preserving the frozen R4 package and its failed result. Regression checks
verify byte identity, the `f=` constructor API, return values, exceptions,
cleanup, row/call accounting, and exclusion of graph computation between calls.
Lazy iterable production inside `writerows` is included and explicitly labeled.
The root's focused suite passed 74 tests. The independently provisioned Windows
full suite passed 409 tests with 61 explicit skips in 14.10 seconds, using
`python -m pytest`; its JUnit evidence is
`work/root-suite-r4-fixes-20260908-v2.xml`. An earlier direct `pytest` invocation
failed collection because its entrypoint omitted the repository root from the
import path; that failed receipt was preserved. A corrected full Kaggle telemetry
rehearsal remains unexecuted. The remaining unprotected display balance is only
0.43 hours, which cannot admit another rehearsal with the measured duration and
required reserve. No retry, submission, recurring activation, or new allocation
is implied by this report.

The final focused Linux WSL check passed 73 tests with one Windows-only skip in
2.24 seconds, using the current telemetry, offline verifier, package and output
modules. JUnit hashes are
`a1fb51e453f8bf69f7bbe9b0289954bad75627a2848037aeee2ad4130c074fd2`
for the native Windows full suite and
`ea105a14f5a076d37a6b13d8b2cb7eeced71bd5b108ce1dc89ea310cbfe0e264`
for `work/root-r4-fixes-wsl-focused-20260908.xml`. This is local CPU verification;
the earlier full Vast Linux/CUDA suite remains separately scoped to its recorded
source revision in the [suite report](campaign-suite-verification-2026-09-08.md).
