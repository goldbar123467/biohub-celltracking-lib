# Campaign full-suite verification

Date: 2026-09-08

Status: **corrected source passed the full test suite on a clean Windows
environment and on the existing Vast Linux worker.** This is software and
operational verification. It does not establish model quality, release
eligibility, or submission approval.

## Failed first synced run

The first synchronized Vast run, `campaign-synced-suite-20260908-01`, preserved
its JUnit evidence under
`work/synced-suite-failure-review/pytest-report.xml`, SHA-256
`c3b8d673f629da17402ad2e633e4e556ba3b4c0e6867a11720b6810812b401d2`.
The report contains 484 collected tests: **441 passed, 21 skipped, and 22
failed** in 38.607 seconds.

The 22 failures were evidence, not a passing rehearsal:

- 21 telemetry and Kaggle-output-operator cases read ignored local R3/R4 package
  directories that were absent from the clean Linux checkout;
- one standalone profiler test inherited the enclosing campaign's `BIOHUB_*`
  identity and exercised campaign mode instead of the intended standalone path.

Review of that run and clean cross-platform reruns produced five bounded fixes:

1. E0 telemetry and output-operator tests now build deterministic temporary
   R3/R4 packages, compute their identities, run the real preflight and release
   validator, and reject tampered source or manifest bytes. They no longer
   depend on ignored private package directories.
2. The standalone profiler test removes inherited campaign identity variables
   before invoking the standalone CLI path.
3. Generic campaign review defers `UNKNOWN` and `RUNNING` Kaggle jobs to the
   exact-version Kaggle reconciler instead of interpreting generic worker
   receipts for them. The Kaggle reservation remains protected while deferred.
4. Artifact retrieval prepares and verifies overlapping destination directories
   serially before parallel downloads. This addresses intermittent false
   containment rejections on Windows by removing concurrent `resolve`/`mkdir`
   calls on overlapping parents, while retaining symlink and containment
   checks. The exact operating-system mechanism was not independently isolated.
5. The synthetic output-operator subprocess adds both the repository root and
   `ROOT/src` to its child import path. This makes the real CLI test independent
   of an editable installation in the parent environment.

## Root-clean Windows verification

The root reran the full suite on native Windows in an environment provisioned
independently with `pytest` and NumPy through `uv`, rather than relying on the
repository's editable test environment. The final JUnit receipt is
`work/root-suite-fixed-20260908-v2.xml`, SHA-256
`87f5dd6cd3f10e678491ccabe8c3e83cb56427e2de547030246fc4a3a88da690`.
It records **405 passed, 61 skipped, 0 failed, and 0 errors** in 13.139 seconds
(13.20 seconds in the command summary). The skips are explicit optional
dependency and platform cases; they are not counted as passes.

## Vast Linux verification

The corrected synchronized run was `campaign-synced-suite-20260908-02`. Its
immutable source identity was:

| Field | Value |
|---|---|
| Git commit | `44f8e5303db317dbe3274f9e02183e734dd3662e` |
| Git tree | `10fc3410eee951820b8000d3e2aba66234b7c791` |
| Run-spec SHA-256 | `e6d816207538c0bf2cb39a4d16d094d07b2ad2d0b6e224fa42bf8b8b1e87e290` |

Pytest reported **469 passed, 21 skipped, 0 failed, and 0 errors** in 38.30
seconds. The application wrapper measured 40.490763460984454 seconds and the
supervisor completion measured 41.06784494704334 seconds. These are separate
timing scopes and should not be substituted for one another.

The downloaded completion is
`reports/campaign-downloads/campaign-synced-suite-20260908-02/reports/campaign-workers/campaign-synced-suite-20260908-02/worker/completion.json`,
SHA-256
`5cb59983c4bb67fb871efbc79869082a0eabae1ed8f03ab4391399b5a9a36a07`.
The root independently rehashed all four declared worker artifacts and the
result manifest; every digest matched the completion record:

| Artifact | SHA-256 |
|---|---|
| `progress.json` | `771996ca43ace5a1083ae66dd1140e1f787f166df26bc9751df6d840a7713cc4` |
| `pytest-report.xml` | `9f69fb2517959d29d542cd7495dfecc752d2bb5542bf1f717bad8a582cc891e8` |
| `pytest.log` | `330a9af75a362a0f23d2c7d7bacbb4717455d29abc216632ed8d6f66d1a8fa84` |
| `suite-summary.json` | `5f6def2745afbf77eeaabf0d7caf4ea608a078ba5493dd65980507e4c9cd0ee3` |
| `result.json` | `33d4b66361bb8d085a61f74c845a2f0b2c0bd6b5d366085c5edc3e28507b8754` |

Canonical review `review-20260908T101926649996Z`, fencing token 60, finalized
the run as `COMPLETE`. It recorded `settlement_needed=false`; no run reservation
remains. The existing Vast operational-verification ledger now records:

| Quantity | Instance hours |
|---|---:|
| Authorized total | `0.25` |
| Confirmed spend | `0.1443081781594703620833333333` |
| Available | `0.1056918218405296379166666667` |
| Outstanding reservations | `0` |

The passing suites verify the synchronized software and its tested campaign
contracts on Windows and Linux. They do not provide a new model evaluation,
held-out metric, Kaggle output result, or model-quality advancement decision.
