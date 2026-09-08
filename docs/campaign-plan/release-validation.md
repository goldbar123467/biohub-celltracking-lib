# E0 downloaded release validation

`biohub_ct.campaign.release_validation.validate_downloaded_e0_release` is the
independent check for rehearsal check 5 in `SUBMISSION_RUNBOOK.md`. It is a
read-only library boundary. It does not download Kaggle output, approve a
candidate, mutate campaign state, or submit anything.

The September 8 full R3 rehearsal now has executed evidence: all 442
exact-version output files were downloaded, and the independent identity,
format, scorer-compatibility and lineage checks passed. The 241,400-row CSV
is also byte-identical to the upstream reference. Quality and resource
admission remain separate: the frozen upstream manifest omits required
scientific and resource fields, so those gates remain blocked.

## Inputs and trust boundary

The validator takes the downloaded `submission.csv`, the downloaded
`public_reference_run_manifest.json`, the reviewed package directory, and an
`ExactVersionOutputProof`. The proof must come from the authenticated,
version-qualified output transport. Its stable fields are:

```text
schema_version=1
kind=KAGGLE_EXACT_VERSION_OUTPUT_PROOF
status=PASS
transport_status=VERIFIED
notebook_slug, notebook_version, version_label=vN
release_digest
run_manifest_sha256, submission_sha256
embedded_release_identity_verified=true
embedded_reference_verified=true
embedded_submission_hash_verified=true
output_files=[{path, basename, bytes, sha256}, ...]
expected_input_shapes_tzyx={dataset: [T,Z,Y,X], ...}
shape_identity_status=VERIFIED
release_validation_status=UNRESOLVED_PENDING_INDEPENDENT_VALIDATOR
```

The transport proof authenticates which exact notebook version produced the
downloaded bytes. This validator does not accept current-slug output, an
unversioned status response, or a shape map supplied without that proof.

Output paths are normalized before inventory uniqueness and basename checks.
Distinct Zarr directories may each contain `zarr.json` or `0`; duplicate full
paths and separator aliases are rejected. The required submission and manifest
must each appear exactly once across the inventory, with exact hashes and byte
counts. Nested copies cannot hide behind a caller-supplied basename. This
behavior passed 47 focused release/output tests and independent review.

The reviewed package is rechecked with `preflight_package`, including all four
package file hashes and the Kaggle CLI-normalized notebook hash. The validator
also recomputes the artifact-lock release digest, binds the package manifest to
the artifact lock, checks the preserved upstream source identity, and compares
the run manifest's full `reference` object with the reviewed artifact lock.
Each mounted input must have the exact reviewed dataset reference, numeric
version, file count, total bytes, and a unique absolute mount root. The notebook
instrumentation already hashed every mounted file before the public code ran;
the exact release digest and exact-version transport bind that check to this
validation.

## CSV and graph checks

The validator hashes the raw CSV bytes and requires the same digest in the
transport proof and both manifest locations. It then streams each contiguous
dataset block and checks:

- the exact ten-column header and a canonical integer in every numeric field;
- global consecutive `id` values and exact discovered dataset coverage;
- node and edge sentinels, nonnegative node IDs, and per-dataset unique node IDs;
- all node coordinates against authenticated per-dataset TZYX shapes;
- edge endpoints in the same dataset, unique endpoint pairs, and valid time direction;
- the E0 lineage contract of one-frame edges, at most one parent, at most two
  children, and two distinct next-frame children for every predicted fork;
- exact row and per-dataset node/edge counts against the run manifest.

The repository's `biohub_ct.submission.validator` runs as a second
implementation after the independent streaming pass. The CSV and manifest are
hashed again afterward to detect changes during validation.

The Kaggle submission format itself specifies the header, row sentinels,
consecutive index, folder-name dataset IDs, and complete dataset coverage. The
pinned organizer converter accepts graphs without imposing lineage degrees.
The pinned scorer discards edges whose target is not exactly one frame after the
source, deduplicates repeated endpoint pairs for scoring, and keeps only two
outgoing edges per source. Therefore, the one-frame, unique-edge, indegree, and
outdegree checks above are the stricter frozen E0 release contract. They prevent
silent scorer filtering and ambiguous lineage topology; this document does not
represent every one as a Kaggle CSV parser prohibition.

Sources checked on 2026-09-08:

- [official competition overview and submission format](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview)
- [pinned organizer CSV reader](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/scripts/csv_to_geffs.py)
- [pinned organizer scorer](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/src/tracking_cellmot/metrics.py)

## Result and admission boundary

A successful call returns `ReleaseValidationResult` with identity, raw hashes,
dataset shapes and counts, fork counts, degree maxima, and the measured full
runtime. `official_format_status`, `scorer_compatibility_status`,
`e0_lineage_contract_status`, and `identity_status` are separate fields.

The returned `full_runtime_seconds` preserves the R3 wrapper's field name and
scope. Its timer starts in the prepended integrity cell and is sampled in the
final validation cell before the manifest is written. It excludes manifest I/O,
notebook/provider shutdown, output collection, and any provider tail. It is not
the actual provider total, a conservative runtime bound, or runtime headroom.

The R3 wrapper does not bind pre-cap candidate counts, cap numerators and
denominators, output-fallback/retry counts, inner stage times, peak RAM/VRAM, an
authenticated runtime limit and headroom calculation, or measured quota debit
into its manifest. Missing fields are listed explicitly. If similarly named
fields later appear, this validator still returns `NOT_EVALUATED` for the
relevant admission because presence alone does not establish units, provenance,
semantics, or acceptance thresholds. Quality and resource admission require a
separate audited decision.

The public source's retention fallback chooses the primary detector when a
blend-retention guard fails; it is not a fabricated output node. Empty output
fails closed. Frame-cap branches are silent, so the current run manifest cannot
support a zero-cap claim.

There is no downloaded E0 rehearsal output at this checkpoint. Unit tests use
clearly synthetic, self-contained fixtures to exercise the contract. Passing
those tests proves validator behavior, not operational E0 completion.
