# Kaggle reference output operator

[`scripts/verify_kaggle_reference_outputs.py`](../../scripts/verify_kaggle_reference_outputs.py)
is the concrete read-only entrypoint for reviewed E0 output review. It combines the
exact-version transport in `biohub_ct.campaign.kaggle_outputs` with the
independent CSV, manifest, package, and lineage checks in
`biohub_ct.campaign.release_validation`.

The operator has no push, submit, approval, campaign-store, or scheduler code.
It writes only inside a new `--destination`. A successful operator exit means
the independent validator completed. It does not change the validator's quality
or resource admission status and does not authorize submission.

## Validate existing local evidence

Local validation is the default. Supply the persisted exact-version proof and
the exact CSV and run-manifest bytes named by that proof. The package defaults to
the reviewed immutable `work/e0-reference/package-r3`:

```powershell
$python = 'C:\Users\thecl\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python scripts\verify_kaggle_reference_outputs.py `
  --destination reports\campaigns\e0-r3-v9-validation `
  --proof-json reports\downloads\e0-r3-v9\kaggle-exact-version-output-proof.json `
  --submission-csv reports\downloads\e0-r3-v9\downloaded\submission.csv `
  --run-manifest-json reports\downloads\e0-r3-v9\downloaded\public_reference_run_manifest.json
```

The destination must not exist, and its parent must already exist. The operator
rejects symlinked inputs, a destination inside the immutable package, an
existing result, and its deterministic temporary-name collision. It flushes a
complete strict JSON temporary file, then uses an atomic no-replace hard-link
install for `validation-result.json`.

The default `--generation r3` selects `E0_R3_PACKAGE_IDENTITY`. Explicit
`--generation r4` selects the independently verified instrumentation candidate
`E0_R4_PACKAGE_IDENTITY` and defaults to `work/e0-reference/package-r4-title-fixed-a`.
There is no CLI option for an arbitrary digest, notebook slug or package identity.
A different `--package-dir` is accepted only when all bytes match the selected
compiled identity. The R3 canonicalization proof cannot substitute for R4 evidence.

## Provider-generated notebook URL

Kaggle may generate the notebook URL from its title even when the upload metadata
requests a different slug. The September 8 R3 attempt encountered this behavior.
It does not make arbitrary notebook URLs interchangeable.

For that case only, `--canonicalization-receipt PATH` supplies the verified
read-only reconciliation proof from `biohub_ct.campaign.kaggle_canonicalization`.
The proof binds the original launch record, captured push-transcript hash,
title-derived URL, exact provider version, downloaded normalized notebook source,
and private/offline operational metadata. The operator verifies the proof before
retrieval and again afterward. It uses the mapped URL and exact version for
transport while continuing to preflight the original frozen R3 package bytes.
The independent release validator consumes the same proof; version or source
drift fails. Without this explicit verified proof, the original slug check stays
strict. This option does not select R4 or change release admission.

## Download one exact version, then validate

`--download` performs a read-only exact-version inventory and output download
before local validation. Every live version, bound, SDK path, and path mapping
must be supplied. Local `--proof-json`, `--submission-csv`, and
`--run-manifest-json` options are forbidden in this mode.

The independently reviewed shapes contract has this exact schema:

```json
{
  "schema_version": 1,
  "kind": "BIOHUB_INDEPENDENT_INPUT_SHAPES",
  "status": "VERIFIED",
  "source": "authenticated hidden-input metadata inventory 2026-09-08",
  "source_receipt_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "shapes_tzyx": {
    "actual-dataset-id": [12, 30, 512, 512]
  }
}
```

`--input-shapes-source-receipt` must name the strict JSON receipt whose raw
SHA-256 is `source_receipt_sha256`. This keeps the shape bound independent of the
notebook's own run manifest. Placeholder provenance such as `TODO`, `unknown`,
or `unresolved` is rejected. The operator hashes both evidence files and binds
their provenance into the exact-version proof.

Resolve the Windows destination with the same WSL distribution before invoking
the operator. The value passed to `--destination-sdk-path` is the expected WSL
mapping of the new destination root. The operator independently runs `wslpath`
after creating the root and again for the transport child directory; either
mismatch blocks the SDK call.

```powershell
$python = 'C:\Users\thecl\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$destination = (Resolve-Path reports\campaigns).Path + '\e0-r3-v9-download-validation'
$destinationForWsl = $destination.Replace('\', '/')
$destinationSdk = (& wsl -d Ubuntu-24.04 -- wslpath -a -u $destinationForWsl).Trim()
& $python scripts\verify_kaggle_reference_outputs.py `
  --destination $destination `
  --download `
  --notebook-version 9 `
  --input-shapes-json reports\reviewed\e0-v9-input-shapes.json `
  --input-shapes-source-receipt reports\reviewed\e0-v9-input-inventory.json `
  --max-files 8 `
  --max-file-bytes 1073741824 `
  --max-total-bytes 2147483648 `
  --max-runtime-seconds 900 `
  --page-size 8 `
  --wsl-distro Ubuntu-24.04 `
  --kaggle-sdk-python /home/thecl/.local/share/uv/tools/kaggle/bin/python `
  --destination-sdk-path $destinationSdk
```

The numeric values above illustrate explicit syntax. They are not recommended
limits for a future run. Select reviewed bounds from expected output size and
the available operator window. The exact transport also retains its independent
helper deadline, byte limits, complete-inventory-before-download rule, HTTPS
restriction, path confinement, and local SHA-256 recomputation.

Download artifacts are placed under:

```text
DESTINATION/
  transport/
    downloaded/
    kaggle-output-transport-receipt.json
    kaggle-exact-version-output-proof.json
  validation-result.json
```

The operator reloads the persisted proof and requires byte-for-byte semantic
equality with the transport return value and its SHA-256. The proof, receipt,
CSV, and run manifest must resolve inside the new transport directory before the
independent validator runs.

## Result interpretation

`validation-result.json` uses schema version 1 and kind
`KAGGLE_REFERENCE_OUTPUT_OPERATOR_RESULT`.

- `status: VALIDATION_COMPLETE` means exact inputs reached the independent
  validator and it returned `identity_status: PASS`.
- `release_validation` is the validator's complete result. Expected R3 output
  can still report `quality_admission` or `resource_admission` as blocked.
- `status: ERROR` records the failed stage, exception type, and a digest of the
  exception representation. Arbitrary exception text is omitted so a provider
  URL cannot enter the result.
- The result explicitly records that no mutation, submission, approval, or
  campaign-store write occurred.

Tests cover the injected local and download flows, missing and conflicting
options, shape provenance, WSL mapping mismatch, result collision, sanitized
failure, and one real CLI subprocess that validates a synthetic CSV and manifest
against the actual immutable R3 package.

## Future reviewed package generations

The title-fixed R4 has an immutable compiled identity and explicit selector;
its two actual builds, all 12 original public cells, package hashes and command-line
preflight were independently verified. Operator tests reject an R3 validator
identity when R4 is selected. This is permission to identify reviewed source in
a bounded rehearsal workflow, not release admission. A future generation needs
its own reviewed immutable identity, generation selector and equivalent checks.
No generation can be introduced through a caller-supplied identity JSON.
