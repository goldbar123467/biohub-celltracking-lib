# Exact-version Kaggle output verification

`biohub_ct.campaign.kaggle_outputs` retrieves output from one positive Kaggle
notebook version through a bounded, read-only SDK request. It exists because the
installed Kaggle CLI 2.2.4 parses the version in `kernels output OWNER/SLUG/N`
but does not copy that version into `ApiListKernelSessionOutputRequest`. Using
the ordinary CLI command could therefore retrieve the latest run while appearing
to name an older version.

The adapter sends these provider fields on every inventory page:

```text
user_name = OWNER
kernel_slug = SLUG
version_label = vN
page_size = configured positive bound
page_token = provider token after the first page
```

The `vN` syntax is an observed provider contract, not an inference from the CLI.
On 2026-09-08, a read-only, one-file inventory request against the existing
`clarkkitchen/biohub-frozen-learned-submission` produced:

```text
v1         PASS  files=1  has_next=true
v999999999 ERROR HTTPError http_status=404
```

No output was downloaded by that live probe, and no signed URL was printed or
stored. Earlier probes showed that `1`, `Version 1`, and `version 1` each returned
404 for the same existing version while `v1` succeeded. The existing-version and
nonexistent-version results establish that the provider request honors the
`version_label` selector. The downloaded run manifest must still bind the
reviewed release identity; a successful transport alone is insufficient.

## API and filesystem contract

Construct `ExactOutputSpec` with an exact owner/slug, positive version, reviewed
release digest, the exact reviewed `reference` object, and finite positive file,
byte, page, and runtime limits. The destination must not exist. The call creates:

```text
DESTINATION/
  downloaded/                                  provider output bytes
  kaggle-output-transport-receipt.json         sanitized transport evidence
  kaggle-exact-version-output-proof.json       exact-version/release proof
```

Pass a literal argv prefix for the Python environment that contains the
authenticated Kaggle SDK. `path_adapter` maps the local download directory into
that process's filesystem. For the current WSL installation, resolve and check
the path with `wslpath` rather than constructing `/mnt/c/...` by string
replacement:

```python
import subprocess
from pathlib import Path

from biohub_ct.campaign.kaggle_outputs import (
    ExactOutputSpec,
    KaggleExactOutputClient,
)


def ubuntu_path(path: Path) -> str:
    result = subprocess.run(
        ["wsl", "-d", "Ubuntu-24.04", "--", "wslpath", "-a", "-u", str(path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
        shell=False,
    )
    converted = result.stdout.strip()
    if not converted.startswith("/") or "\n" in converted:
        raise RuntimeError("wslpath returned an invalid absolute path")
    return converted


client = KaggleExactOutputClient(
    [
        "wsl",
        "-d",
        "Ubuntu-24.04",
        "--",
        "/home/thecl/.local/share/uv/tools/kaggle/bin/python",
    ],
    path_adapter=ubuntu_path,
)
result = client.retrieve_and_verify_outputs(spec, Path("reports/kaggle/e0-vN"))
```

The helper validates a random marker before writing, enumerates the complete
version-specific inventory before downloading, rejects duplicate and unsafe
paths, follows only HTTPS output URLs, and enforces the configured file-count,
per-file, total-byte, and wall-clock bounds. The helper arms its own deadline at
90% of the parent subprocess timeout immediately after parsing its configuration.
On POSIX it uses `setitimer`; on Windows it uses a daemon watchdog. Both call
`os._exit(124)` so provider retry handlers cannot swallow a timeout exception.
This local deadline terminates an already-started Linux downloader even if
timing out `wsl.exe` does not terminate its child. The parent timeout still
bounds launcher and startup waiting; the helper deadline begins only after its
interpreter has started and parsed stdin. The parent process removes the marker,
independently checks every path, size, and SHA-256, and rejects unreported files
or symlinks. It persists hashes of stdout and stderr, not their contents. Signed
URLs remain only in helper memory.

The transport receipt is written on provider errors and timeouts. `ERROR` does
not mean the version is absent unless the sanitized provider evidence explicitly
contains HTTP 404. A timeout or malformed response is simply a failed read and
must not trigger a mutation or an automatic retry loop.

## Proof meaning and independent validation

The proof has schema version 1 and kind
`KAGGLE_EXACT_VERSION_OUTPUT_PROOF`. `status: PASS` plus
`transport_status: VERIFIED` means only:

1. the request used the exact `vN` version label;
2. all reported output bytes were downloaded within the configured bounds and
   independently re-hashed;
3. exactly one `public_reference_run_manifest.json` and `submission.csv` were
   found by basename;
4. the run manifest contains the expected release digest and exact reviewed
   `reference` object; and
5. both manifest CSV hash fields match the downloaded `submission.csv` bytes.

It does not establish output schema, graph validity, coordinate bounds, model
quality, resource fitness, promotion eligibility, or submission approval. The
proof therefore records
`release_validation_status: UNRESOLVED_PENDING_INDEPENDENT_VALIDATOR`.
Pass its raw `submission_path`, `run_manifest_path`, hashes, file inventory, and
the immutable local package directory to the independent public-reference
validator.

`expected_input_shapes_tzyx` is optional because the output transport cannot
derive an independent hidden-input shape contract. If omitted, the proof records
`shape_identity_status: UNAVAILABLE`; bounds validation is not admitted. If the
caller supplies a nonempty `[t,z,y,x]` mapping and a source label, the adapter
requires both run-manifest shape fields to match it exactly and records
`shape_identity_status: VERIFIED`. The source label should identify a reviewed,
authenticated inventory receipt rather than restating the notebook's own
manifest.

Unit coverage is self-contained and uses a fake SDK plus an executed helper
subprocess. The live evidence above is a bounded authenticated integration check;
it was intentionally limited to inventory and did not depend on absent E0 run
outputs.
