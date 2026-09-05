# Testing and offline submission

Use the same source and frozen configuration for validation and Kaggle. The
classical detector is an executable baseline. Learned detectors and linkers remain
experiments, not trained models supplied by this change.

## Environment and checks

```bash
python -m pip install -e '.[dev,kaggle,official-metric]'
python -m pytest -q
```

The official extra pins both organizer code and TracksData to commits. Evaluation
also checks the normalized SHA256 of the metric implementation before scoring.
The existing Vast environment has these packages. Dependency-light environments
skip optional real-IO and official-parity tests explicitly. CI checks core and
real Zarr IO on Linux and Windows; official parity is verified on Vast.

## Frozen whole-embryo evaluation

`configs/embryo-splits.json` contains all 199 train IDs from the authenticated file
inventory: 71 clips from `44b6`, 128 from `6bba`. `fold0` holds out `44b6`; `fold1`
holds out `6bba`. The adjacent provenance JSON contains the split checksum.
Evaluation rejects missing IDs and embryo overlap, so partially downloaded data
cannot silently shrink a full fold.

After the download finishes and `scripts/verify_data.py --sha256` passes:

```bash
python scripts/evaluate_local.py --data-dir data/train \
  --split configs/embryo-splits.json --fold fold0 \
  --config configs/classical.json --output reports/classical-fold0.md
python scripts/evaluate_local.py --data-dir data/train \
  --split configs/embryo-splits.json --fold fold1 \
  --config configs/classical.json --output reports/classical-fold1.md
```

For substantial Vast runs, use the logged launcher described in `AGENTS.md`.
The evaluator saves Markdown, JSON, and one prediction CSV per dataset. It scores
the CSV-reloaded graph, checks graph round-trip equality, preserves the estimated
cell count from GEFF, and uses the organizer's aggregation. The JSON includes raw
and adjusted edge Jaccard, node recall, predicted/estimated counts, division
counts, runtime, environment, source/config/split identities and input inventory
identities. JSON uses null for undefined metrics, not nonstandard NaN literals.

`--metric-backend local-probe --metadata-smoke-only` is exclusively for synthetic
fixtures. Its report is labeled accordingly and cannot establish model quality.

## Inference, recovery and CSV validation

```bash
python scripts/make_submission.py --data-dir data/test \
  --config configs/classical.json --output reports/rehearsal/submission.csv \
  --cache-dir reports/rehearsal/graphs --deadline-seconds 32400
python -m biohub_ct.submission.validator reports/rehearsal/submission.csv
```

Images must be uint16 TZYX with positive scale. Production checks competition v3
chunk paths before reading each frame; missing chunks do not become zeros.
Metadata-only `--debug` is a fixture smoke mode and never supplies real predictions.
The baseline reads one frame at a time. Peak selection uses original intensities
to avoid clipped plateaus. NMS and greedy association use exact spatial bins,
with original distance/tie rules checked against brute-force references.

The runtime keeps one dataset graph at a time. It checkpoints completed graphs as
JSON and validates their checksums on restart. Reuse the same command to resume.
A source, config or input-inventory change rejects the cache; use a new run/cache
directory. Inputs are assumed immutable. Input identity includes metadata hashes
and chunk size/mtime, not full chunk-content hashes. Dataset integrity is the
separate SHA256 download-verification gate. Use one process per output/cache
folder. Resume retains completed datasets, not a partially processed dataset.

The final CSV is replaced atomically after schema, exact coverage, coordinates,
dataset grouping, duplicate IDs/edges, endpoints, consecutive times, one-parent
and two-child constraints pass. A failure preserves the previous final CSV and
completed graphs. A runtime deadline is checked between frames/linking steps;
the generated Kaggle notebook adds a hard process alarm. No time-strided edges
are used to salvage an incomplete run.

## Build and execute the Kaggle package

```bash
python scripts/package_kaggle_notebook.py --output-dir work/kaggle-release \
  --kernel-id clarkkitchen/biohub-submission-rehearsal \
  --config configs/classical.json --deadline-seconds 32400
kaggle kernels push -p work/kaggle-release --timeout 43200
kaggle kernels status clarkkitchen/biohub-submission-rehearsal
kaggle kernels output clarkkitchen/biohub-submission-rehearsal -p reports/kaggle-release
```

On the configured Windows host run Kaggle CLI through WSL as documented in
`AGENTS.md`. The builder writes a self-contained notebook, metadata and manifest.
It embeds only package Python source in a deterministic ZIP, checks its SHA256
before extraction, validates extraction paths, and imports that exact source.
Windows/Linux builds were compared. The attached dependency notebook's manifest
is pinned by SHA256; every wheel is checked before `pip --no-index`. Models,
credentials, datasets and local SSH configuration are not embedded.

Generated metadata uses private CPU execution, internet disabled, the competition
mount and `clarkkitchen/biohub-offline-dependencies`. Actual test IDs are discovered
at runtime. The rehearsal uses a one-hour bound; a release build defaults to nine
hours for inference, leaving a reserve within the competition's 12-hour limit.
Check runtime on representative clips before using that allowance. A notebook
push is not completion, and completion is not a scored competition submission.

See [current discussion evidence](research/alpha-2026-09-05.md),
[architecture decision](decisions/0003-testing-and-offline-release.md) and
[executed results](experiments/2026-09-05-architecture.md).
