# E3 teacher provenance checkpoint

Checked 2026-09-08 UTC. E1 left detector density unresolved, so the experiment
plan permits this provenance audit. The failed E1 advancement threshold does
not prohibit researching E3. No teacher inference or student fitting occurred.

## Verified identity and access

The official [FOCUS-3D source repository](https://github.com/yu-lab-vt/FOCUS-3D)
resolved to commit `5c4b53f743a0fbbae056e2c1a139895ae819f069`, dated September 2.
Its pinned LICENSE is BSD 3-Clause. This establishes source-code terms; it does
not independently establish terms for separately distributed weights.

The repository directs pretrained-model downloads to
[Qinghua-thu/FOCUS-3D](https://huggingface.co/Qinghua-thu/FOCUS-3D).
The public model API returned revision
`115258efcc9ee44e69db3902bce2511d0ae24e2f`, `private=false`, `gated=auto`,
no populated model-card metadata, and these model filenames:
`model_final.pth`, `model_final_membrane.pth`, `model_final_nuclei.pth`.
An anonymous read of the README at that exact revision returned HTTP 401.
No gate was accepted, model downloaded, or competition image uploaded.

The source README describes nucleus/cell segmentation in napari, Z-to-XY
spacing, XY cell radius, intensity-based removal, instance-size filtering,
normalization percentiles, strides, batches and stitching thresholds. These
are preprocessing decisions that must be frozen before a teacher pilot. The
README's interactive workflow does not establish a tested headless inference
entrypoint for this campaign. The full preprint was not accessible through
the browsing tool in this audit; no paper performance claim is adopted.

The raw metadata snapshot is
`reports/campaigns/campaign-20260908-01/e3-provenance/snapshot.json`, SHA-256
`ceb16058bfc2eef9803e847f8390d587e18ec598423e560effd0269aed72695e`.
It also binds the downloaded pinned code license by SHA-256.

## Decision and remaining admission evidence

E3 is **PROVENANCE_UNRESOLVED**, not rejected for poor teacher quality.
Model-weight usage/redistribution terms and pretraining membership remain
unknown. The gate blocks confirming them from this unauthenticated source.
The prior Hugging Face Space BSD metadata is insufficient to close this gap.
Competition eligibility requires the actual model's terms and the current
competition rules; source availability alone is insufficient.

After eligibility is established, prepare a frozen stratified panel exclusively
from each direction's gradient-training IDs. Reserve a distinct training-internal
QC subset for coordinate corrections. Bind each input's split, shape, physical
voxel spacing and hash, plus teacher revision/weight hash, dependency versions,
normalization, resampling, radius, thresholds and filtering. Record XY/XZ/YZ
views, merges, fragments, missed faint centers, border truncation, centroid
offsets, density and wall/GPU cost. Preserve annotations and mask uncertain
regions. Unknown pretraining overlap must remain explicitly overlap-unknown.

Only a successful QC pilot can justify matched sparse-versus-distilled student
runs. Match initial weights/seeds, optimizer updates, batch and real voxel
exposure, and report teacher-labeling costs separately. Use the preregistered E1
count/recall rule together with real development adjusted-edge improvement.
The finite experiment allocation and trained-model evaluation budget remain
unresolved. This checkpoint does not authorize accepting gated terms, spending
additional money, bulk labeling, or fitting a student.
