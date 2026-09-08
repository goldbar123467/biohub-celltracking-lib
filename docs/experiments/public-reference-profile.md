# E0 public-reference compatibility profile

Status: **executed on Vast and independently verified: campaign `COMPLETE`, profile `PASS` for the bounded R5 detector window**.

This profile answers a narrow operational question: can the exact pinned primary and secondary public detector checkpoints load and run their effective float32, eight-call detector TTA and primary edge-feature TTA path on the existing single RTX 4070 environment, and what do bounded frame I/O, normalization, forward passes, transforms, extraction, and peak VRAM cost on one detector window?

It is not a submission rehearsal or a quality evaluation. It does not generate candidates, score edges, run the ILP, invoke DeepCenter, create a CSV, compute a metric, or read a full video. A passing receipt is compatibility and resource evidence only. It cannot reproduce or validate the upstream public score.

## Actual campaign evidence: `e0-profile-20260908-01`

Independent review of the downloaded R5 worker specification, staged input manifest, profile receipt, result manifest, supervisor completion receipt, and worker log found no acceptance-gate mismatch. The reviewer independently recomputed run-spec digest `7676dff3dccf07559912187445383ae5244962e3f2f87f14b48408256d05d3da`. Run ID `e0-profile-20260908-01`, intent `launch-e0-profile-20260908-01`, and fencing token `12` agree across the records. The process exited zero, the result is `COMPLETE` with one completed unit, and the profile is `PASS`.

The executed profile is immutably bound to the R5 entrypoint SHA-256 `1445c1e239a81bf4970b8f6fcddce0ea507c6f92d3e8ae0d75260f6be6b52c11`. The downloaded profile SHA-256 is `7096d9f08453c23268beeef8396f15fc436e4d8693dd7d8fc6435b6b5a1da4f7`; the result manifest records the same artifact hash. The downloaded result manifest SHA-256 is `e11cea67ffcc1d2e87bef4d457133238cc607ca1318d8c0a1ca3fcc69710c4b9`, matching the supervisor receipt, and the worker-log SHA-256 is `e329ad51c77529f687ce6c6e1897bc76efb024f6b1f12db5bd82fba35e6c998e`. The current local next-revision entrypoint is `669070383bccb58091ff1e93ea33bd9cd5482b92bffd626c12128550576ee266`; its changes are not part of this executed result.

The R5 profile directly reports matching pins for the four operational support files it checks: predictor `c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9`, trainer `c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea`, temporal U-Net `d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac`, and node transformer `b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e`. The immutable worker specification separately pins all 13 staged support Python files, including both package initializers. The two 8,363,159-byte checkpoint hashes are `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771` for primary and `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f` for secondary; both configs hash to `e9b4e396c58081bca08adf8275bd0bd1c2d3fd6eb091a1912a5116cb6de7b50a`.

Data identity matches the independently staged manifest: raw shape `100 x 64 x 256 x 256`, frames `0` and `1`, partial frame hashes `f4df31c300fe188842ced6e60a2b667999c602c7830ea5c8427f2c32b44e7939` and `c8b6d5158e34e0f1d8f81870362e6b785eb7527120fb08c8e4e44cfe15e60b94`, and Zarr metadata hashes `c3768f1dfc78a0005d113b174c610c0c6aca375b085ad3b94da28c0a1fbbfda0` and `73e3ff105fe0c50983cff97a50bfa95f8c0da16e1cb5adb56968d1cb7e1c41b0`.

The receipt records one visible NVIDIA GeForce RTX 4070 SUPER with 12,462,456,832 bytes of device memory, Torch `2.9.1+cu128`, CUDA `12.8`, cuDNN `91002`, float32, no autocast, and no gradient tracking. Profile elapsed time was 3.947828511 seconds; supervisor elapsed time was 4.526773297 seconds. Overall peak allocated VRAM was 748,850,688 bytes (6.01% of device capacity) and peak reserved VRAM was 1,077,936,128 bytes (8.65%). Both models returned `[1, 2, 32, 64, 64, 64]` retained features and finite detection/fusion tensors. Each made eight calls over seven unique effective transforms; `anti_transpose` was effectively the duplicated `flip_x`. The primary mean absolute feature delta from identity was 0.323242128, while the secondary identity-feature delta was exactly zero, as required by the pinned public path.

The supervisor completion receipt contains its last sampled initial worker-progress record with zero completed units because this 4.5-second worker finished between polls. The independently hashed result manifest is the completion authority here: it records one completed unit and the required profile artifact, and the supervisor validated it as complete with quality review required.

These measurements apply to this one current-data, two-frame detector window. They do not support a full-video runtime extrapolation. Candidate generation, association, transformer scoring, ILP, DeepCenter, CSV generation, and metric execution remain unmeasured by this profile.

## Fixed execution contract

The profiler in [`scripts/profile_public_reference.py`](../../scripts/profile_public_reference.py) is hard-bound to:

- training dataset `44b6_0113de3b`, frames `0` and `1`, with frame `0` as the requested anchor;
- the observed fixed raw `TZYX` shape `100 x 64 x 256 x 256`, with one `64 x 64 x 64` frame volume after the public `(1, 4, 4)` spatial stride; the default and hard maximum Y/X tile size are both `64`;
- exactly two checkpoints in primary then secondary order;
- the pinned support-source and checkpoint/config SHA-256 values recorded by E0;
- one visible CUDA device whose reported name contains `4070`;
- float32, no autocast, inference mode, eight upstream detection encode calls, primary edge-feature TTA accumulation, and the secondary model's identity-view edge features;
- a default internal wall alarm/stage deadline of 120 seconds and a hard configurable maximum of 540 seconds; the launch below gives the default run a separate 150-second process guard.

The data root must directly contain `44b6_0113de3b.zarr`. The support root may be the extracted support archive root, its `repo` directory, or its `repo/src` directory. ZIP files are rejected because extraction would make this diagnostic mutate its environment. The two checkpoint paths and their adjacent `config.json` files are explicit.

The code imports NumPy, Torch, and Zarr from the existing environment. It never invokes pip, uv, conda, apt, or a network client. Missing libraries produce a JSON receipt with `BLOCKED_UNAVAILABLE_DEPENDENCY`; this exposes environment uncertainty without changing it.

## Vast command

First follow `/etc/vast-agents-guide.md` and `/workspace/AGENTS.md`, activate the existing project environment, and substitute the actual already-extracted roots. Put the timeout inside `run-job.sh`'s detached command so it supervises the worker rather than only the short-lived launcher:

```bash
source /workspace/biohub-cell-tracking/scripts/activate.sh
bash /workspace/biohub-cell-tracking/scripts/run-job.sh \
  e0-compat-profile-20260907 \
  timeout --signal=TERM --kill-after=15s 150s \
  python /workspace/biohub-cell-tracking/scripts/profile_public_reference.py \
    --data-root /workspace/biohub-cell-tracking/data/train \
    --support-root /workspace/e0-support \
    --checkpoints \
      /workspace/e0-support/weights/unet_transformer/split_0/edge_predictor_best.pth \
      /workspace/e0-seed314159/weights/unet_transformer/split_0/edge_predictor_best.pth \
    --tile-yx 64 \
    --max-wall-seconds 120 \
    --output /workspace/biohub-cell-tracking/reports/e0-public-reference-profile.json
```

Do not reuse the example run ID. Verify the actual support/checkpoint paths before launching. This command does not authorize extracting archives, installing dependencies, restarting the instance, or running the full public notebook.

For an admitted production campaign run, use `scripts/campaign_worker.py` with an immutable worker ticket instead of `run-job.sh`. The ticket supervisor injects `BIOHUB_RUN_ID`, `BIOHUB_RUN_SPEC_SHA256`, `BIOHUB_INTENT_ID`, `BIOHUB_FENCING_TOKEN`, `BIOHUB_ATTEMPT_DIR`, and `BIOHUB_PROGRESS_PATH`. The profile output must be inside the exclusive attempt directory. The run specification must set:

- `execution.entrypoint_file` to `scripts/profile_public_reference.py` and bind that file in `source_file_sha256`;
- `execution.max_steps_or_clips` and `expected_completed_units` to `1`;
- `execution.worker_progress_path` to the injected attempt's `progress.json`;
- `expected_artifacts` to the repository-relative profile JSON path inside that attempt;
- `result_manifest_path` to the repository-relative `result.json` inside that attempt.

The profiler rejects a partial or malformed campaign environment. It writes initial progress with zero completed units, writes the profile receipt, writes a `COMPLETE` result manifest that hashes and references that exact receipt, then writes final progress with one completed unit. A failed profile writes the structured receipt and error progress but no complete result manifest. The campaign supervisor independently applies GNU `timeout`, validates the result identity and artifact hash, and marks the result for quality review.

## Exact upstream view sequence

The pinned notebook labels its augmentation as eight-view planar “D4-style” TTA, but its eighth input expression is `rot90(1).transpose(-1, -2)`. For a square Y/X plane that expression is exactly `flip_x`, which is already the second view. The executed sequence therefore has eight model calls and seven unique effective transforms. The profiler preserves the exact expression and inverse expression for compatibility, records both the source label and effective transform for every call, and does not describe the sequence as the eight unique D4 symmetries. This may be an upstream implementation mistake; correcting it would define a different experiment.

## Receipt and acceptance criteria

The current JSON receipt format binds the profiler entrypoint, six imported or adapted support-source files, the frozen upstream notebook pin, both checkpoints, both configs, available Zarr metadata files, raw sampled frame bytes, normalized tensor bytes, averaged feature/detection tensors, and fused detection tensors with SHA-256 values, shapes, dtypes, and finite-value checks. It also records library/CUDA versions, device identity, exact effective constants, the eight-call/seven-unique view contract, per-view transform/forward/inverse timings, data/normalization/transfer/load/extraction timings, parameter sizes, and allocated/reserved peak VRAM. JSON writing rejects NaN and infinity. The notebook pin is recorded provenance rather than a live notebook-byte verification because this bounded entrypoint does not read or execute the notebook. The executed R5 receipt predates the two package-initializer fields in the current format; its immutable external worker specification pins those files and every other staged support Python file.

Accept the compatibility profile only when all of these hold:

1. process exit code is zero and receipt status is `PASS`;
2. the selected device is exactly one visible RTX 4070-class device;
3. both source/config/checkpoint pins match, all tensor summaries are finite, and primary/secondary feature shapes agree;
4. each model records the exact eight-call upstream sequence and seven unique effective transforms, the primary model records a finite nonzero edge-feature TTA delta, and the secondary model records unchanged identity-view edge features as the public source does;
5. elapsed time is at most the requested 120 seconds and overall allocated peak VRAM is below the recorded device capacity;
6. the job log has a final exit code and the receipt is copied to durable Windows storage with its own SHA-256.

A blocked dependency receipt is evidence that compatibility remains unknown. A failed pin or shape contract means the supplied artifact/path is wrong or the source has drifted. A CUDA OOM or wall-budget failure rejects this detector-window/device combination. A pass measures one full-size, two-frame temporal detector input after the public XY stride. It does not establish complete-video feasibility because candidate density, transformer/ILP costs, graph post-processing, and complete-video I/O are intentionally absent.

## Local verification boundary

`tests/test_public_reference_profile.py` checks the hard bounds, exact eight-call/seven-unique TTA contract, fixed strided frame slice, fail-closed config validation, strict finite JSON, campaign receipt/result/progress ordering, dependency-unavailable reporting, transform round trips, and the checkpoint wrapper's CPU `B,W,Z,Y,X -> B,W,C,Z,Y,X` encode contract. Torch-dependent CPU tests skip when Torch is absent locally and must execute, not skip, in the Vast environment before accepting its receipt.
