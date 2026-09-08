# GPU allocation and execution

## Inventory before scheduling

Create the live `resources` section from `STATE_TEMPLATES.md`. Read the existing provider allocation and Kaggle account rather than copying balances from reports. Record GPU model/count/VRAM, CPU cores, RAM, persistent and ephemeral storage, driver/runtime versions, data location, billing basis, remaining authorized rental funds or prepaid hours, rental expiry, and Kaggle quota/reset time.

The September 7 balance of 29.98/30 Kaggle GPU hours is stale. The reported 12-hour notebook limit and five daily submissions must also be refreshed. A notebook accelerator ID being documented does not establish account entitlement. Run a real small CUDA operation and record device properties before a costly workload.

## Work placement

| Work | Preferred placement | Required output before leaving that stage |
| --- | --- | --- |
| Source audit, manifests, scoring, cached threshold/NMS sweep | Local/controller CPU | Provenance, metrics table, reproducible configuration |
| Full-volume logits, teacher segmentation, controlled student fit | Existing rented GPU allocation | Cached outputs or checkpoint plus manifest and timing |
| Assignment/ILP and gap/division sweeps on cached nodes | CPU; rented host CPU only when economical | Graphs, component metrics, solver diagnostics |
| Final hardware compatibility and offline inference | Kaggle GPU | Completed committed version, CSV, logs, run manifest |
| Small training pilot already staged on Kaggle | Kaggle GPU only after protecting the release reserve | Recoverable checkpoint and real development evaluation |

Avoid making the rented GPU wait for serial downloads, repeated decompression, CPU graph solves, or manual review. Stage data and wheels first. Run CPU diagnostics alongside GPU work if measured host memory/I/O remain adequate. GPU utilization alone is not the objective; completed valid experiments per billed hour is.

## Reserve calculation

Use two independent ledgers: provider dollars/instance-hours and Kaggle-account quota hours. Also report GPU-hours for scientific comparisons. An instance containing two GPUs may bill per instance; Kaggle quota charging may differ. Measure each platform's debit instead of assuming both equal physical GPU count times time.

Let `K` be live remaining Kaggle quota, `r` the measured quota debit rate per wall hour, and `T` a conservative end-to-end release duration including setup, inference, graph solving, CSV writing, and validation. Protect two complete release attempts:

`K_reserve = 2 * r * T`.

Before `T` is credible, reserve two runs at the verified allowed runtime. This deliberately leaves little Kaggle quota for research; use the rented machine or CPU until profiling reduces uncertainty. If available quota cannot cover the reserve, protect one runnable release, stop Kaggle training, and prioritize reducing runtime. Do not assume hidden scoring and rehearsal have identical quota treatment; reconcile actual debits.

For each paid job reserve its full worst-case cost, including known storage or transfer charges and the startup/shutdown allowance. Admission requires:

`spent + outstanding_reservations + proposed_reservation <= authorized_total`.

Do not subtract a reservation from remaining funds twice when reconciling spend. Record cumulative spend, incremental debits, and unreconciled estimates separately. Settling a completed reservation releases only its unused portion. An uncertain provider bill remains reserved.

If the allocation is prepaid, enforce expiry and remaining instance-hours even when incremental GPU cost is zero. A null cap means unresolved, not unlimited. The current request permits using an established allocation; it does not set a numerical new-purchase budget.

## Initial allocation policy

These are proposed starting fractions of the *remaining authorized rented compute*, not mandatory spending targets or empirical optima. The helper may shift unused time after a gate passes.

| Bucket | Initial share | Purpose |
| --- | ---: | --- |
| Public reference reproduction and inference profiling | 30% | Establish the strongest runnable anchor |
| Frozen-detector cache and numerical diagnosis | 15% | Determine whether calibration rescues useful recall |
| Conditional teacher/distillation pilot | 20% | Test supervision quality at matched budget |
| Conditional winning-model fit/refit | 15% | Scale only demonstrated improvements |
| Evaluation, recovery, export, and transfer reserve | 20% | Finish a release even after a failed experiment |

Unused distillation or fit time returns to the common pool. If the public reference dominates and can be deployed reliably, invest the freed time in association, division ablations, and release verification. Do not run all branches to satisfy these percentages.

## Admission and stop rules

Record both a hard deadline and a maximum work unit for every job. Start with a profile, then a pilot, then an expansion. Suggested policy defaults are a 10–15 minute representative profile and a pilot capped at the smaller of 60 minutes or 10% of the remaining rental allocation, if that can produce a meaningful comparison. Otherwise reduce the task and document that it is only a smoke run. Include data staging and evaluation in the reservation.

The launcher must enforce limits independently of the hourly helper. A job deadline may fall before the next review. Emit a heartbeat at least every minute during normal work; checkpoint completed work at a measured interval targeting at most ten minutes of lost training, and at every natural inference chunk/clip boundary. Tune the interval if checkpoint I/O is material. Bound solver time and retain a feasible incumbent.

Stop or checkpoint a job on its time/cost cap, non-finite loss/output, repeated OOM, input identity mismatch, or an explicit experiment stop criterion. A first OOM can justify one logged retry with a smaller microbatch/tile under the same reservation. Preserve effective batch size with accumulation where needed. Do not silently change numerical behavior or restart a failed job indefinitely.

A stale heartbeat is a signal to inspect, not an instruction to kill. Confirm process identity, last step duration, I/O state, and provider status. Kill only the verified campaign process after a graceful checkpoint attempt. Do not stop unrelated jobs or delete the only copy of weights. Stopping Python is not proof that rental billing stopped; record provider state separately and use only the authorized lifecycle operation after durable persistence.

## Caches and data movement

Key each cache by input data identity, split membership, model hash, code revision, preprocessing, tiling, precision, augmentation, and output schema. Changes to threshold/NMS can reuse compatible logits; changes to weights, resampling, tile blending, or temporal context cannot. Preserve node IDs when caching association experiments.

Prefer storing raw logits in float32 on the small diagnostic panel. Quantizing already computed logits can introduce plateaus that a later float32 sigmoid cannot undo. If production caches are compressed or lower precision, validate that representation against the reference before sweeping them. Stream Zarr frames/tiles; avoid materializing all 4D image volumes in RAM.

Student checkpoints include model, optimizer, scheduler, scaler if used, random states, sampler progress, split hash, step, and measured spend. Export inference-only weights separately. Write a temporary file, flush, atomically rename, hash, transfer to persistent storage, and verify at destination before reclaiming ephemeral storage. Log which seed and step resumed; resuming must not replay the same shard unintentionally.

## Timing and hardware choices

Profile representative shape, density, and time-length strata, including difficult capped frames. Report setup, data read, model, candidate extraction, pair scoring, solver, serialization, and validation separately, plus peak RAM/VRAM. Pair scoring may grow with `n_t * n_(t+1)` before gating; detector inflation can therefore increase cost faster than node count.

Estimate runtime from actual frame/tile counts and candidate-edge counts. Use measured tail behavior and a planning reserve, initially 25%, while labeling it an estimate. Four visible clips and the historical 548.80 seconds do not establish a hidden-test total. The earlier 199-clip case is a planning scenario, not a guaranteed test size.

Use AMP only after checking peak extraction and graph-level changes against a float32 reference. Do sigmoid, peak comparisons, and probability fusion in float32 where practical. Keep spatial units and resize offsets explicit. Do not enable compilation, alternate kernels, multi-GPU execution, or a different GPU rental merely on theoretical speedups: include warmup, setup, transfer, numerical drift, and remaining campaign duration in the break-even calculation.
