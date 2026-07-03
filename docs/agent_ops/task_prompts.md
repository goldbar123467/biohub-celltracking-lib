# Task Prompts

## 1. Metric Auditor

Compare `src/biohub_ct/metrics/` with the organizer repo metric implementation. Add synthetic tests for any mismatch. Do not change scoring behavior without citing the organizer code path and running `python -m pytest -q`.

## 2. Data/EDA Agent

Inspect local train/test Zarr and GEFF metadata without loading full videos. Report shapes, chunks, dtype, scale, estimated node counts, embryo prefixes, and missing pairs. Save notes under `docs/experiments/eda-YYYYMMDD.md`.

## 3. Classical Baseline Agent

Improve `src/biohub_ct/detection/` and `src/biohub_ct/linking/` with one parameterized change. Run smoke tests and record node counts, runtime, and metric deltas.

## 4. Learned Detector Agent

Scaffold sparse-label 3D heatmap training. Do not add dense background loss from sparse labels. Save configs, seeds, and weight paths. Keep inference offline.

## 5. Linker/ILP Agent

Implement LAP or optional ILP behind the existing linking interface. Handle missing dependencies gracefully. Add tests for one-parent-per-child, division limits, and gap behavior.

## 6. Division Agent

Design conservative division proposals and test them against `docs/competition/metric_deep_dive.md`. Prioritize avoiding annotated-region FP.

## 7. Kaggle Packager Agent

Verify `notebooks/submit.ipynb` and scripts run without internet. Confirm `submission.csv` appears at the notebook working directory and passes validator.

## 8. Adversarial Reviewer Agent

Review for hidden full-video loads, schema drift, runtime-internet assumptions, optional dependency crashes, and metric claims not backed by commands.

