# Source Index

Retrieved on 2026-07-03.

## Competition And Organizer Sources

| Source | URL | What It Establishes |
| --- | --- | --- |
| Kaggle overview | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development | Task is 3D+time cell detection, tracking, divisions, lineage reconstruction. |
| Kaggle data page | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data | Zarr v3 image volumes, `(T,Z,Y,X)` shape, typical `(100,64,256,256)`, one-timepoint chunks, physical scale, train/test layout, sparse GEFF ground truth, CC0 data. |
| Kaggle rules | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules | Code competition submission constraints; notebook runtime and internet-off constraints should be rechecked in Kaggle UI before final submission. |
| Kaggle leaderboard | https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard | Public leaderboard uses about 29% of test data in the crawled snapshot, so public LB tuning is risky. |
| Organizer repo | https://github.com/royerlab/kaggle-cell-tracking-competition | Official baseline code, BSD-3 license, OME-Zarr/GEFF layout, sparse supervision, 3D U-Net plus transformer linker baseline. |
| Metric prose | https://raw.githubusercontent.com/royerlab/kaggle-cell-tracking-competition/main/metrics.md | 7 um node matching, sparse GT edge FP rules, adjusted edge Jaccard, division component coverage, final score form. |
| Metric code | https://raw.githubusercontent.com/royerlab/kaggle-cell-tracking-competition/main/src/tracking_cellmot/metrics.py | Duplicate edge dedupe, valid predicted edge definition, node-count penalty implementation, micro aggregation. |
| Division code | https://raw.githubusercontent.com/royerlab/kaggle-cell-tracking-competition/main/src/tracking_cellmot/division_metrics.py | GT division extraction, component coverage, predicted fork requirement, bipartite pairing, annotated-region division FP behavior. |
| Organizer IO | https://raw.githubusercontent.com/royerlab/kaggle-cell-tracking-competition/main/src/tracking_cellmot/io.py | Default scale, lazy shape metadata path, tracksdata GEFF loading, dataset listing. |
| Organizer training/prediction scripts | https://github.com/royerlab/kaggle-cell-tracking-competition/tree/main/scripts | Baseline detector/linker training and inference flow to inspect before learned-model work. |

## Public Kaggle Work Snapshot

Search results on 2026-07-03 showed active public notebooks including classical DoG/physical-linking baselines, data-model EDA baselines, learned graph trackers with safe divisions, and U-Net/ILP-style baselines. These are listed in `kaggle_public_work.md`. Do not copy notebook code without opening and checking license/rules/provenance.

## Cell Tracking And Bioimage Sources

| Source | URL | Useful For |
| --- | --- | --- |
| Ultrack repo | https://github.com/royerlab/ultrack | Segmentation-uncertainty tracking, terabyte-scale zebrafish context, solver tradeoffs. |
| Ultrack docs | https://royerlab.github.io/ultrack/ | Out-of-memory design and 2D/3D tracking interfaces. |
| Ultrack Nature Methods | https://www.nature.com/articles/s41592-025-02778-0 | Method context and biological-scale tracking claims. |
| Cell Tracking Challenge | https://celltrackingchallenge.net/ | Benchmark framing and CTC conventions. |
| CTC 10-year paper | https://www.nature.com/articles/s41592-023-01879-y | Objective benchmarking lessons and metric caution. |
| StarDist | https://stardist.net/ and https://github.com/stardist/stardist | Star-convex object detection/segmentation, 3D microscopy options. |
| Cellpose | https://www.cellpose.org/ | Generalist cellular segmentation and Cellpose-SAM direction. |
| Trackastra | https://github.com/weigertlab/trackastra and https://arxiv.org/abs/2405.15700 | Transformer association for tracking-by-detection and division-aware linking. |
| btrack | https://btrack.readthedocs.io/ and https://github.com/quantumjot/btrack | Bayesian tracklets, appearance/motion likelihoods, integer-programming hypotheses. |
| Motile | https://github.com/funkelab/motile | ILP graph optimization interface. |
| LapTrack | https://academic.oup.com/bioinformatics/article/39/1/btac799/6887138 | LAP-based particle/cell tracking ideas. |
| Live Image Tracking Tools | https://liveimagetrackingtools.org/pages/awesome-tools/ | Tool landscape. |
| GEFF Figshare | https://janelia.figshare.com/articles/code/GEFF_Graph_Exchange_File_Format/31943145 | GEFF exchange-format purpose, packages, v1.1 snapshot, MIT license. |
| GEFF repo | https://github.com/live-image-tracking-tools/geff | Python read/write libraries and supported graph backends. |
| Zarr Python | https://zarr.readthedocs.io/ | Lazy/chunk-aware array access. |
| Zarr project | https://zarr.dev/ | Format ecosystem. |

## Agent Operation Sources

| Source | URL | Useful For |
| --- | --- | --- |
| OpenAI Codex docs | https://developers.openai.com/codex | Codex docs, customization, AGENTS, worktrees, subagents. |
| OpenAI Codex cloud/web | https://developers.openai.com/codex/cloud | Hosted/offloaded agent behavior. |
| Codex CLI repo | https://github.com/openai/codex | Local coding agent and install docs. |
| Codex app announcement | https://openai.com/index/introducing-the-codex-app/ | Parallel agents/worktrees positioning. |
| Codex agent loop | https://openai.com/index/unrolling-the-codex-agent-loop/ | Agent-loop design context. |
| AGENTS convention | https://agents.md/ | Repository-local agent instruction convention. |

## Unresolved Questions

- Confirm current Kaggle UI runtime and internet rules immediately before final submission.
- Verify hidden-test folder mount path in an actual Kaggle notebook.
- Inspect public notebook code manually before adopting ideas.
- Install `tracksdata` and compare local synthetic probes against the official metric implementation before trusting local CV scores.

