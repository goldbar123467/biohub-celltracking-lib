# Competition contract

Verified 2026-09-05 from the signed-in Kaggle Overview, Data and Rules pages.

- Competition: `biohub-cell-tracking-during-development`.
- Task: detect cells, link them through 3D+time microscopy, and reconstruct divisions.
- Maximize `adjusted_edge_jaccard + 0.1 * division_jaccard`. Preserve raw TP/FP/FN and both components when comparing runs. Scores can exceed one.
- Matching uses physical coordinates with a 7 micrometer threshold. Spatial voxel scale `(z,y,x)` is `(1.625,0.40625,0.40625)` micrometers.
- Images: Zarr v3, array `0`, axes `(T,Z,Y,X)`, typically `(100,64,256,256)`, uint16, one timepoint per chunk, blosc/zstd.
- Training graphs: paired GEFF directories; integer centroids and directed source/target edges. Labels are sparse. Unannotated cells must not automatically become training negatives.
- GEFF metadata contains `estimated_number_of_nodes`, an estimate rather than the number of annotated nodes.
- First folder-name segment identifies the embryo. Proposed validation: hold out complete embryos, keeping all fields of view and timepoints from an embryo together. Decide exact folds after the dataset audit.
- Downloadable test samples are examples copied from train. The real hidden test is embryo-disjoint and is approximately the training set's size. Never use example-test performance as evidence of generalization, and never hardcode four visible test IDs.
- Data page: 24,886 files, 87.61 GB. These are website values, not a completed local inventory.
- Submission: Kaggle notebook rerun, internet disabled, CPU/GPU runtime at most 12 hours, output named `submission.csv`. The repository's CSV-upload wording does not replace Kaggle's notebook requirement.
- Exact CSV header: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`. Node/edge unused fields are -1; id is consecutive; dataset identifiers must cover every actual test folder without `.zarr`.
- Five submissions per day; up to two final selections; team maximum five members.
- Entry and team merger: 2026-09-22 23:59 UTC. Final submission: 2026-09-29 23:59 UTC.
- Freely/publicly available external data and pretrained models are allowed subject to the rules. Dataset license is CC0; winning solution license is MIT. Review any model/code license before adoption.
- Account already accepted the competition rules; verified in browser. Authenticated sample download succeeded from WSL.

## Sources

- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data
- https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md

The host-linked metric documentation is more precise than the overview's division summary. Reproduce the pinned metric code before model selection; neither text alone nor a public leaderboard delta proves equivalence with the scoring server.
