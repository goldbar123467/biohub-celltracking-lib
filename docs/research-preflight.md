# Setup research, 2026-09-05

Decision: provision data access and preserve the host baseline as the first reference. No final model architecture or training budget is selected.

## Inspected sources

1. OpenAI GPT-6 Astra guide, fetched as markdown from the user-supplied official page. Applied autonomous follow-through, clear task boundaries, evidence-based reporting and proportionate verification. No model settings were changed.
   https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra
2. Kaggle CLI official documentation and the installed WSL CLI help (2.2.4). Existing access token lives in WSL, not Windows' usual `.kaggle` folder. Listing competition files and downloading the 890-byte sample submission succeeded.
   https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md
3. KaggleHub official documentation: supports `~/.kaggle/access_token`, `competition_download`, output directory and cache configuration. Use the same credential on the authorized server without printing it.
   https://github.com/Kaggle/kagglehub
4. Royerlab host-linked baseline, BSD-3-Clause, cloned locally at `075fc5f5a52d11077f9dc2b074644618f26939e2`. README, metric documentation, dependency specification and path configuration inspected. Temporal 3D U-Net plus node transformer is a candidate reference, not yet reproduced. Requires Python >=3.11,<3.14, Torch >=2.9.1, Zarr v3 and tracksdata. The tracksdata dependency points to main and must be pinned before reproducibility claims.
   https://github.com/royerlab/kaggle-cell-tracking-competition
5. Competition discussions inspected through the authenticated CLI. Topic 739686 reports possible local/server metric differences; topic 739352 reports CV/public-LB disagreement. These are participant reports, not verified defects. Default CLI comment excerpts are truncated, so they are discovery evidence only. Read full relevant replies before making a metric decision.
   https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739686
   https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739352

## Next method decision

First inventory embryos and shapes, decode a training image chunk and graph, exercise the official metric fixtures, and freeze an embryo-disjoint split. Before choosing a new model, inspect primary papers and author repositories for 3D detection and division-aware tracking and compare them against the host baseline. A full method literature review has not been completed in this environment-setup phase.

Acceptance for the first future experiment: bounded forward/backward/inference run on training-only data, finite loss and gradients, saved/reloaded checkpoint, valid graph-to-CSV round trip, measured memory/runtime, and score from a held-out embryo. Training results and competition submissions do not yet exist.
