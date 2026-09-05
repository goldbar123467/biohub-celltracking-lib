#!/usr/bin/env bash
# Launch one reviewed source snapshot with a server-enforced campaign deadline.
set -euo pipefail
project=/workspace/biohub-cell-tracking
cd "$project"
if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]; then
    echo 'Usage: start-training-campaign.sh unique-run-id' >&2
    exit 2
fi
run_id=$1
if [[ -n $(git status --porcelain) ]]; then
    echo 'Commit and synchronize reviewed source before launch.' >&2
    exit 2
fi
revision=$(git rev-parse HEAD)
snapshot="$project/work/campaign-sources/$run_id"
mkdir -p "$project/work/campaign-sources"
mkdir "$snapshot"
git archive HEAD | tar -x -C "$snapshot"
source "$project/scripts/activate.sh"
# The 7.5-hour shared campaign allocation includes fitting, selection and evaluation.
# External timeout is independent of Python, with 120 seconds for clean shutdown.
bash "$project/scripts/run-job.sh" "$run_id" env \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 BIOHUB_SOURCE_COMMIT="$revision" \
    PYTHONPATH="$snapshot/src" \
    timeout --signal=TERM --kill-after=30s 27120s \
    python "$snapshot/scripts/run_training_campaign.py" --run-id "$run_id" \
    --total-seconds 27000 --fold-seconds 9000 --refit-seconds 3600
