#!/usr/bin/env bash
# Launch one reviewed source snapshot with a server-enforced campaign deadline.
set -euo pipefail
project=/workspace/biohub-cell-tracking
cd "$project"
if [[ $# -lt 1 || $# -gt 3 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]; then
    echo 'Usage: start-training-campaign.sh unique-run-id [failed-parent-run-id [repair-seconds]]' >&2
    exit 2
fi
run_id=$1
total_seconds=27000
resume_args=()
if [[ $# -ge 2 ]]; then
    total_seconds=$(.venv/bin/python -m biohub_ct.training.recovery "$2" --repair-seconds "${3:-600}")
    resume_args=(--resume-campaign "$2" --repair-seconds "${3:-600}")
fi
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
    timeout --signal=TERM --kill-after=30s "$((total_seconds + 120))s" \
    python "$snapshot/scripts/run_training_campaign.py" --run-id "$run_id" \
    --total-seconds "$total_seconds" --fold-seconds 9000 --refit-seconds 3600 "${resume_args[@]}"
python -c 'import sys; from pathlib import Path; from biohub_ct.training.checkpoint import atomic_json; atomic_json(Path("reports/campaign-active.json"), {"run_id": sys.argv[1]})' "$run_id"
