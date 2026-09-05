#!/usr/bin/env bash
# Usage: run-job.sh unique-run-id command [arguments...]
set -euo pipefail
project=/workspace/biohub-cell-tracking
if [[ $# -lt 2 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ ]]; then
    echo 'Usage: run-job.sh unique-run-id command [arguments...]' >&2
    exit 2
fi
run_id=$1
shift
mkdir -p "$project/reports/jobs"
# Refuse to overwrite prior jobs or silently launch the same run twice.
mkdir "$project/reports/jobs/$run_id"
bash "$project/scripts/tmux-session.sh" --detach
printf -v launch '%q ' bash "$project/scripts/job-entrypoint.sh" "$run_id" "$@"
tmux -L biohub new-window -d -t biohub -n "$run_id" -c "$project" "$launch"
printf 'Started %s in tmux; logs: %s/reports/jobs/%s\n' "$run_id" "$project" "$run_id"
