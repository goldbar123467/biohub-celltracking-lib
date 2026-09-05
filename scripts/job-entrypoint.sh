#!/usr/bin/env bash
set -euo pipefail
run_id=$1
shift
source /workspace/biohub-cell-tracking/scripts/activate.sh
run_dir="$BIOHUB_PROJECT/reports/jobs/$run_id"
date -u +%FT%TZ > "$run_dir/started_at.txt"
printf '%s\n' "$$" > "$run_dir/pid.txt"
set +e
"$@" 2>&1 | tee "$run_dir/output.log"
results=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${results[0]}" > "$run_dir/exit-code.tmp"
mv "$run_dir/exit-code.tmp" "$run_dir/exit-code.txt"
printf '%s\n' "${results[1]}" > "$run_dir/log-exit-code.txt"
date -u +%FT%TZ > "$run_dir/finished_at.txt"
printf '\nJob %s finished: command=%s log=%s\n' "$run_id" "${results[0]}" "${results[1]}"
if [[ "${results[0]}" -ne 0 ]]; then exit "${results[0]}"; fi
exit "${results[1]}"
