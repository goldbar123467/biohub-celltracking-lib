#!/usr/bin/env bash
set -euo pipefail
project=/workspace/biohub-cell-tracking
server=biohub
if ! tmux -L "$server" has-session -t biohub 2>/dev/null; then
    tmux -L "$server" -f "$project/configs/tmux.conf" new-session -d -s biohub -n work -c "$project" "bash --rcfile $project/scripts/tmux.bashrc -i"
    tmux -L "$server" new-window -t biohub -n download -c "$project" "tail -n 20 -F $project/reports/download.log"
    tmux -L "$server" new-window -t biohub -n gpu -c "$project" 'watch -n 10 nvidia-smi'
    tmux -L "$server" select-window -t biohub:work
fi
if [[ "${1:-}" != --detach ]]; then
    exec tmux -L "$server" attach-session -t biohub
fi
