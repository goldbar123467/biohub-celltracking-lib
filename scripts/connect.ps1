$ErrorActionPreference = 'Stop'
ssh -t -F (Join-Path $PSScriptRoot '..\configs\ssh_config') vast-biohub 'bash /workspace/biohub-cell-tracking/scripts/tmux-session.sh'
exit $LASTEXITCODE
