$ErrorActionPreference = 'Stop'
ssh -F (Join-Path $PSScriptRoot '..\configs\ssh_config') vast-biohub 'supervisorctl status biohub-download; cat /workspace/biohub-cell-tracking/reports/download-status.json; df -h /workspace'
exit $LASTEXITCODE
