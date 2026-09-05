$ErrorActionPreference = 'Stop'
ssh -N -F (Join-Path $PSScriptRoot '..\configs\ssh_config') vast-biohub-tunnel
exit $LASTEXITCODE
