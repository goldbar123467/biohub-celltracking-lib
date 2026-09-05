param([Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')][string]$RunId)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$sshConfig = Join-Path $projectRoot 'configs/ssh_config'
$snapshotLines = & ssh -F $sshConfig vast-biohub "cd /workspace/biohub-cell-tracking && .venv/bin/python scripts/inspect_campaign.py $RunId"
if ($LASTEXITCODE -ne 0) { throw 'Remote campaign inspection failed' }
$snapshotText = $snapshotLines -join "`n"
$snapshot = $snapshotText | ConvertFrom-Json
$backupRoot = Join-Path $projectRoot "reports/campaign-backups/$RunId"
New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
foreach ($property in $snapshot.checkpoints.PSObject.Properties) {
    $relative = $property.Name
    if ($relative -notmatch '^(fold0|fold1|refit)/checkpoints/step-[0-9]+-[0-9a-f]+\.pt$') { throw 'Invalid backup path' }
    $destination = Join-Path $backupRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    if (!(Test-Path -LiteralPath $destination)) {
        $temporary = $destination + '.partial'
        & scp -F $sshConfig "vast-biohub:/workspace/biohub-cell-tracking/reports/campaigns/$RunId/$relative" $temporary
        if ($LASTEXITCODE -ne 0) { throw "Checkpoint download failed: $relative" }
        if ((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value.sha256) { throw 'Downloaded checkpoint hash mismatch' }
        Move-Item -LiteralPath $temporary -Destination $destination
    }
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value.sha256) { throw 'Backup checkpoint hash mismatch' }
    $sidecar = [System.IO.Path]::ChangeExtension($destination, '.sha256.json')
    @{sha256=$property.Value.sha256; file=$property.Value.file} | ConvertTo-Json | Set-Content -LiteralPath $sidecar -Encoding utf8
}
# Snapshot includes a consistent manifest referencing the immutable verified files above.
$snapshotText | Set-Content -LiteralPath (Join-Path $backupRoot 'snapshot.json') -Encoding utf8
foreach ($property in $snapshot.reports.PSObject.Properties) {
    if ($property.Name -match '\.\.' -or $property.Name -notmatch '^[a-zA-Z0-9/_.-]+$') { throw 'Invalid report path' }
    $destination = Join-Path $backupRoot $property.Name
    New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
    $property.Value | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $destination -Encoding utf8
}
$snapshot | ConvertTo-Json -Depth 100
