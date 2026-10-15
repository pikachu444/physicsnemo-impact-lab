$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskWork = Join-Path $taskRoot 'work\bumper-fem-rerun'
New-Item -ItemType Directory -Force -Path $taskWork | Out-Null
$taskManifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'source-manifest.json') -Raw | ConvertFrom-Json
foreach ($taskItem in $taskManifest.files) {
    $taskTarget = Join-Path $taskWork $taskItem.name
    if (-not (Test-Path -LiteralPath $taskTarget)) { Invoke-WebRequest -Uri $taskItem.url -OutFile $taskTarget }
    if ((Get-Item -LiteralPath $taskTarget).Length -ne $taskItem.bytes) { throw ('Input size mismatch: ' + $taskItem.name) }
    if ((Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskItem.sha256) { throw ('Input SHA256 mismatch: ' + $taskItem.name) }
}
Write-Output 'Run224 original inputs and upstream logs verified at pinned dataset revision.'
