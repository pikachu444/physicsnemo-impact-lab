$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskRun = Join-Path $PSScriptRoot 'run'
$taskRunKey = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'execution-metrics.json') -Algorithm SHA256).Hash.Substring(0,16).ToLowerInvariant()
$taskRaw = Join-Path $taskRoot ('work\opencourant-experiment\raw-results\' + $taskRunKey)
New-Item -ItemType Directory -Force -Path $taskRaw | Out-Null
foreach ($taskFile in (Get-ChildItem -LiteralPath $taskRun -File)) {
    $taskRawFile = Join-Path $taskRaw $taskFile.Name
    # Keep an existing untouched raw copy when preparing an already-sanitized result.
    if (-not (Test-Path -LiteralPath $taskRawFile)) { Copy-Item -LiteralPath $taskFile.FullName -Destination $taskRawFile }
    if ($taskFile.Extension -in '.out', '.log') {
        $taskText = Get-Content -LiteralPath $taskFile.FullName -Raw
        if ($null -ne $taskText) {
            $taskText = $taskText.Replace($taskRoot, '<WORKSPACE>')
            if ($env:COMPUTERNAME) { $taskText = $taskText.Replace($env:COMPUTERNAME, '<HOST>') }
            Set-Content -LiteralPath $taskFile.FullName -Value $taskText -Encoding utf8
        }
    }
}
$taskExecutableFiles = Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File | Where-Object {$_.Extension -in '.exe', '.dll', '.zip'}
if ($taskExecutableFiles) { throw 'Executable package/library found in publication folder' }
Write-Output 'Preserved raw results in work/ and removed machine identifiers from public text logs. Honor .gitignore when publishing.'
