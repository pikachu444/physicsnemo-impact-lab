$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskWork = Join-Path $taskRoot 'work\opencourant-experiment'
New-Item -ItemType Directory -Force -Path $taskWork | Out-Null
$taskArchive = Join-Path $taskWork 'OpenCourant_win64.zip'
$taskExpected = '6D0EA8B43B68AA12F0A2787B339384A554978CBB65A5D3DBAF1E40598F02E7E6'
if (-not (Test-Path -LiteralPath $taskArchive)) {
    Invoke-WebRequest -Uri 'https://github.com/OpenCourant/OpenCourant/releases/download/latest-20261011/OpenCourant_win64.zip' -OutFile $taskArchive
}
if ((Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash -ne $taskExpected) { throw 'Solver archive SHA256 mismatch' }
if ((Get-Item -LiteralPath $taskArchive).Length -ne 350476183) { throw 'Solver archive size mismatch' }
if (-not (Test-Path -LiteralPath (Join-Path $taskWork 'solver\OpenCourant\exec\engine_win64.exe'))) {
    Expand-Archive -LiteralPath $taskArchive -DestinationPath (Join-Path $taskWork 'solver')
}
$taskSource = Join-Path $taskWork 'box-crash'
New-Item -ItemType Directory -Force -Path $taskSource | Out-Null
$taskCommit = 'cb760c206e346eff12161a2268b134ef9d48a4ef'
$taskFiles = @('BOX1BL_0000.rad','BOX1BL_0001.rad','qadiags.inc','ref.extract','readme.txt','motcle')
foreach ($taskFile in $taskFiles) {
    Invoke-WebRequest -Uri ('https://raw.githubusercontent.com/OpenCourant/OpenCourant/' + $taskCommit + '/qa-tests/miniqa/CRASH/crashsection/box1_42/data/' + $taskFile) -OutFile (Join-Path $taskSource $taskFile)
}
$taskReference = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'validation-summary.json') -Raw | ConvertFrom-Json
foreach ($taskFile in $taskFiles) {
    $taskExpectedInput = $taskReference.input_sha256.$taskFile
    if ((Get-FileHash -LiteralPath (Join-Path $taskSource $taskFile) -Algorithm SHA256).Hash -ne $taskExpectedInput) { throw ('Input SHA256 mismatch: ' + $taskFile) }
}
Write-Output 'Pinned solver package and original FEM input downloaded and SHA256 verified.'
