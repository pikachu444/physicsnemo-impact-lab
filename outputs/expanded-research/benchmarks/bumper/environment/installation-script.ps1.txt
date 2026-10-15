$ErrorActionPreference = 'Stop'
$installRoot = $PSScriptRoot
$python = Join-Path $installRoot 'venv/Scripts/python.exe'
$wheelDir = Join-Path $installRoot 'wheels'
$env:TEMP = Join-Path $installRoot 'tmp'
$env:TMP = $env:TEMP
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PIP_NO_CACHE_DIR = '1'
$env:PYTHONUTF8 = '1'
$commands = [Collections.Generic.List[object]]::new()
$commands.Add([ordered]@{
    step = 'create_venv_completed_before_script'
    executable = 'C:\Program Files\Python313\python.exe'
    arguments = @('-m', 'venv', '--system-site-packages', (Join-Path $installRoot 'venv'))
    exit_code = 0
    log = 'install-venv.log'
})

function Invoke-RecordedPip {
    param([string]$Step, [string[]]$PipArguments)
    $logPath = Join-Path $installRoot ('install-' + $Step + '.log')
    $record = [ordered]@{
        step = $Step
        executable = $python
        arguments = @('-m', 'pip') + $PipArguments
        started_at_utc = [DateTime]::UtcNow.ToString('o')
        log = [IO.Path]::GetFileName($logPath)
    }
    & $python -m pip @PipArguments *> $logPath
    $record.exit_code = $LASTEXITCODE
    $record.finished_at_utc = [DateTime]::UtcNow.ToString('o')
    $commands.Add($record)
    $commands | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $installRoot 'install-commands.json') -Encoding utf8
    Write-Output ($Step + ': exit ' + $record.exit_code)
    Get-Content -LiteralPath $logPath -Tail 12
    if ($record.exit_code -ne 0) { throw ($Step + ' failed; see ' + $logPath) }
}

Invoke-RecordedPip -Step 'download-pytorch' -PipArguments @(
    '--disable-pip-version-check', '--no-cache-dir', 'download', '--no-deps', '--only-binary=:all:',
    '--dest', $wheelDir, '--index-url', 'https://download.pytorch.org/whl/cu126',
    'torch==2.10.0+cu126', 'torchvision==0.25.0+cu126'
)
Invoke-RecordedPip -Step 'download-scatter' -PipArguments @(
    '--disable-pip-version-check', '--no-cache-dir', 'download', '--no-deps', '--only-binary=:all:',
    '--dest', $wheelDir, '--no-index', '--find-links', 'https://data.pyg.org/whl/torch-2.10.0+cu126.html',
    'torch_scatter==2.1.2+pt210cu126'
)
Invoke-RecordedPip -Step 'download-pyg' -PipArguments @(
    '--disable-pip-version-check', '--no-cache-dir', 'download', '--no-deps', '--only-binary=:all:',
    '--dest', $wheelDir, '--index-url', 'https://pypi.org/simple', 'torch_geometric==2.8.1', 'xxhash==4.0.1'
)

$expectedHashes = @{
    'torch-2.10.0+cu126-cp313-cp313-win_amd64.whl' = '91e21e7ad572bf0136e5b7f192714f120c8abde8e128f1a0759f158951643822'
    'torchvision-0.25.0+cu126-cp313-cp313-win_amd64.whl' = '6f7994a27b7fe26f6b4828df20421af004986d01091b98ddbc9e763e384ce60a'
}
$wheelRecords = @(Get-ChildItem -LiteralPath $wheelDir -Filter '*.whl' | ForEach-Object {
    $sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expectedHashes.ContainsKey($_.Name) -and $sha256 -ne $expectedHashes[$_.Name]) {
        throw ('Official SHA256 mismatch: ' + $_.Name)
    }
    [ordered]@{
        filename = $_.Name
        bytes = $_.Length
        sha256 = $sha256
        official_expected_sha256 = $expectedHashes[$_.Name]
    }
})
$wheelRecords | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $installRoot 'install-wheels.json') -Encoding utf8
$wheelPaths = @(Get-ChildItem -LiteralPath $wheelDir -Filter '*.whl' | Select-Object -ExpandProperty FullName)
Invoke-RecordedPip -Step 'packages' -PipArguments (@(
    '--disable-pip-version-check', '--no-cache-dir', 'install', '--no-index', '--no-deps', '--ignore-installed',
    '--report', (Join-Path $installRoot 'install-report.json')
) + $wheelPaths)

Write-Output 'Installation finished. Framework import and GPU checks are intentionally delegated to the exclusive execution worker.'
