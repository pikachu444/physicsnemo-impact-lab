$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskSolver = Join-Path $taskRoot 'work\opencourant-experiment\solver\OpenCourant'
$taskSource = Join-Path $taskRoot 'work\opencourant-experiment\box-crash'
$taskRun = Join-Path $PSScriptRoot 'run'
$taskOriginal = Join-Path $PSScriptRoot 'original-input'
New-Item -ItemType Directory -Force -Path $taskRun, $taskOriginal | Out-Null
Get-ChildItem -LiteralPath $taskSource -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $taskOriginal -Force
    Copy-Item -LiteralPath $_.FullName -Destination $taskRun -Force
}
$env:OPENCOURANT_PATH = $taskSolver
$env:RAD_CFG_PATH = Join-Path $taskSolver 'hm_cfg_files'
$env:RAD_H3D_PATH = Join-Path $taskSolver 'extlib\h3d\lib\win64'
$env:KMP_STACKSIZE = '400m'
$env:OMP_NUM_THREADS = '2'
$env:PATH = (Join-Path $taskSolver 'extlib\hm_reader\win64') + ';' + (Join-Path $taskSolver 'extlib\h3d\lib\win64') + ';' + (Join-Path $taskSolver 'extlib\intelOneAPI_runtime\win64') + ';' + $env:PATH
$taskStages = @(
    @{ Name = 'starter'; Exe = 'starter_win64.exe'; Args = '-i BOX1BL_0000.rad -np 1 -nt 2' },
    @{ Name = 'engine'; Exe = 'engine_win64.exe'; Args = '-i BOX1BL_0001.rad -nt 2' }
)
$taskResults = @()
foreach ($taskStage in $taskStages) {
    $taskWatch = [System.Diagnostics.Stopwatch]::StartNew()
    $taskProcess = Start-Process -FilePath (Join-Path $taskSolver ('exec\' + $taskStage.Exe)) -ArgumentList $taskStage.Args -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun ($taskStage.Name + '.stdout.log')) -RedirectStandardError (Join-Path $taskRun ($taskStage.Name + '.stderr.log'))
    $taskPeak = 0L
    while (-not $taskProcess.HasExited) {
        try {
            $taskProcess.Refresh()
            $taskPeak = [Math]::Max($taskPeak, $taskProcess.PeakWorkingSet64)
        } catch { }
        Start-Sleep -Milliseconds 50
    }
    $taskProcess.WaitForExit()
    $taskWatch.Stop()
    $taskResult = [ordered]@{ stage = $taskStage.Name; executable = $taskStage.Exe; arguments = $taskStage.Args; exit_code = $taskProcess.ExitCode; elapsed_seconds = $taskWatch.Elapsed.TotalSeconds; sampled_peak_working_set_bytes = $taskPeak }
    $taskResults += $taskResult
    $taskResults | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'execution-metrics.json') -Encoding utf8
    if ($taskProcess.ExitCode -ne 0) { throw ('Stage failed: ' + $taskStage.Name + ', exit=' + $taskProcess.ExitCode) }
}
$taskResults | ConvertTo-Json -Depth 6
