$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskWork = Join-Path $taskRoot 'work\bumper-fem-rerun'
$taskSolver = Join-Path $taskRoot 'work\opencourant-experiment\solver\OpenCourant'
$taskRun = Join-Path $taskWork 'run-extended'
New-Item -ItemType Directory -Force -Path $taskRun | Out-Null
foreach ($taskName in @('Bumper_Beam_AP_meshed_0000.rad','Bumper_Beam_AP_meshed_0001.rad')) {
    Copy-Item -LiteralPath (Join-Path $taskWork $taskName) -Destination $taskRun -Force
}
$env:OPENCOURANT_PATH = $taskSolver
$env:RAD_CFG_PATH = Join-Path $taskSolver 'hm_cfg_files'
$env:RAD_H3D_PATH = Join-Path $taskSolver 'extlib\h3d\lib\win64'
$env:KMP_STACKSIZE = '400m'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:PATH = (Join-Path $taskSolver 'extlib\hm_reader\win64') + ';' + (Join-Path $taskSolver 'extlib\h3d\lib\win64') + ';' + (Join-Path $taskSolver 'extlib\intelOneAPI_runtime\win64') + ';' + $env:PATH
$taskStages = @(
    @{ Name = 'starter'; Exe = 'starter_win64_sp.exe'; Args = '-i Bumper_Beam_AP_meshed_0000.rad -np 1 -nt 2' },
    @{ Name = 'engine'; Exe = 'engine_win64_sp.exe'; Args = '-i Bumper_Beam_AP_meshed_0001.rad -nt 2' }
)
$taskResults = @()
foreach ($taskStage in $taskStages) {
    $taskWatch = [System.Diagnostics.Stopwatch]::StartNew()
    $taskProcess = Start-Process -FilePath (Join-Path $taskSolver ('exec\' + $taskStage.Exe)) -ArgumentList $taskStage.Args -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun ($taskStage.Name + '.stdout.log')) -RedirectStandardError (Join-Path $taskRun ($taskStage.Name + '.stderr.log'))
    $taskPeak = 0L
    $taskTerminated = $false
    while (-not $taskProcess.HasExited) {
        try {
            $taskProcess.Refresh()
            $taskPeak = [Math]::Max($taskPeak, $taskProcess.PeakWorkingSet64)
        } catch { }
        if ($taskWatch.Elapsed.TotalSeconds -gt 1200 -or $taskPeak -gt 600000000) {
            $taskTerminated = $true
            Stop-Process -Id $taskProcess.Id -Force
            break
        }
        Start-Sleep -Milliseconds 100
    }
    $taskProcess.WaitForExit()
    $taskProcess.Refresh()
    $taskWatch.Stop()
    $taskResults += [ordered]@{ attempt = 2; stage = $taskStage.Name; executable = $taskStage.Exe; arguments = $taskStage.Args; exit_code = $taskProcess.ExitCode; elapsed_seconds = $taskWatch.Elapsed.TotalSeconds; sampled_peak_working_set_bytes = $taskPeak; terminated_by_resource_cap = $taskTerminated; threads = 2; memory_cap_bytes = 600000000; wall_time_cap_seconds_per_stage = 1200 }
    $taskResults | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'execution-metrics.json') -Encoding utf8
    Write-Output ($taskResults[-1] | ConvertTo-Json -Compress)
    if ($taskProcess.ExitCode -ne 0 -or $taskTerminated) { throw ('Stage failed or stopped: ' + $taskStage.Name) }
}
