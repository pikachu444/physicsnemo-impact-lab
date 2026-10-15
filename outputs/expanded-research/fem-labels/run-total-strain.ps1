param([string]$RunDirectory='', [string]$ReportDirectory='')
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$taskRun = Join-Path $taskRoot 'work\expanded-fem-labels\run-total-strain'
if($RunDirectory){$taskRun=(Resolve-Path -LiteralPath $RunDirectory).Path}
$taskReport=$PSScriptRoot
if($ReportDirectory){New-Item -ItemType Directory -Force -Path $ReportDirectory|Out-Null;$taskReport=(Resolve-Path -LiteralPath $ReportDirectory).Path}
$taskSolver = Join-Path $taskRoot 'work\opencourant-experiment\solver\OpenCourant'
if(Test-Path -LiteralPath (Join-Path $taskRun 'engine.stdout.log')){throw 'Existing total-strain execution found; preserve this run instead of overwriting it.'}
$env:OPENCOURANT_PATH = $taskSolver
$env:RAD_CFG_PATH = Join-Path $taskSolver 'hm_cfg_files'
$env:RAD_H3D_PATH = Join-Path $taskSolver 'extlib\h3d\lib\win64'
$env:KMP_STACKSIZE = '400m'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:PATH = (Join-Path $taskSolver 'extlib\hm_reader\win64') + ';' + (Join-Path $taskSolver 'extlib\h3d\lib\win64') + ';' + (Join-Path $taskSolver 'extlib\intelOneAPI_runtime\win64') + ';' + $env:PATH
$taskStages=@(@{name='starter';exe='starter_win64_sp.exe';args='-i Bumper_Beam_AP_meshed_0000.rad -np 1 -nt 2'},@{name='engine';exe='engine_win64_sp.exe';args='-i Bumper_Beam_AP_meshed_0001.rad -nt 2'})
$taskResults=@()
foreach($taskStage in $taskStages){
    $taskWatch=[Diagnostics.Stopwatch]::StartNew()
    $taskProcess=Start-Process -FilePath (Join-Path $taskSolver ('exec\'+$taskStage.exe)) -ArgumentList $taskStage.args -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun ($taskStage.name+'.stdout.log')) -RedirectStandardError (Join-Path $taskRun ($taskStage.name+'.stderr.log'))
    $taskPeak=0L
    $taskCapped=$false
    while(-not $taskProcess.HasExited){
        try{$taskProcess.Refresh();$taskPeak=[Math]::Max($taskPeak,$taskProcess.PeakWorkingSet64)}catch{}
        if($taskWatch.Elapsed.TotalSeconds -gt 1200 -or $taskPeak -gt 600000000){$taskCapped=$true;Stop-Process -Id $taskProcess.Id -Force;break}
        Start-Sleep -Milliseconds 100
    }
    $taskProcess.WaitForExit();$taskProcess.Refresh();$taskWatch.Stop()
    $taskResults+=@{stage=$taskStage.name;executable=$taskStage.exe;arguments=$taskStage.args;exit_code=$taskProcess.ExitCode;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds;sampled_peak_working_set_bytes=$taskPeak;terminated_by_cap=$taskCapped;threads=2;wall_time_cap_seconds_per_stage=1200;memory_cap_bytes=600000000}
    $taskResults|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $taskReport 'execution-metrics.json') -Encoding utf8
    Write-Output ($taskResults[-1]|ConvertTo-Json -Compress)
    if($taskProcess.ExitCode -ne 0 -or $taskCapped){throw ('Solver stage failed/stopped: '+$taskStage.name)}
}
