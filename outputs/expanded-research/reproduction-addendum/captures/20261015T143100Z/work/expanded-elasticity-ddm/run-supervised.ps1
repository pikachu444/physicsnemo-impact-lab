param([string]$OutputDirectory='')
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskOutput=Join-Path $taskRoot 'outputs\expanded-research\benchmarks\elasticity_numerical\full-ddm\attempt02'
if($OutputDirectory){$taskOutput=$OutputDirectory}
$taskPython='C:\Program Files\Python313\python.exe'
$taskMinimum=[long](1.5*1GB)
$taskAvailable=[long](& $taskPython -c 'import psutil;print(psutil.virtual_memory().available)')
if($LASTEXITCODE -ne 0){throw 'Memory preflight failed.'}
if($taskAvailable -lt $taskMinimum){Write-Output (@{status='waiting_for_memory_gate';available_bytes=$taskAvailable;minimum_bytes=$taskMinimum;framework_imported=$false}|ConvertTo-Json -Compress);exit 2}
Start-Sleep -Seconds 10
$taskAvailableSecond=[long](& $taskPython -c 'import psutil;print(psutil.virtual_memory().available)')
if($LASTEXITCODE -ne 0){throw 'Second memory preflight failed.'}
if($taskAvailableSecond -lt $taskMinimum){Write-Output (@{status='waiting_for_second_memory_gate';first_available_bytes=$taskAvailable;second_available_bytes=$taskAvailableSecond;minimum_bytes=$taskMinimum;framework_imported=$false}|ConvertTo-Json -Compress);exit 2}
if(Test-Path -LiteralPath (Join-Path $taskOutput 'execution.json')){throw 'Preserve completed execution record.'}
if(Test-Path -LiteralPath (Join-Path $taskOutput 'driver.stdout.log')){throw 'Preserve previous launch logs.'}
New-Item -ItemType Directory -Force -Path $taskOutput|Out-Null
$taskOutput=(Resolve-Path -LiteralPath $taskOutput).Path
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:NUMEXPR_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='-1';$env:PYTHONDONTWRITEBYTECODE='1';$env:MPLBACKEND='Agg'
$taskDriver=Join-Path $PSScriptRoot 'run-full-ddm.py'
$taskArguments=@(('"'+$taskDriver+'"'),'--execute','--out',('"'+$taskOutput+'"'))
$taskWatch=[Diagnostics.Stopwatch]::StartNew()
$taskStartedUtc=[DateTime]::UtcNow.ToString('o')
$taskProcess=Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'driver.stdout.log') -RedirectStandardError (Join-Path $taskOutput 'driver.stderr.log')
$taskPeak=0L;$taskMinimumObserved=[Math]::Min($taskAvailable,$taskAvailableSecond);$taskStopReason=$null
try{
    $taskProcess.PriorityClass=[Diagnostics.ProcessPriorityClass]::BelowNormal
    Write-Output (@{status='cpu_full_ddm_launched';pid=$taskProcess.Id;started_utc=$taskStartedUtc;available_before_bytes=$taskAvailable;second_available_before_bytes=$taskAvailableSecond;cpu_threads=1;process_priority='BelowNormal';CUDA_VISIBLE_DEVICES='-1'}|ConvertTo-Json -Compress)
    while(-not $taskProcess.HasExited){
        $taskProcess.Refresh()
        $taskPeak=[Math]::Max($taskPeak,$taskProcess.PeakWorkingSet64)
        $taskFree=[long]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1KB)
        $taskMinimumObserved=[Math]::Min($taskMinimumObserved,$taskFree)
        if($taskWatch.Elapsed.TotalSeconds -gt 300){$taskStopReason='wall_time_cap_300_seconds'}
        if($taskPeak -gt 350MB){$taskStopReason='working_set_cap_350_MiB'}
        if($taskFree -lt 512MB){$taskStopReason='host_free_memory_floor_512_MiB'}
        if($taskStopReason){$taskProcess.Kill($true);break}
        Start-Sleep -Milliseconds 200
    }
}finally{
    if(-not $taskProcess.HasExited){$taskProcess.Kill($true);$taskStopReason='supervisor_cleanup'}
    $taskProcess.WaitForExit();$taskProcess.Refresh();$taskWatch.Stop()
}
$taskRecord=@{stage='actual_official_numerical_2x2_ddm';resource_protocol_version=3;pid=$taskProcess.Id;started_utc=$taskStartedUtc;ended_utc=[DateTime]::UtcNow.ToString('o');observed_exit_code=$taskProcess.ExitCode;terminated_by_guard=[bool]$taskStopReason;stop_reason=$taskStopReason;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds;sampled_peak_working_set_bytes=$taskPeak;working_set_cap_bytes=350MB;wall_cap_seconds=300;host_free_memory_floor_bytes=512MB;minimum_observed_free_bytes=$taskMinimumObserved;available_before_launch_bytes=$taskAvailable;second_available_before_launch_bytes=$taskAvailableSecond;stable_gate_interval_seconds=10;required_available_bytes=$taskMinimum;cpu_threads=1;process_priority='BelowNormal';CUDA_VISIBLE_DEVICES='-1';python_path=$taskPython;process_reaped=$taskProcess.HasExited;concurrent_performance_only=$true}
$taskRecord|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $taskOutput 'supervisor.json') -Encoding utf8
Write-Output ($taskRecord|ConvertTo-Json -Compress)
if($taskProcess.ExitCode -ne 0 -or $taskStopReason){exit 1}
