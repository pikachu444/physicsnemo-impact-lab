$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskSolver=Join-Path $taskRoot 'work\opencourant-experiment\solver\OpenCourant'
$taskSource=Join-Path $PSScriptRoot '500us-run'
$taskRun=Join-Path $PSScriptRoot '500us-resume02-run'
$taskEvidence=Join-Path $taskRoot 'outputs\expanded-research\benchmarks\drop_test\500us-resume02'
if(Test-Path -LiteralPath (Join-Path $taskRun 'engine.stdout.log')){throw 'Prior resume attempt exists: preserve it'}
New-Item -ItemType Directory -Force -Path $taskRun,$taskEvidence | Out-Null
$taskExpectedCheckpoint='8434daa6788ff7ca094eb3ad02270e9cb06281ca162affca1a335cd67bd58aca'
$taskExpectedEngine='13a9d6551fb9206b6c27c8af819e752edf3550e4725e446a2c3042c6c1a2b9d1'
if((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskSource 'Cell_Phone_Drop_0003_0001_I.rst')).Hash.ToLowerInvariant() -ne $taskExpectedCheckpoint){throw 'Checkpoint changed'}
if((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskSolver 'exec\engine_win64_sp.exe')).Hash.ToLowerInvariant() -ne $taskExpectedEngine){throw 'Engine binary changed'}
Copy-Item -LiteralPath (Join-Path $taskSource 'Cell_Phone_Drop_0003_0001_I.rst') -Destination (Join-Path $taskRun 'Cell_Phone_Drop_0003_0001.rst')
Copy-Item -LiteralPath (Join-Path $taskSource 'Cell_Phone_Drop_0000.rad'),(Join-Path $taskSource 'Cell_Phone_Drop_0003.rad'),(Join-Path $taskSource 'Cell_Phone_Drop_0003.out'),(Join-Path $taskSource 'Cell_Phone_DropT03') -Destination $taskRun
'/RERUN/Cell_Phone_Drop/3' | Set-Content -LiteralPath (Join-Path $taskRun 'CHECK_DATA') -Encoding ascii
"/TIME/4.9999e-4`n/ANIM" | Set-Content -LiteralPath (Join-Path $taskRun 'Cell_Phone_Drop_0003.ctl') -Encoding ascii
[ordered]@{status='USER_AUTHORIZED_EXACT_CHECKPOINT_RESUME';source_checkpoint_sha256=$taskExpectedCheckpoint;engine_binary_sha256=$taskExpectedEngine;source_run='500us-run';source_checkpoint_file='Cell_Phone_Drop_0003_0001_I.rst';copied_checkpoint_file='Cell_Phone_Drop_0003_0001.rst';invocation='engine_win64_sp.exe -i CHECK_DATA -nt 2';engine_number=3;restart_mode='RERUN MCHECK=1 preserves TT DT NCYCLE output state';requested_final_time_s=0.0005;threads=2;priority='BelowNormal';process_memory_cap_MiB=600;system_free_memory_graceful_stop_MiB=256;preflight_free_memory_MiB=1280;wall_cap_seconds=10800;changed_physics='None';AI_targets_unchanged=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'frozen-inputs-and-guards.json') -Encoding utf8
Copy-Item -LiteralPath (Join-Path $taskRun 'CHECK_DATA'),(Join-Path $taskRun 'Cell_Phone_Drop_0003.rad'),(Join-Path $taskRun 'Cell_Phone_Drop_0003.ctl') -Destination $taskEvidence
$taskSystem=Get-CimInstance Win32_OperatingSystem
while($taskSystem.FreePhysicalMemory -lt 1310720){[ordered]@{status='waiting_for_1_25GiB_free_RAM';free_memory_KiB=$taskSystem.FreePhysicalMemory;free_virtual_memory_KiB=$taskSystem.FreeVirtualMemory} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'job-progress.json') -Encoding utf8;Start-Sleep -Seconds 5;$taskSystem=Get-CimInstance Win32_OperatingSystem}
$env:OPENCOURANT_PATH=$taskSolver
$env:RAD_CFG_PATH=Join-Path $taskSolver 'hm_cfg_files'
$env:RAD_H3D_PATH=Join-Path $taskSolver 'extlib\h3d\lib\win64'
$env:KMP_STACKSIZE='100m';$env:OMP_NUM_THREADS='2'
$env:PATH=(Join-Path $taskSolver 'extlib\hm_reader\win64')+';'+(Join-Path $taskSolver 'extlib\h3d\lib\win64')+';'+(Join-Path $taskSolver 'extlib\intelOneAPI_runtime\win64')+';'+$env:PATH
$taskWatch=[Diagnostics.Stopwatch]::StartNew()
$taskProcess=Start-Process -FilePath (Join-Path $taskSolver 'exec\engine_win64_sp.exe') -ArgumentList '-i CHECK_DATA -nt 2' -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun 'engine.stdout.log') -RedirectStandardError (Join-Path $taskRun 'engine.stderr.log')
$taskHandle=$taskProcess.Handle
$taskProcess.PriorityClass='BelowNormal'
[ordered]@{wrapper_PID=$PID;engine_PID=$taskProcess.Id;owned_run_directory=$taskRun;priority=$taskProcess.PriorityClass.ToString();started_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'process-ownership.json') -Encoding utf8
$taskPeak=0L;$taskStopped=$null;$taskStopAt=$null;$taskNextCheck=0.0;$taskMinFreeKiB=[long]::MaxValue;$taskMinVirtualKiB=[long]::MaxValue
while(-not $taskProcess.HasExited){
 $taskProcess.Refresh();$taskPeak=[Math]::Max($taskPeak,$taskProcess.PeakWorkingSet64)
 if($taskWatch.Elapsed.TotalSeconds -gt $taskNextCheck){
  $taskSystem=Get-CimInstance Win32_OperatingSystem;$taskFreeKiB=$taskSystem.FreePhysicalMemory
  $taskMinFreeKiB=[Math]::Min($taskMinFreeKiB,$taskFreeKiB);$taskMinVirtualKiB=[Math]::Min($taskMinVirtualKiB,$taskSystem.FreeVirtualMemory)
  $taskNextCheck=$taskWatch.Elapsed.TotalSeconds+2
  [ordered]@{status=if($null -eq $taskStopped){'running'}else{'graceful_checkpoint_stop_requested'};pid=$taskProcess.Id;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds;peak_working_set_bytes=$taskPeak;free_memory_KiB=$taskFreeKiB;free_virtual_memory_KiB=$taskSystem.FreeVirtualMemory;stop_reason=$taskStopped} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'job-progress.json') -Encoding utf8
 }
 if($null -eq $taskStopped -and ($taskPeak -gt 600MB -or $taskFreeKiB -lt 262144 -or $taskWatch.Elapsed.TotalSeconds -gt 10800)){
  $taskStopped=if($taskPeak -gt 600MB){'memory_cap_600_MiB'}elseif($taskFreeKiB -lt 262144){'system_free_below256MiB'}else{'wall_cap10800s'}
  "/ANIM`n/CHKPT`n/STOP" | Set-Content -LiteralPath (Join-Path $taskRun 'Cell_Phone_Drop_0003.ctl') -Encoding ascii
  $taskStopAt=$taskWatch.Elapsed.TotalSeconds
 }
 if($null -ne $taskStopAt -and $taskWatch.Elapsed.TotalSeconds -gt ($taskStopAt+30)){Stop-Process -Id $taskProcess.Id -Force;$taskStopped+='_forced_after30s_grace';break}
 Start-Sleep -Milliseconds 100
}
$taskProcess.WaitForExit();$taskWatch.Stop()
$taskMetrics=[ordered]@{stage='exact_checkpoint_resume_to500us';exit_code=$taskProcess.ExitCode;stopped_reason=$taskStopped;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds;sampled_peak_working_set_bytes=$taskPeak;minimum_sampled_system_free_memory_KiB=$taskMinFreeKiB;minimum_sampled_system_free_virtual_memory_KiB=$taskMinVirtualKiB;omp_threads=2;priority='BelowNormal';requested_final_time_s=.0005;source_checkpoint_sha256=$taskExpectedCheckpoint;AI_targets_unchanged=$true}
$taskMetrics | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'execution-metrics.json') -Encoding utf8
Copy-Item -LiteralPath (Join-Path $taskRun 'engine.stdout.log'),(Join-Path $taskRun 'engine.stderr.log'),(Join-Path $taskRun 'Cell_Phone_Drop_0003.out') -Destination $taskEvidence
$taskStatus=if($taskProcess.ExitCode -eq 0 -and $null -eq $taskStopped -and (Select-String -LiteralPath (Join-Path $taskRun 'engine.stdout.log') -Pattern 'NORMAL TERMINATION' -Quiet)){'completed'}else{'stopped_needs_checkpoint_resume'}
[ordered]@{status=$taskStatus;metrics='execution-metrics.json';stopped_reason=$taskStopped;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'job-progress.json') -Encoding utf8
$taskMetrics | ConvertTo-Json
