$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskSolver=Join-Path $taskRoot 'work\opencourant-experiment\solver\OpenCourant'
$taskSource=Join-Path $PSScriptRoot '500us-run'
$taskRun=Join-Path $PSScriptRoot '500us-resume03-run'
$taskEvidence=Join-Path $taskRoot 'outputs\expanded-research\benchmarks\drop_test\500us-resume03'
if(Test-Path -LiteralPath (Join-Path $taskRun 'engine.stdout.log')){throw 'Preserve prior500us attempt; resume from its checkpoint instead of overwriting it'}
New-Item -ItemType Directory -Force -Path $taskRun,$taskEvidence | Out-Null
$taskExpectedCheckpoint='8434daa6788ff7ca094eb3ad02270e9cb06281ca162affca1a335cd67bd58aca'
$taskExpectedEngine='13a9d6551fb9206b6c27c8af819e752edf3550e4725e446a2c3042c6c1a2b9d1'
if((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskSource 'Cell_Phone_Drop_0003_0001_I.rst')).Hash.ToLowerInvariant() -ne $taskExpectedCheckpoint){throw '117.6us checkpoint changed'}
if((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskSolver 'exec\engine_win64_sp.exe')).Hash.ToLowerInvariant() -ne $taskExpectedEngine){throw 'Engine binary changed'}
Copy-Item -LiteralPath (Join-Path $taskSource 'Cell_Phone_Drop_0003_0001_I.rst'),(Join-Path $taskSource 'Cell_Phone_Drop_0000.rad') -Destination $taskRun
@'
# User-authorized center continuation117.6us to500us. Output changes only.
/ANIM/DT
0.0 1e-5
/ANIM/BRICK/TENS/STRESS/111
/ANIM/BRICK/TENS/STRESS/112
/ANIM/BRICK/TENS/STRESS/121
/ANIM/BRICK/TENS/STRESS/211
/ANIM/BRICK/TENS/STRAIN/111
/ANIM/BRICK/TENS/STRAIN/112
/ANIM/BRICK/TENS/STRAIN/121
/ANIM/BRICK/TENS/STRAIN/211
/ANIM/BRICK/VONM
/ANIM/BRICK/EPSP
/ANIM/GPS/TENS
/ANIM/VECT/VEL
/DT1TET10
/NEGVOL/STOP
/PARITH/OFF
0.99 0.05 0 1 1 1
/TFILE/0
1e-7
/PRINT/-500
/RFILE/2
2000
/RUN/Cell_Phone_Drop/4/I
5e-4
'@ | Set-Content -LiteralPath (Join-Path $taskRun 'Cell_Phone_Drop_0004.rad') -Encoding ascii
"/TIME/4.9999e-4`n/ANIM" | Set-Content -LiteralPath (Join-Path $taskRun 'Cell_Phone_Drop_0004.ctl') -Encoding ascii
$taskPlan=[ordered]@{status='USER_AUTHORIZED_LOCAL_FEM_CONTINUATION';source_checkpoint_sha256=$taskExpectedCheckpoint;engine_binary_sha256=$taskExpectedEngine;source_actual_time_s=0.0001176364;requested_final_time_s=0.0005;engine_number=4;threads=2;priority='BelowNormal';process_memory_cap_MiB=600;system_free_memory_graceful_stop_MiB=256;preflight_free_memory_MiB=1280;wall_cap_seconds=7200;expected_wall_seconds_linear_estimate=3607.56;estimate_is_measurement=$false;checkpoint='RFILE2 every2000cycles, plus CHKPT/STOP if resource guard fires';output='10us animations;4validIP stress+strain;VM/EPSP/GPS/VEL;0.1us globalhistory';changed_physics='None. Original geometry/material/wall/velocity/gravity and20usAIcache unchanged.';source_checkpoint='work/expanded-drop/500us-run/Cell_Phone_Drop_0003_0001_I.rst';source_restart_boundary_check='../500us/restart-boundary-bitwise-check.json';supersedes_direction_status_of='../100us/500us-continuation-plan-NOT-EXECUTED.json';new_engine_file_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskRun 'Cell_Phone_Drop_0004.rad')).Hash.ToLowerInvariant()}
$taskPlan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskEvidence 'frozen-inputs-and-guards.json') -Encoding utf8
Copy-Item -LiteralPath (Join-Path $taskRun 'Cell_Phone_Drop_0004.rad'),(Join-Path $taskRun 'Cell_Phone_Drop_0004.ctl') -Destination $taskEvidence
do{
    $taskSystem=Get-CimInstance Win32_OperatingSystem
    $taskFreeKiB=$taskSystem.FreePhysicalMemory
    [ordered]@{status='waiting_for_1_25GiB_free_RAM';free_memory_KiB=$taskFreeKiB;free_virtual_memory_KiB=$taskSystem.FreeVirtualMemory} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'job-progress.json') -Encoding utf8
    if($taskFreeKiB -lt 1310720){Start-Sleep -Seconds 5}
}while($taskFreeKiB -lt 1310720)
$env:OPENCOURANT_PATH=$taskSolver
$env:RAD_CFG_PATH=Join-Path $taskSolver 'hm_cfg_files'
$env:RAD_H3D_PATH=Join-Path $taskSolver 'extlib\h3d\lib\win64'
$env:KMP_STACKSIZE='100m';$env:OMP_NUM_THREADS='2'
$env:PATH=(Join-Path $taskSolver 'extlib\hm_reader\win64')+';'+(Join-Path $taskSolver 'extlib\h3d\lib\win64')+';'+(Join-Path $taskSolver 'extlib\intelOneAPI_runtime\win64')+';'+$env:PATH
$taskWatch=[Diagnostics.Stopwatch]::StartNew()
$taskProcess=Start-Process -FilePath (Join-Path $taskSolver 'exec\engine_win64_sp.exe') -ArgumentList '-i Cell_Phone_Drop_0004.rad -nt 2' -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun 'engine.stdout.log') -RedirectStandardError (Join-Path $taskRun 'engine.stderr.log')
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
    if($null -eq $taskStopped -and ($taskPeak -gt 600MB -or $taskFreeKiB -lt 262144 -or $taskWatch.Elapsed.TotalSeconds -gt 7200)){
        $taskStopped=if($taskPeak -gt 600MB){'memory_cap_600_MiB'}elseif($taskFreeKiB -lt 262144){'system_free_below256MiB'}else{'wall_cap7200s'}
        "/ANIM`n/CHKPT`n/STOP" | Set-Content -LiteralPath (Join-Path $taskRun 'Cell_Phone_Drop_0004.ctl') -Encoding ascii
        $taskStopAt=$taskWatch.Elapsed.TotalSeconds
    }
    if($null -ne $taskStopAt -and $taskWatch.Elapsed.TotalSeconds -gt ($taskStopAt+30)){Stop-Process -Id $taskProcess.Id -Force;$taskStopped+='_forced_after30s_grace';break}
    Start-Sleep -Milliseconds 100
}
$taskProcess.WaitForExit();$taskWatch.Stop()
$taskMetrics=[ordered]@{stage='checkpoint_general_restart_to500us';exit_code=$taskProcess.ExitCode;stopped_reason=$taskStopped;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds;sampled_peak_working_set_bytes=$taskPeak;minimum_sampled_system_free_memory_KiB=$taskMinFreeKiB;minimum_sampled_system_free_virtual_memory_KiB=$taskMinVirtualKiB;omp_threads=2;priority='BelowNormal';requested_final_time_s=.0005;source_checkpoint_sha256=$taskExpectedCheckpoint;AI_targets_unchanged=$true}
$taskMetrics | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'execution-metrics.json') -Encoding utf8
Copy-Item -LiteralPath (Join-Path $taskRun 'engine.stdout.log'),(Join-Path $taskRun 'engine.stderr.log'),(Join-Path $taskRun 'Cell_Phone_Drop_0004.out') -Destination $taskEvidence
$taskStatus=if($taskProcess.ExitCode -eq 0 -and $null -eq $taskStopped -and (Select-String -LiteralPath (Join-Path $taskRun 'engine.stdout.log') -Pattern 'NORMAL TERMINATION' -Quiet)){'completed'}else{'stopped_needs_checkpoint_resume'}
[ordered]@{status=$taskStatus;metrics='execution-metrics.json';stopped_reason=$taskStopped;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskEvidence 'job-progress.json') -Encoding utf8
$taskMetrics | ConvertTo-Json


