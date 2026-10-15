$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskBase=Join-Path $taskRoot 'outputs\expanded-research\benchmarks\drop_test'
$taskDestination=Join-Path $taskBase '500us-final'
$taskSummary=Get-Content -LiteralPath (Join-Path $taskDestination 'summary.json') -Raw | ConvertFrom-Json
if(-not $taskSummary.completed_to_requested_time){throw 'Finish the authorized500us FEM first'}
$taskSource=Split-Path (Join-Path $taskRoot $taskSummary.last_frame.file)
$taskRun=Join-Path $PSScriptRoot '500us-final-boundary-run'
if(Test-Path -LiteralPath $taskRun){throw 'Preserve existing final boundary verification folder'}
$taskPrevious=Get-ChildItem -LiteralPath $taskSource -Filter 'Cell_Phone_Drop_*.rad' | Where-Object {$_.BaseName -ne 'Cell_Phone_Drop_0000'} | Sort-Object Name | Select-Object -Last 1
$taskPreviousNumber=[int]($taskPrevious.BaseName -split '_')[-1]
$taskPreviousTag='{0:d4}' -f $taskPreviousNumber
$taskNextNumber=$taskPreviousNumber+1
$taskNextTag='{0:d4}' -f $taskNextNumber
$taskCheckpoint=Get-ChildItem -LiteralPath $taskSource -Filter ('Cell_Phone_Drop_'+$taskPreviousTag+'_0001*.rst') | Sort-Object LastWriteTimeUtc | Select-Object -Last 1
if($null -eq $taskCheckpoint){throw 'No final checkpoint'}
$taskSolver=Join-Path $taskRoot 'work\opencourant-experiment\solver\OpenCourant'
if((Get-FileHash -LiteralPath (Join-Path $taskSolver 'exec\engine_win64_sp.exe') -Algorithm SHA256).Hash.ToLowerInvariant() -ne '13a9d6551fb9206b6c27c8af819e752edf3550e4725e446a2c3042c6c1a2b9d1'){throw 'Solver build changed'}
$taskSystem=Get-CimInstance Win32_OperatingSystem
while($taskSystem.FreePhysicalMemory -lt 1048576){Start-Sleep -Seconds 5;$taskSystem=Get-CimInstance Win32_OperatingSystem}
New-Item -ItemType Directory -Path $taskRun | Out-Null
Copy-Item -LiteralPath $taskCheckpoint.FullName -Destination (Join-Path $taskRun ('Cell_Phone_Drop_'+$taskPreviousTag+'_0001.rst'))
Copy-Item -LiteralPath (Join-Path $taskSource 'Cell_Phone_Drop_0000.rad') -Destination $taskRun
$taskInput=(Get-Content -LiteralPath $taskPrevious.FullName -Raw).Replace(('/RUN/Cell_Phone_Drop/'+$taskPreviousNumber+'/I'),('/RUN/Cell_Phone_Drop/'+$taskNextNumber+'/')).Replace(('/RUN/Cell_Phone_Drop/'+$taskPreviousNumber+'/J'),('/RUN/Cell_Phone_Drop/'+$taskNextNumber+'/'))
if(-not $taskInput.Contains('/RUN/Cell_Phone_Drop/'+$taskNextNumber+'/')){throw 'Unexpected source input run card'}
$taskInput=$taskInput -replace '/RFILE/2\s+2000','/RFILE/OFF'
$taskInputPath=Join-Path $taskRun ('Cell_Phone_Drop_'+$taskNextTag+'.rad')
$taskInput | Set-Content -LiteralPath $taskInputPath -Encoding ascii
$env:OPENCOURANT_PATH=$taskSolver;$env:RAD_CFG_PATH=Join-Path $taskSolver 'hm_cfg_files';$env:RAD_H3D_PATH=Join-Path $taskSolver 'extlib\h3d\lib\win64'
$env:KMP_STACKSIZE='100m';$env:OMP_NUM_THREADS='2'
$env:PATH=(Join-Path $taskSolver 'extlib\hm_reader\win64')+';'+(Join-Path $taskSolver 'extlib\h3d\lib\win64')+';'+(Join-Path $taskSolver 'extlib\intelOneAPI_runtime\win64')+';'+$env:PATH
$taskWatch=[Diagnostics.Stopwatch]::StartNew()
$taskProcess=Start-Process -FilePath (Join-Path $taskSolver 'exec\engine_win64_sp.exe') -ArgumentList ('-i '+(Split-Path $taskInputPath -Leaf)+' -nt 2') -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun 'engine.stdout.log') -RedirectStandardError (Join-Path $taskRun 'engine.stderr.log')
$taskHandle=$taskProcess.Handle;$taskProcess.PriorityClass='BelowNormal';$taskPeak=0L;$taskReason=$null
while(-not $taskProcess.HasExited){
 $taskProcess.Refresh();$taskPeak=[Math]::Max($taskPeak,$taskProcess.PeakWorkingSet64);$taskSystem=Get-CimInstance Win32_OperatingSystem
 if($null -eq $taskReason -and ($taskPeak -gt 600MB -or $taskSystem.FreePhysicalMemory -lt 262144 -or $taskWatch.Elapsed.TotalSeconds -gt 120)){$taskReason='boundary_resource_guard';"/ANIM`n/CHKPT`n/STOP" | Set-Content -LiteralPath (Join-Path $taskRun ('Cell_Phone_Drop_'+$taskNextTag+'.ctl')) -Encoding ascii}
 if($taskWatch.Elapsed.TotalSeconds -gt 150){Stop-Process -Id $taskProcess.Id -Force;throw 'Boundary load exceeded150s'}
 Start-Sleep -Milliseconds 500
}
$taskProcess.WaitForExit();$taskWatch.Stop()
[ordered]@{stage='final_checkpoint_reload_only';exit_code=$taskProcess.ExitCode;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds;peak_working_set_bytes=$taskPeak;stop_reason=$taskReason;source_checkpoint=$taskCheckpoint.FullName;source_checkpoint_sha256=(Get-FileHash -LiteralPath $taskCheckpoint.FullName -Algorithm SHA256).Hash.ToLowerInvariant();engine_number=$taskNextNumber;threads=2;priority='BelowNormal';requested_stop_s=.0005;no_source_file_modified=$true} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskDestination 'final-boundary-engine-metrics.json') -Encoding utf8
if($taskProcess.ExitCode -ne 0 -or $null -ne $taskReason){throw 'Boundary engine load failed: preserve logs'}
Copy-Item -LiteralPath (Join-Path $taskRun 'engine.stdout.log') -Destination (Join-Path $taskDestination 'final-boundary-engine.stdout.log')

