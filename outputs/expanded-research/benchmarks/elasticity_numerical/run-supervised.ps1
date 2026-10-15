param([string]$OutputDirectory='', [double]$MaxSeconds=300, [long]$MaxWorkingSetBytes=2000000000, [string]$PythonPath='C:\Program Files\Python313\python.exe')
$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')).Path
$taskOutput=Join-Path $PSScriptRoot 'cpu-execution'
if($OutputDirectory){$taskOutput=$OutputDirectory}
$taskPython=$PythonPath
$taskAvailable=[long](& $taskPython -c 'import psutil;print(psutil.virtual_memory().available)')
if($LASTEXITCODE -ne 0){throw 'Available memory preflight failed without importing Torch.'}
if($taskAvailable -lt 1073741824){Write-Output (@{status='deferred_low_memory';available_memory_bytes=$taskAvailable;minimum_available_memory_bytes=1073741824;torch_imported=$false}|ConvertTo-Json -Compress);exit 2}
if(Test-Path -LiteralPath $taskOutput){throw 'Select a new output directory; preserve previous run records.'}
New-Item -ItemType Directory -Path $taskOutput|Out-Null
$taskOutput=(Resolve-Path -LiteralPath $taskOutput).Path
$taskDriver=Join-Path $PSScriptRoot 'run-official-q1.py'
$env:OMP_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OPENBLAS_NUM_THREADS='1';$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUDA_VISIBLE_DEVICES='-1'
$taskWatch=[Diagnostics.Stopwatch]::StartNew()
$taskArguments=@(('"'+$taskDriver+'"'),'--execute','--device','cpu','--out',('"'+$taskOutput+'"'))
$taskProcess=Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'driver.stdout.log') -RedirectStandardError (Join-Path $taskOutput 'driver.stderr.log')
$taskPeak=0L;$taskCapped=$false
while(-not $taskProcess.HasExited){
    try{$taskProcess.Refresh();$taskPeak=[Math]::Max($taskPeak,$taskProcess.PeakWorkingSet64)}catch{}
    if($taskWatch.Elapsed.TotalSeconds -gt $MaxSeconds -or $taskPeak -gt $MaxWorkingSetBytes){$taskCapped=$true;Stop-Process -Id $taskProcess.Id -Force;break}
    Start-Sleep -Milliseconds 100
}
$taskProcess.WaitForExit();$taskProcess.Refresh();$taskWatch.Stop()
$taskRecord=@{stage='official_q1_cpu_validation';observed_exit_code=$taskProcess.ExitCode;terminated_by_cap=$taskCapped;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds;sampled_peak_working_set_bytes=$taskPeak;wall_cap_seconds=$MaxSeconds;working_set_cap_bytes=$MaxWorkingSetBytes;cpu_threads=1;GPU_used=$false;available_memory_before_launch_bytes=$taskAvailable;python_environment='preserved base Python, not GPU venv';CUDA_VISIBLE_DEVICES='-1'}
$taskRecord|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $taskOutput 'supervisor.json') -Encoding utf8
Write-Output ($taskRecord|ConvertTo-Json -Compress)
if($taskProcess.ExitCode -ne 0 -or $taskCapped){exit 1}
