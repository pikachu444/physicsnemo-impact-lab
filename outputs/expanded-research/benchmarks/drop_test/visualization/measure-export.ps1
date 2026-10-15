$ErrorActionPreference = 'Stop'
$taskPython = (Get-Command python).Source
$taskWatch = [Diagnostics.Stopwatch]::StartNew()
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList (Join-Path $PSScriptRoot 'write_phone_vtu.py') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $PSScriptRoot 'export.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'export.stderr.log')
$taskRetainedHandle = $taskProcess.Handle
$taskPeak=0L
while (-not $taskProcess.HasExited) {
    $taskProcess.Refresh();$taskPeak=[Math]::Max($taskPeak,$taskProcess.PeakWorkingSet64)
    Start-Sleep -Milliseconds 20
}
$taskProcess.WaitForExit();$taskWatch.Stop()
$taskRecord=[ordered]@{stage='two_testcase_nominal20us_FEM_VTU_export';exit_code=$taskProcess.ExitCode;elapsed_seconds=$taskWatch.Elapsed.TotalSeconds;sampled_peak_working_set_bytes=$taskPeak;BLAS_threads=1;GPU_used=$false;source_cache_modified=$false}
$taskRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'execution-metrics.json') -Encoding utf8
if ($taskProcess.ExitCode -ne 0) {throw 'VTU export failed'}
$taskRecord | ConvertTo-Json
