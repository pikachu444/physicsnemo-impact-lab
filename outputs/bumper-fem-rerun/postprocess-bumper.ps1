$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskRun = Join-Path $taskRoot 'work\bumper-fem-rerun\run-extended'
$taskExeDir = Join-Path $taskRoot 'work\opencourant-experiment\solver\OpenCourant\exec'
$taskStageLog = @()
foreach ($taskFile in (Get-ChildItem -LiteralPath $taskRun -File | Where-Object {$_.Name -match '^Bumper_Beam_AP_meshedA\d{3}$'})) {
    $taskOutput = $taskFile.FullName + '.vtk'
    $taskP = Start-Process -FilePath (Join-Path $taskExeDir 'anim_to_vtk_win64.exe') -ArgumentList $taskFile.Name -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput $taskOutput -RedirectStandardError ($taskOutput + '.stderr.log')
    $taskP.WaitForExit()
    $taskP.Refresh()
    $taskStageLog += @{input=$taskFile.Name;converter='anim_to_vtk';exit_code=$taskP.ExitCode;output_bytes=(Get-Item -LiteralPath $taskOutput).Length}
    if ($taskP.ExitCode -ne 0) {throw ('VTK conversion failed: '+$taskFile.Name)}
}
$taskP = Start-Process -FilePath (Join-Path $taskExeDir 'th_to_csv_win64.exe') -ArgumentList 'Bumper_Beam_AP_meshedT01' -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun 'th-converter.stdout.log') -RedirectStandardError (Join-Path $taskRun 'th-converter.stderr.log')
$taskP.WaitForExit()
$taskP.Refresh()
$taskStageLog += @{input='Bumper_Beam_AP_meshedT01';converter='th_to_csv';exit_code=$taskP.ExitCode;output_bytes=(Get-Item -LiteralPath (Join-Path $taskRun 'Bumper_Beam_AP_meshedT01.csv')).Length}
if ($taskP.ExitCode -ne 0) {throw 'Time-history CSV conversion failed'}
$taskStageLog | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'conversion-metrics.json') -Encoding utf8
Write-Output ('Converted '+($taskStageLog.Count-1)+' actual FEM animation frames and one time-history file.')
