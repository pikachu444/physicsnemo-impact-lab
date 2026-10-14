$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$taskSolver = Join-Path $taskRoot 'work\opencourant-experiment\solver\OpenCourant\exec'
$taskRun = Join-Path $PSScriptRoot 'run'
$taskP = Start-Process -FilePath (Join-Path $taskSolver 'th_to_csv_win64.exe') -ArgumentList 'BOX1BLT01' -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRun 'th-converter.stdout.log') -RedirectStandardError (Join-Path $taskRun 'th-converter.stderr.log')
$taskP.WaitForExit()
$taskP.Refresh()
if ($taskP.ExitCode -ne 0) { throw 'Time-history CSV conversion failed' }
foreach ($taskFile in (Get-ChildItem -LiteralPath $taskRun -Filter 'BOX1BLA*.gz')) {
    $taskUnzipped = Join-Path $taskRun ([IO.Path]::GetFileNameWithoutExtension($taskFile.Name))
    $taskIn = [IO.File]::OpenRead($taskFile.FullName)
    $taskOut = [IO.File]::Create($taskUnzipped)
    $taskGz = [IO.Compression.GZipStream]::new($taskIn, [IO.Compression.CompressionMode]::Decompress)
    try { $taskGz.CopyTo($taskOut) } finally { $taskGz.Dispose(); $taskOut.Dispose(); $taskIn.Dispose() }
    $taskP = Start-Process -FilePath (Join-Path $taskSolver 'anim_to_vtk_win64.exe') -ArgumentList ([IO.Path]::GetFileNameWithoutExtension($taskFile.Name)) -WorkingDirectory $taskRun -WindowStyle Hidden -PassThru -RedirectStandardOutput ($taskUnzipped + '.vtk') -RedirectStandardError ($taskUnzipped + '.vtk.stderr.log')
    $taskP.WaitForExit()
    $taskP.Refresh()
    if ($taskP.ExitCode -ne 0) { throw ('VTK conversion failed: ' + $taskFile.Name) }
}
Write-Output 'Converted 11 FEM animation frames to VTK and the time history to CSV.'
