param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('phone-full-c6','bumper-c7')]
    [string]$Dataset,
    [Parameter(Mandatory=$true)]
    [ValidateSet('transolver','flare','geotransolver','geoflare','meshgraphnet','figconvunet')]
    [string]$Architecture
)
$ErrorActionPreference='Continue'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $taskRoot
$env:PYTHONUTF8='1'
$env:TORCHDYNAMO_DISABLE='1'
$env:PHYSICSNEMO_FORCE_TE='False'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$taskPython=Join-Path $PSScriptRoot 'venv/Scripts/python.exe'
if($Dataset -eq 'phone-full-c6') {
    $taskManifest='work/expanded-ml/phone-full-cache-v2/manifest.json'
    $taskNormalization='work/expanded-ml/phone-full-normalization.json'
    $taskDigest='aa0848d5da1ce3ab103a03bcad6ae356896869357d5eec00cde82bb8d1ef1dca'
    $taskOutput="outputs/expanded-research/benchmarks/drop_test/probes-full-c6/$Architecture"
} else {
    $taskManifest='work/expanded-ml/multifield7-cache/manifest.json'
    $taskNormalization='work/expanded-ml/bumper7-normalization.json'
    $taskDigest='af937bc0798d9362e7e5298e5387099ac6aa305dccbee31b123a103c03c35e41'
    $taskOutput="outputs/expanded-research/benchmarks/bumper/probes-c7/$Architecture"
}
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
if($Dataset -eq 'phone-full-c6') {
    Copy-Item -LiteralPath 'outputs/expanded-research/benchmarks/drop_test/common-cache-provenance-clarification.json' -Destination "$taskOutput/data-provenance-clarification.json"
}
$taskArguments=@('-X','utf8','work/expanded-ml/run_benchmark.py','--cache',$taskManifest,'--out',$taskOutput,
    '--architecture',$Architecture,'--mode','probe','--probe-steps','3',
    '--require-ready','--expected-manifest-sha256',$taskDigest,'--normalization',$taskNormalization,
    '--workload-note','Single GPU owner for native support resource probe; no other model training. Public FEM training case only. User applications remain open; timings are on a shared workstation.')
$taskLog=Join-Path $PSScriptRoot "$Dataset-$Architecture-probe.log"
$taskStart=[DateTimeOffset]::UtcNow
& $taskPython @taskArguments *> $taskLog
$taskExitCode=$LASTEXITCODE
$taskFinish=[DateTimeOffset]::UtcNow
@{dataset=$Dataset;architecture=$Architecture;command=@($taskPython)+$taskArguments;working_directory=$taskRoot;
  started_utc=$taskStart.ToString('o');finished_utc=$taskFinish.ToString('o');
  seconds=($taskFinish-$taskStart).TotalSeconds;exit_code=$taskExitCode;log=$taskLog
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$taskOutput/execution-command.json"
Copy-Item -LiteralPath $taskLog -Destination "$taskOutput/run.log"
Copy-Item -LiteralPath $PSCommandPath -Destination "$taskOutput/executed_run_native_probe.ps1"
exit $taskExitCode
