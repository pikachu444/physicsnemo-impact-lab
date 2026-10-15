param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('transolver','flare','geotransolver','geoflare','meshgraphnet','figconvunet')]
    [string]$Architecture,
    [ValidateSet('train','fresh','predict')]
    [string]$Stage='train'
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
$taskManifest='work/expanded-structural-coupon/cache-initial13-nodal7/manifest.json'
$taskOutput="outputs/expanded-research/benchmarks/structural_coupon/models/$Architecture"
$taskCheckpoint="$taskOutput/best_checkpoint.pt"
$taskDigest='08876879b2b003f8222b831bb51a15c63f8d69d30e6b83e077385f80850169b1'
if($Stage -eq 'train') {
    $taskArguments=@('-X','utf8','work/expanded-ml/run_benchmark.py','--cache',$taskManifest,'--out',$taskOutput,
        '--architecture',$Architecture,'--mode','train','--epochs','1000','--min-epochs','100','--patience','100',
        '--require-ready','--expected-manifest-sha256',$taskDigest,'--normalization','work/expanded-ml/coupon13-normalization.json',
        '--workload-note','Single GPU owner. Concurrent CPU OpenRadioss phone simulations and local baseline/cache work; user applications remain open.')
} elseif($Stage -eq 'fresh') {
    $taskArguments=@('-X','utf8',"$taskOutput/source/run_benchmark.py",'--cache',$taskManifest,'--out',"$taskOutput/fresh-inference",
        '--mode','infer','--checkpoint',$taskCheckpoint,'--reference',"$taskOutput/predictions.npy",
        '--require-ready','--expected-manifest-sha256',$taskDigest)
} else {
    $taskArguments=@('-X','utf8','work/expanded-ml/predict_inputs.py','--checkpoint',$taskCheckpoint,
        '--inputs','work/expanded-ml/coupon-heldout-inputs-only.npz','--out',"$taskOutput/inputs-only-inference")
}
$taskLog=Join-Path $PSScriptRoot "coupon-$Architecture-$Stage.log"
$taskStart=[DateTimeOffset]::UtcNow
& $taskPython @taskArguments *> $taskLog
$taskExitCode=$LASTEXITCODE
$taskFinish=[DateTimeOffset]::UtcNow
@{architecture=$Architecture;stage=$Stage;command=@($taskPython)+$taskArguments;working_directory=$taskRoot;
  started_utc=$taskStart.ToString('o');finished_utc=$taskFinish.ToString('o');
  seconds=($taskFinish-$taskStart).TotalSeconds;exit_code=$taskExitCode;log=$taskLog
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $PSScriptRoot "coupon-$Architecture-$Stage-command.json")
exit $taskExitCode
