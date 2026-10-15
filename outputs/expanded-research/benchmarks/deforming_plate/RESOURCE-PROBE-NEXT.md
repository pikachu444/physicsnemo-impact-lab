# Plate official-capacity resource measurement — default completed

After the root assigned an exclusive GPU slot, the default width128/processor15 attempt completed on 2026-10-15 at 04:15:18 UTC. The process exited and the GPU slot was returned. See official128-resource-summary.md for measured resources. Code preparation had been kept separate from execution; this file now records the completed command and the remaining optional optimization path.

## Default official capacity

Use a fresh output directory and fresh log filename:

```powershell
$env:PHYSICSNEMO_FORCE_TE = 'False'
& work/expanded-ml/venv/Scripts/python.exe -X utf8 work/expanded-plate/run_resource_probe.py --output outputs/expanded-research/benchmarks/deforming_plate/probe128_default --hidden 128 --processor 15 --checkpoint-segments 0 --probe 3 --probe-trajectory train_0011 --probe-frame 319 --train-count 32 --valid-count 6 --test-start 2 --test-stop 8 *> outputs/expanded-research/benchmarks/deforming_plate/probe128_default.log
```

Three timed Adam updates follow two warm-up Adam updates. Separate three-sample forward timing uses two warm-up forward passes per timing mode; it performs no additional Adam update or held-out accuracy evaluation. This is a resource test, not accuracy training. The selected training geometry has 1,488 nodes and 60,298 directed edges at frame 319.

`resource-attempt.json` records completed/failed/interrupted status, source hashes, exact settings, measured parameter count, elapsed wall time including imports/setup, 20ms RSS samples, Windows high-water fields when available, Torch tensor-allocated/allocator-reserved peak memory, and actual timing. It preserves traceback/OOM details on failure. Existing output directories are refused before framework import. A default-capacity failure remains evidence and is not relabeled as a successful optimized run.

## Activation-checkpointing attempt, only if separately chosen

A new output directory such as `probe128_checkpoint15` and a distinct log are required. Change only `--checkpoint-segments 15`, preserve the remaining architecture and data settings, and report this explicitly as a separate memory optimization measurement. This attempt has not run either.

## Training-cap guide after a successful measurement

```powershell
& work/expanded-ml/venv/Scripts/python.exe -X utf8 work/expanded-plate/training_time_guide.py --probe-dir outputs/expanded-research/benchmarks/deforming_plate/probe128_default --split-contract outputs/expanded-research/benchmarks/deforming_plate/final-split-contract.json --output outputs/expanded-research/benchmarks/deforming_plate/probe128_default/time-guide.json
```

The guide is standard-library only and refuses a failed/incomplete or differently sized model measurement. It verifies the attempt/config/probe hashes and emits nine candidate budgets for 32 training trajectories (8/16/32 step anchors × 5/10/20 epochs). It also accounts for six internal-validation trajectories and six fresh-final-test trajectories, including 399 AR steps plus a separate 399-step teacher-forced evaluation per trajectory. It does not select a budget. Data I/O, checkpoint serialization, strain calculation, plotting and OS paging remain outside the timing proxies; the actual first epoch and validation curve must refine the selected cap.

Original test0/1 are development. Fresh original test2..7 remain unevaluated and must not drive architecture, data-count, epoch or checkpoint decisions. Neither this resource measurement nor its time guide opens fresh-test prediction accuracy.

The code received an independent read-only review (`review/resource-wrapper-review.md`). A subsequent actual default resource run succeeded, establishing that this sampled width128/15 training step fits the measured GPU. It does not establish useful accuracy. The activation-checkpointing variant and expanded accuracy training remain unexecuted.
