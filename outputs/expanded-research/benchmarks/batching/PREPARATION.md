# Native batch probe preparation

Prepared 2026-10-15. GPU execution is pending the scheduler's explicit slot. No Torch, NumPy, TensorFlow, PhysicsNeMo or CUDA import occurred during preparation. The source and stdlib-only validation are separate from the shared training implementation.

Source directory: `work/expanded-batch/`.

- `batch_adapters.py`: original six model classes via main `build_model`; native batched tensor layouts, original FEM graph blocks for MeshGraphNet, native FIG batch KNN with contiguous input and no duplicate offsets.
- `train_probe_data.py`: reads frozen metadata/normalization and only selected training inputs and targets. It deliberately avoids `benchmark_data.Cache`, whose constructor hashes all split labels.
- `run_batch_probe.py`: one fresh model/domain/batch subprocess; correctness, independence, gradient comparison, timings and exactly three Adam updates.
- `run_batch_suite.py`: ascending B1, B2, B5 by default. An OOM, failure, timeout or equivalence failure stops all larger batches for that model/domain. Other models remain isolated.
- `static_check.py`: AST compilation, metadata-only train selection and adversarial held-out alias/path traversal checks. It asserts no ML module was imported.

## Scheduler command

Use a new evidence directory each time. Run this only after the main agent allocates the GPU:

```powershell
& 'work/expanded-ml/venv/Scripts/python.exe' 'work/expanded-batch/run_batch_suite.py' --gpu-slot-authorized --domains bumper phone --batches 2 5 --out 'outputs/expanded-research/benchmarks/batching/allocated-slot-01'
```

The scheduler plans to precede this with main B1 functionality/resource probes. Omitting `--batches 2 5` runs B1→B2→B5; every B2/B5 probe independently compares original B1 calls at the same initial weights regardless of whether suite B1 is included. `--architectures` can select a single model. Default per-process timeout is 180 seconds; raw logs and completed events survive a timeout. The explicit slot flag is checked before any ML import.

## Preserved checks and workload

Only the first B training cases are selected, at most five. Bumper uses 13,675 nodes, 10 times, seven fields and three globals; phone full v2 uses 76,065 nodes, 10 times, six fields, two globals and its original 561,184 directed FEM edges. Phone's scalar is a reference volume-cuberoot length feature; its cache's legacy `thickness` key does not imply physical shell thickness.

At identical initial weights in eval mode:

1. Native batch outputs versus the original main B1 adapter applied individually.
2. Reversed sample order and perturbing only the last sample's normalized global input; unchanged samples must remain independent. The changed sample also matches original B1 execution.
3. Gradient of batch-mean MSE versus the mean of B original B1 gradients. One correctness backward pass per path also supplies backward warmup.
4. Two warm CUDA-synchronized measurements of B original B1 forwards versus one native batch forward, and the corresponding forward/backward workload. Ordering alternates between repeats.
5. Exactly three actual Adam updates on that batch, with finite outputs/gradients/weights and changed-weight verification.

Output tolerances are `atol=1e-5, rtol=1e-4`; gradient tolerances are `atol=1e-5, rtol=5e-4`, elementwise. Full per-parameter maximum absolute and relative L2 differences are retained. A failed tolerance is reported and never automatically relaxed. Seeds and deterministic cuDNN settings are recorded; native scatter may retain float32 reduction-order variation, so bitwise repeatability is not claimed and strict deterministic-algorithm mode is not forced.

Each worker writes `result.json`, flushed `events.jsonl`, source snapshots and hashes. The suite retains unmodified combined stdout/stderr in `B*.raw.log`, and OOM workers additionally write `failure_traceback.txt`. Both forward paths retain all B outputs through synchronization and memory sampling. CUDA allocated/reserved peaks and RSS are measured at phase boundaries; `windows_process_peak_wset_bytes` is process-wide context and is not a per-phase CPU peak. CUDA allocator memory excludes driver allocations. Phase timing includes adapter layout/graph preparation and excludes data reading and device transfer.

The separate three-update Adam probe observes `3*B` simulations, whereas B1's three updates observe three. Equal epoch counts with larger batches reduce optimizer updates per epoch to `ceil(train_count/B)`. Throughput gain alone does not establish equal convergence or justify altering the main training recipe. Only the root agent may decide to apply batching after actual successful measurements.

## Finite measurement budget

| Batch | Forward-only model calls | Forward/backward model calls, including Adam | Forward simulation observations | Forward/backward simulation observations |
| --- | ---: | ---: | ---: | ---: |
| B1 | 6 | 9 | 6 | 9 |
| B2 | 12 | 12 | 17 | 18 |
| B5 | 21 | 21 | 41 | 45 |

For B2+B5, the no-batching-gain scheduling approximation per model/domain is `58 * warm_B1_forward_seconds + 63 * warm_B1_forward_backward_seconds + two fresh-process startup times`. This is a queue estimate, not a prediction of batch feasibility or speedup. Current rough planning range is 8–18 minutes for both domains and all six models at B2/B5, excluding interruptions; main full-domain B1 measurements can tighten it. OOM/failure stops larger batches and may reduce time. No batch has yet been executed.

Static validation evidence: `static-validation.json`. GPU correctness, actual time, memory and OOM behavior remain unverified until the allocated run.
