# Independent preparation review

2026-10-15 — read-only reviewer `/root/batch_probe/review_batch_static`: **approve** after one bounded correction.

Confirmed statically:

- MeshGraphNet keeps real FEM edges and edge-feature order in disconnected `b*N` graph blocks.
- FIGConvUNet receives contiguous batched inputs and uses native per-case KNN offsets.
- The loader opens only selected training input/target arrays and rejects held-out target aliases before loading.
- Eval output/order/global independence, original-B1 mean-gradient comparison, exactly three Adam steps, fresh subprocesses and stop-after-failure sequencing are present.
- Seed, TF32 and deterministic cuDNN settings are recorded with the native scatter numerical-rounding limitation.
- Both forward timing paths retain B simulation outputs through synchronization/memory sampling. Phase-boundary RSS is distinguished from Windows process-wide peak working set.

The reviewer performed no ML-library import, array loading or GPU execution. This approval covers source/preparation correctness; runtime numerical tolerances, actual batching feasibility, CUDA memory and speed remain unverified until an allocated run.
