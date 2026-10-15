# Official-size plate final metadata audit

PASS_SCOPED_METADATA_AND_JSON_AGGREGATE_AUDIT

This is independent stdlib metadata/provenance and JSON aggregate consistency only. It is not raw-array verification, a visual audit or an accuracy qualification.

Checks: 68, failures: 0. CPU1 applied: True; no framework/array/NN access.
Selected epoch: 5; fresh test2..7 six completed AR399 attempts; no teacher forcing.
Checkpoint: `c29e4ecd258553622ecee13fc054a58854950e667b3a838610ab52f799339429`.

| Case | u relative L2 | stress relative L2 | derived GL relative L2 | predicted/reference inverted frames |
|---|---:|---:|---:|---:|
| test_0002 | 8.96173505 | 0.717597849 | 9374.64931 | 385/0 |
| test_0003 | 2.16639171 | 0.682944131 | 57.437999 | 73/0 |
| test_0004 | 2.39484793 | 0.66334808 | 9.76844469 | 0/0 |
| test_0005 | 2.01097207 | 0.647729102 | 32.3338752 | 24/0 |
| test_0006 | 54.6968388 | 0.887477714 | 31.4261555 | 119/0 |
| test_0007 | 1.85774633 | 0.676662619 | 20.6621403 | 0/0 |

Relative L2 values above are fractions, not percentages. Native stress units remain unconfirmed; derived GL is dimensionless.
Training wrapper wall 2319.975s; final cold wall 291.370s. Training metrics timer includes validation/selection/checkpoint I/O and is not optimizer-only. Sampled RSS, Windows peak working set, commit, tensor allocation and allocator reserve remain separate fields.

Limitations:
- JSON arithmetic checks reuse producer per-case metrics; not independent raw/FEM/prediction-array recomputation.
- Source and checkpoint byte hashes verified; checkpoint tensors not loaded.
- Fresh array hashes in gate are recorded claims, not reverified in this audit.
- Marker/access order is source plus matching receipts, not OS file-open tracing.
- One attempt and no extra inference supported by exclusive marker/source/receipts; no global syscall proof.
- New-PID strict checkpoint load is supported; no same-validation numerical reload equality check performed.
- Allocator allocated/reserved differ from total device usage; host RSS differs from commit. External device observation samples were not numerically re-audited.
- Derived GL covers21 sampled frames; not native COMSOL strain.
- Runtime completed but errors/inversions do not qualify engineering accuracy; no visual review.
