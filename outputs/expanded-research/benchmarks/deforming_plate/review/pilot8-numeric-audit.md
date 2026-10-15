# Pilot8 independent numeric audit

Actual training and held-out evaluation completed; the bounded pilot fails useful-accuracy checks. CPU-only independent recomputation of preserved autoregressive displacement, scalar stress, derived six-component Green–Lagrange strain, zero/constant baselines, and pooled metrics passed. All 12 source NPZ hashes, disjoint trajectory splits, training-only normalization names, source/helper snapshots, checkpoint hash, and minimum-validation checkpoint epoch 5 match. Checkpoint scalar metadata was inspected using pickle opcodes without deserializing tensors or importing Torch.

| Pooled quantity | Relative L2 | Component RMSE | Comparison |
|---|---:|---:|---|
| 399-step free-node displacement | 1.753035 | 0.04896863 dataset coordinate units | Worse than zero increment (L2 1) |
| Free-node scalar stress | 0.756237 | 41675.53 dataset stress units | Worse than training mean (L2 0.737609) |
| Derived GL strain, 21 selected frames | 5.416296 | 0.08375337 dimensionless | Invalid deformation growth |

Final predicted plate inversion counts are 75 and 1 on the two test trajectories; matching ground-truth counts are 0 and 0. Prescribed actuator/clamp positions match the original next-step positions at all 399 steps. Strain tensors were independently reconstructed with linear solves rather than reusing the worker helper. These are kinematic strain labels derived from positions, not exported COMSOL strain labels. Stress and coordinate SI units remain unconfirmed.

Teacher-forced predictions were not saved in the NPZ files. Their per-trajectory numeric errors are therefore not independently recalculated; only pooling arithmetic and reviewed source formulas are checked. This limitation does not affect the independently checked autoregressive results.

This is 8 training, 2 validation and 2 test trajectories with a reduced 119,284-parameter HybridMeshGraphNet, 12 epochs and 20 selected training increments per trajectory. It does not reproduce the full published training recipe, and its failure is not evidence that the full algorithm cannot solve the domain. Test labels must not be used to select a replacement model.
