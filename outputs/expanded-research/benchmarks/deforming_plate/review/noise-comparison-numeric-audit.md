# Independent numeric audit of plate recipe comparison

All preserved autoregressive and teacher-forced position/stress predictions, six-component derived strain, physical-unit RMSE, pooled metrics, masks, source labels and sampled inversion counts passed independent NumPy recomputation. Snapshot/helper/data/checkpoint hashes and best-validation epoch 5 were checked; checkpoint scalar metadata was read without tensor unpickling or importing Torch. Arrays were handled one trajectory at a time.

| Recipe and evaluation group | Displacement L2 | Displacement RMSE* | Stress L2 | Stress RMSE* | Derived GL L2 |
|---|---:|---:|---:|---:|---:|
| noise .003: validation2 | 1.340183 | 0.04254616 | 0.808130 | 28219.44 | 15.728644 |
| noise 0: validation2 | 0.996230 | 0.03162684 | 0.842427 | 29417.10 | 13.166351 |
| noise 0: development source test2 | 0.984190 | 0.02749202 | 0.734151 | 40458.44 | 4.743463 |

*Dataset coordinate and stress units, respectively; SI units are unconfirmed. Green–Lagrange strain is dimensionless and derived from positions, not an original COMSOL strain export. Its reported metric uses 21 selected frames and unweighted tensor components/elements.

Noise .003 validation final predicted inversions are 3 and 109. Noise 0 validation final predicted inversions are 3 and 6; its development source test final inversions are 1 and 1. All matching source FEM counts are zero. Removing noise improves displacement relative to the first recipe but leaves it near the zero-increment baseline; stress improves only marginally on the development test group and remains worse than the training-mean stress baseline on validation. Neither recipe establishes useful long-rollout strain or valid deformed meshes.

The target normalization changes along with the noise setting. This is therefore a coupled recipe comparison, not a strict noise-only causal experiment. Its normalized validation losses cannot be compared directly; the table uses errors in original dataset units. Graph preparation, loss and validation functions are identical across the executed source snapshots. The reuse snapshot additionally passes processor checkpoint segments = 0 explicitly, matching the installed constructor default and recorded configuration; architecture sizes are unchanged. Evaluation/serialization code differs and was reviewed by each run's snapshot/hash.

Source test0/1 are development cases after their initial results were inspected, not a fresh blind final test. Validation trajectories also selected checkpoints. The checkpoint reuse run does not train for its inert default epochs argument.

Sampled peak RSS: 61,894,656 bytes, below 100 MB. This is sampled process RSS rather than a continuously measured OS peak. No Torch import or GPU use occurred.
