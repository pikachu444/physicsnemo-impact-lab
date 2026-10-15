# Training-driver read-only review

Reviewed initial `work/expanded-plate/train_plate.py` (hash recorded in `driver-review.json`), without importing the program, Torch, or using the GPU. Syntax compilation, isolated metric checks, and an independent NumPy recomputation of clean train-only target normalization passed.

No blocking feature/label alignment error was found before the first training run:

- Reference and current mesh edge features occupy four columns each; current world edge features are repeated to eight columns, and the concatenated edge index preserves mesh-then-world order.
- The source target is the next position increment and next scalar stress, with fresh positional noise only on type 0 nodes and the matching negative noise correction to the increment target.
- The analytic variance addition uses the equal-trajectory fraction of free nodes, matching the stated equal-trajectory statistics convention. It intentionally differs from the official code's single finite sampled-noise realization and should remain documented.
- Only training trajectories supply normalization. Validation frames select a checkpoint; test trajectories are used in the final evaluator, not loss selection.
- The autoregressive state feeds back predictions and prescribes the future actuator/clamp motion. Teacher-forced and autoregressive metrics are separate. Free-node masks prevent prescribed boundary position accuracy from dominating displacement results.
- The strain element mask excludes actuator type 1 and retains clamped plate type 3. Component order and reference geometry are consistent with the independently checked helper.

Actionable adjustments sent to the writer before training:

1. In checkpoint reuse, initialize best epoch and validation score from the saved checkpoint. Otherwise the JSON writer emits nonstandard `Infinity` for the unused fresh best score. Save the actually loaded normalization after loading, rather than retaining a recomputed file beside checkpoint-driven inference.
2. Record the strain weighting as unweighted across sampled tetrahedra and tensor components, and identify the sampled frames. This metric is not volume weighted or a full 399-frame strain metric. Tensor shear is not engineering shear.
3. Record the helper source hash as well as the driver source hash, because graph construction and strain reside in `plate_data.py`.
4. Name `evaluation_seconds` as combined graph construction, autoregressive inference, and teacher-forced inference time; it cannot be reported as pure neural forward latency or as a single-rollout duration.

Remaining methodological limit: the checkpoint criterion is teacher-forced one-step loss on 12 selected validation frames per trajectory. It does not establish stable or converged 399-step rollout behavior. Actual held-out rollout metrics and inversion counts determine whether this bounded model is usable; a failed rollout is a valid reported result, not a reason to describe only one-step accuracy.
