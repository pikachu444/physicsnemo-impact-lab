# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V3 native scalar outputs for the pinned GeoTransolver AR/TF rollouts.

The first three core outputs are normalized acceleration. Remaining outputs are
native, train-normalized scalar fields at the predicted frame. Only positions
and velocities recur; scalar predictions never become model inputs or geometry.
The factory imports the ML runtime only when called after resource allocation.
"""

EPS = 1e-8
STRATEGIES = ("ar_rollout_training", "one_step_teacher_forcing")


def make_model(core_class, strategy, **config):
    """Build a pinned GeoTransolver subclass with unchanged core state keys.

    ``config`` has the official rollout constructor arguments, with ``out_dim``
    equal to the target channel count (C6/C7, or C3 for equivalence checks).
    Returned tensors are [N, T, C]: position3 followed by normalized scalars.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported native scalar rollout strategy: {strategy}")
    if config.get("out_dim", 0) < 3:
        raise ValueError("out_dim must include three acceleration channels")
    if config.get("num_time_steps", 0) < 2:
        raise ValueError("num_time_steps must include an initial and future frame")

    import torch
    from torch.utils.checkpoint import checkpoint as ckpt

    def _geo_global_emb(sample):
        if sample.global_features is None:
            return None
        g = torch.stack([sample.global_features[k] for k in sample.global_features], dim=0)
        return g.unsqueeze(0).unsqueeze(0)

    class NativeScalarAutoregressiveRollout(core_class):
        """Official autoregressive position integration plus native scalars."""

        def __init__(self, *args, **kwargs):
            self.dt = kwargs.pop("dt")
            self.initial_vel = kwargs.pop("initial_vel")
            self.rollout_steps = kwargs.pop("num_time_steps") - 1
            super().__init__(*args, **kwargs)

        def forward(self, sample, data_stats):
            inputs = sample.node_features
            coords = inputs["coords"]
            features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
            global_emb = _geo_global_emb(sample)

            y_t1 = coords
            y_t0 = y_t1 - self.initial_vel * self.dt

            outputs = []
            for t in range(self.rollout_steps):
                vel = (y_t1 - y_t0) / self.dt
                vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                    data_stats["node"]["norm_vel_std"] + EPS
                )
                fx_t = torch.cat([vel_norm, features], dim=-1)

                def step_fn(local_emb, geometry, local_pos):
                    return super(NativeScalarAutoregressiveRollout, self).forward(
                        local_embedding=local_emb,
                        geometry=geometry,
                        local_positions=local_pos,
                        global_embedding=global_emb,
                    )

                if self.training:
                    raw = ckpt(
                        step_fn,
                        fx_t.unsqueeze(0),
                        y_t1.unsqueeze(0),
                        y_t1.unsqueeze(0),
                        use_reentrant=False,
                    ).squeeze(0)
                else:
                    raw = step_fn(
                        fx_t.unsqueeze(0), y_t1.unsqueeze(0), y_t1.unsqueeze(0)
                    ).squeeze(0)

                acc = (
                    raw[..., :3] * data_stats["node"]["norm_acc_std"]
                    + data_stats["node"]["norm_acc_mean"]
                )
                vel = self.dt * acc + vel
                y_t2 = self.dt * vel + y_t1

                outputs.append(torch.cat([y_t2, raw[..., 3:]], dim=-1))
                y_t1, y_t0 = y_t2, y_t1

            return torch.stack(outputs, dim=0).transpose(0, 1)

    class NativeScalarOneStepRollout(core_class):
        """Position-only teacher forcing in training; free rollout in eval."""

        def __init__(self, *args, **kwargs):
            self.dt = kwargs.pop("dt", 5e-3)
            self.initial_vel = kwargs.pop("initial_vel")
            self.rollout_steps = kwargs.pop("num_time_steps") - 1
            super().__init__(*args, **kwargs)

        def forward(self, sample, data_stats):
            inputs = sample.node_features
            coords0 = inputs["coords"]
            features = inputs.get("features", coords0.new_zeros((coords0.size(0), 0)))
            global_emb = _geo_global_emb(sample)

            if self.training:
                gt_seq = torch.cat(
                    [coords0.unsqueeze(0), sample.node_target[..., :3].transpose(0, 1)],
                    dim=0,
                )

            outputs = []
            y_t0 = coords0 - self.initial_vel * self.dt
            y_t1 = coords0

            for t in range(self.rollout_steps):
                if self.training and t > 0:
                    y_t0, y_t1 = gt_seq[t - 1], gt_seq[t]

                vel = (y_t1 - y_t0) / self.dt
                vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                    data_stats["node"]["norm_vel_std"] + EPS
                )
                fx_t = torch.cat([vel_norm, features], dim=-1)

                def step_fn(local_emb, geometry, local_pos):
                    return super(NativeScalarOneStepRollout, self).forward(
                        local_embedding=local_emb,
                        geometry=geometry,
                        local_positions=local_pos,
                        global_embedding=global_emb,
                    )

                if self.training:
                    raw = ckpt(
                        step_fn,
                        fx_t.unsqueeze(0),
                        y_t1.unsqueeze(0),
                        y_t1.unsqueeze(0),
                        use_reentrant=False,
                    ).squeeze(0)
                else:
                    raw = step_fn(
                        fx_t.unsqueeze(0), y_t1.unsqueeze(0), y_t1.unsqueeze(0)
                    ).squeeze(0)

                acc = (
                    raw[..., :3] * data_stats["node"]["norm_acc_std"]
                    + data_stats["node"]["norm_acc_mean"]
                )
                vel_pred = self.dt * acc + vel
                y_t2_pred = self.dt * vel_pred + y_t1

                outputs.append(torch.cat([y_t2_pred, raw[..., 3:]], dim=-1))

                if not self.training:
                    y_t0, y_t1 = y_t1, y_t2_pred

            return torch.stack(outputs, dim=0).transpose(0, 1)

    if strategy == "ar_rollout_training":
        return NativeScalarAutoregressiveRollout(**config)
    return NativeScalarOneStepRollout(**config)
