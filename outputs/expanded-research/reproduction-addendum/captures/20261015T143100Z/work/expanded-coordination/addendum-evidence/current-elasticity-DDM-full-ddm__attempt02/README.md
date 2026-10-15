# Completed numerical elasticity DDM

The pinned official structural-only elasticity recipe completed on CPU and converged after **64 Schwarz iterations**, with interface RMSE **8.62933e-9** against tolerance1e-8. Independent manufactured-field and energy checks passed **25/25**.

This executes the official `run_elasticity` function, actual Q1/BiCGSTAB numerical solver, domain assembly, exchange handler and Schwarz iteration at commit `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`. The isolated Schwarz copy removes exactly two unused thermal/FNO imports. The other six copied scientific files are byte-identical to the pinned source. This is an adapted function execution, not an untouched official CLI or full installed PhysicsNeMo-package claim.

The problem is the source default32x32 unit square,2x2 overlapping18x18 patches, overlap4, maximum100iterations, E=120 and nu=.25 in consistent model units. No neural network was trained:65 numerical batch calls covered one monolithic problem and256 local patch problems.

| Field | DDM versus monolithic relative L2 | DDM versus manufactured target relative L2 |
|---|---:|---:|
| Displacement |2.780326e-6 |2.781775e-6 |
| Small strain tensor |5.573448e-6 |5.575791e-6 |
| Plane stress tensor |6.589849e-6 |6.592613e-6 |
| Von Mises stress |3.798825e-6 |3.800702e-6 |

Monolithic versus manufactured relative L2 errors are1.87573e-9 for displacement,3.85277e-9 for strain,4.19850e-9 for stress and2.84245e-9 for von Mises, using the same boundary-aware recovery target.

The manufactured comparison accounts for the source's first-order one-sided recovery on outer y rows. Uncorrected continuum full-grid errors are0.00314902 for strain,0.00419156 for stress and0.00120959 for von Mises; these contain the known boundary derivative error. They are retained in `independent-checks.json` rather than presented as zero-error fields.

The analytic displacement is u_y=.01y² and u_x=-.032xy. The independent continuum strain/stress formulas give zero body force and continuum energy0.031744. The exact nodal Q1 interpolant energy is0.031743556018036766; independent quadrature of DDM displacement gives0.03174355601937683, relative difference4.22154e-11. DDM free-force residual relative to full nodal internal-force norm is1.65253e-5; monolithic is2.25559e-9. The checks validate this linear, static manufactured case only.

These are derived small linear strain and plane-stress elastic Cauchy fields. They are not native nonlinear, contact, dynamic, plastic, fracture or appliance FEM labels. The source declares no SI material units.

![Monolithic and numerical DDM comparison](${WORKSPACE}/outputs/expanded-research/benchmarks/elasticity_numerical/full-ddm/attempt02/comparison.png)

The first two columns share their color scale. The last column is the absolute difference of each displayed scalar magnitude or norm. Table errors above use complete vector/tensor fields.

PID26748 ran2026-10-15T08:28:27.3674913Z to08:29:17.7874967Z and exited0. Supervisor wall time was50.4202s, official recipe40.3691s and monolithic solve1.45038s. Maximum observed process working set was209,960,960bytes (200.23MiB), minimum observed host free memory2,265,337,856bytes. Two available-memory gates10seconds apart observed3,881,943,040 and3,948,777,472bytes. Global Torch2.13.0+cpu used CPU1/BelowNormal; CUDA was invisible and uninitialized. No thermal/FNO modules were imported. The child was reaped and independently confirmed absent at08:29:54.6231872Z.

The owned GPU/static-learning suite had already exited before this CPU window. `execution-context-correction.json` corrects a stale literal in the preserved raw `execution.json`; original raw hashes and journal are unchanged. Stage times include observation overhead and normal shared-host variability; no isolated speed comparison is claimed.

`execution.json`, `supervisor.json`, `stage-events.jsonl`, `elasticity_metrics.csv`, `independent-checks.json`, `fields.csv`, stdout/stderr and PNG preserve the execution evidence. Earlier1.3GiB gate deferral and guard-stopped11-iteration attempt01 remain in the parent directory. The v1/v2/v3 resource protocols, executed wrapper, source manifest, independent checker and observer self-check remain under `work/expanded-elasticity-ddm`.
