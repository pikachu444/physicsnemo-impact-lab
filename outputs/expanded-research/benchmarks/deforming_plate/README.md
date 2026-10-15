# DeformingPlate execution record

This folder contains a bounded execution of NVIDIA PhysicsNeMo's structural DeformingPlate example, not a reproduction of the published 8-H100 experiment.

## Sources and data

- NVIDIA example revision: `11685edb3ec1fb6f25665e96cf7a8c4d11a23f16`, `examples/structural_mechanics/deforming_plate/`.
- Original COMSOL trajectories: `https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/`.
- HEAD measurements: train 9,906,860,823 bytes; valid 793,258,168 bytes; test 805,682,868 bytes. The full files were not downloaded.
- The first 8 original training, 2 validation and 2 test trajectories were acquired as complete 400-step records with HTTP Range. Each length and payload CRC32C was checked. `data-manifest.json` records ranges, hashes and shapes. This selection preserves independent trajectories and original splits; it is not a representative random sample of the full data.
- Actual transfer: 91,480,300 bytes including repeated record headers; 57.06 seconds. Compressed caches remain in `work/expanded-plate/data/` and are not published in Git.
- `meta.json` contains tetrahedra, node types, reference coordinates, 400 world-position frames and scalar nodal stress. It contains no original strain label or unit declaration. Do not label the scalar stress Pa/MPa from its numerical magnitude. The primary paper identifies it as von Mises stress and the problem as quasi-static, with no physical time step.

## Model and assessment contract

The official `HybridMeshGraphNet` predicts three position increments and next-step scalar stress. Inputs preserve the official node-type one-hot encoding; bidirectional tetrahedral mesh edges; reference/current relative xyz and lengths; radius-0.03 world edges excluding mesh edges; separate mesh/world encoders. The stock inference discards stress and labels world-position norm as displacement. This driver retains and denormalizes stress, and evaluates displacement from reference coordinates.

Training normalization uses training trajectories only. Checkpoints are selected by clean teacher-forced validation MSE across four normalized output channels. Autoregressive 399-step held-out evaluation is separate: the predicted free-node positions are fed back, while actuator and clamp motion is prescribed as in the official example. Teacher-forced and long-rollout errors must not be conflated.

A separate derived strain assessment uses `F = Ds @ inv(Dm)` and `E = 0.5*(F.T @ F-I)` on tetrahedra that do not touch the actuator. The six channels are `xx,yy,zz,xy,yz,xz`, with tensor shear. Both FEM and AI strains are derived from nodal coordinates. This is dimensionless Green-Lagrange strain, not an original COMSOL strain output, logarithmic strain, plastic strain or an independent constitutive validation. Element inversions and the sampled frames are reported. Separate solver strain outputs are needed to meet the original-label validation requirement.

## Reproduce

Use the project's separate native PyTorch/PyG/torch-scatter GPU runtime; the earlier PyTorch 2.13 runtime is left unchanged. Copy `acquire_plate_executed.py` as `work/expanded-plate/acquire_plate.py`, together with `plate_data.py`, `train_plate.py`, `plot_plate.py` and `meta.json`. The scripts expect this repository root as the working directory. Run data acquisition, preprocessing, the measured GPU resource probe, then the bounded training command recorded with each run. A successful resource probe is not a trained accuracy result.

Independent CPU source/kinematics checks and actual GPU numeric audits are in `review/`. Completed models and their accuracy failures are recorded below.

## Rights and limitations

NVIDIA source snapshots retain their Apache-2.0 headers. The DeepMind repository code license does not by itself establish a separate data license or provide a reusable COMSOL model. Original large data, third-party binaries and environment folders are excluded from publication.

## Completed GPU pilot: 8 training trajectories

The native GPU execution completed 12 epochs (1,920 Adam updates; 20 evenly spaced adjacent-step training pairs per independent trajectory, spanning current frames 0..398 and next frames 1..399). This sampling was chosen from the actual 47.44 ms resource probe and a roughly two-minute training budget, not from an accuracy claim. The seed is 20261015. The official model family and graph contract are retained; hidden width 48 and three processor blocks are a bounded capacity setting, not the README's width-128/15-block full recipe.

The fresh-noise standard deviation is the official loader default 0.003. Training target variance includes the analytic variance of this noise; this differs from the pinned loader's single realized noise draw but has the same zero-mean augmentation objective. Training statistics use eight training trajectories only. Validation/test source splits remain separate.

| Actual measured result | Value |
|---|---:|
| Trainable parameters | 119,284 |
| Training time, excluding imports/data setup/evaluation | 126.880 s |
| Selected epoch, clean teacher-forced validation MSE | 5 / 0.170232 |
| Actual CUDA peak tensor allocation | 263,746,048 bytes |
| Sampled process RSS peak (not a continuous OS high-water mark) | 1,671,213,056 bytes |
| Two test trajectories, 399-step AR displacement relative L2 | **175.303%** |
| Zero-increment displacement baseline relative L2 | 100.000% |
| AR von Mises stress relative L2 | **75.624%** |
| Constant training-mean stress baseline relative L2 | **73.761%** |
| Derived six-component GL strain relative L2, 21 sampled frames | **541.630%** |
| Final predicted inverted plate tetrahedra, tests 0000/0001 | **75 / 1** |
| Corresponding FEM reference inversions | 0 / 0 |

This pilot completed genuine training, inference and FEM-label comparison, but **failed the useful-accuracy gate**. Its displacement is worse than zero motion, its stress is worse than a constant training-mean predictor, and its predicted geometry contains inverted elements. The result does not establish that PhysicsNeMo cannot solve the problem; it establishes that this small-data/small-capacity/short-training configuration is not adequate. The learning curve falls in training loss but does not show stable validation improvement. More data or altered training must be tested instead of calling this a successful surrogate.

`pilot8/metrics.json`, `history.json`, `config.json` and checkpoint preserve the measurements. `pilot8/fields_comparison.png` shows actual FEM/prediction/error fields, on reference XY coordinates, with shared scales for FEM and prediction. `pilot8/learning_curve.png` shows the selected checkpoint. No response vector is supplied to the model as an input.

A fresh Python process reconstructed the model and loaded the saved checkpoint for the first 20 AR steps of each test trajectory. Maximum position difference was 5.96e-8 dataset coordinate units and maximum stress difference was 0.1133 dataset stress units. These are small CUDA scatter reduction differences; bitwise equality is not claimed. This is a checkpoint reuse check, not a second independent 399-step accuracy experiment (`checkpoint-reuse-check.json`). The original accuracy run itself also reloads the selected saved state before evaluation.

### Actual commands

```powershell
$env:PHYSICSNEMO_FORCE_TE = 'False'
& work/expanded-ml/venv/Scripts/python.exe -X utf8 work/expanded-plate/train_plate.py --output outputs/expanded-research/benchmarks/deforming_plate/probe48 --probe 10
& work/expanded-ml/venv/Scripts/python.exe -X utf8 work/expanded-plate/train_plate.py --output outputs/expanded-research/benchmarks/deforming_plate/pilot8 --epochs 12 --frames-per-trajectory 20 --rollout-steps 399
& work/expanded-ml/venv/Scripts/python.exe -X utf8 work/expanded-plate/train_plate.py --output outputs/expanded-research/benchmarks/deforming_plate/pilot8_reuse --checkpoint outputs/expanded-research/benchmarks/deforming_plate/pilot8/checkpoint.pt --rollout-steps 20
```

Native runtime: PyTorch 2.10.0+cu126, PhysicsNeMo 2.2.2, PyG 2.8.1 and official compiled torch-scatter 2.1.2+pt210cu126. The radius graph is constructed on CPU using SciPy cKDTree with no neighbor cap; model training and inference are on CUDA. No fake scatter or replacement neural network is used. The runtime's Warp-driver warning is retained in the logs; this model uses the native Torch/PyG/scatter path and does not claim Warp GPU-kernel validation.

## Completed training-configuration sensitivity run

A second model used the same eight training trajectories, architecture, seed, 20 loading-step anchors and 12 epochs, with noise set to zero. **The target normalization changes as well**, so this is a joint noise/normalization/loss-weighting sensitivity experiment, not a strict noise-only ablation. Both selected checkpoints were then evaluated on the same two internal validation trajectories, in physical dataset units. Their normalized training/validation losses must not be compared directly across settings.

| Internal validation, 2 trajectories × 399 AR steps | Noise 0.003 | Noise 0 |
|---|---:|---:|
| Displacement component RMSE [dataset coordinate units] | 0.0425462 | 0.0316268 |
| Displacement relative L2 | 134.018% | 99.623% |
| Von Mises stress RMSE [dataset stress units] | 28,219.4 | 29,417.1 |
| Von Mises stress relative L2 | 80.813% | 84.243% |
| Derived GL component RMSE [dimensionless] | 0.111134 | 0.093030 |
| Final inverted tetrahedra, valid0000 / valid0001 | 3 / 109 | 3 / 6 |

Zero-motion validation displacement RMSE is 0.0317465; constant training-mean stress RMSE is 28,988.4. Noise zero reduces drift and inversion count but leaves motion near the zero-motion baseline and worsens stress. **Neither model passes the accuracy gate.** This tradeoff does not justify selecting a usable surrogate. Noise-zero training took 141.614 seconds and selected epoch 5. Its development test0/1 relative errors were 98.419% displacement, 73.415% stress and 474.346% derived GL strain. These values are exploratory development results, not a final blind benchmark.

The internal validation reference and both models' full fields are preserved locally in `pilot8_validation/` and `noise0_pilot8/`. New runs also retain teacher-forced prediction arrays, allowing independent numerical audits of those values. The first pilot did not retain teacher-forced arrays, and its teacher-forced individual numbers are checked from source/formula rather than independently regenerated from an output file.

The message-passing depth is deliberately reduced to three blocks for resource calibration. Limited graph reach is one possible cause of poor whole-plate coupling; this has **not** been established as the cause by a depth comparison. A width-128 / 15-block official-architecture resource probe is prepared but has not run. The failed reduced-capacity pilots are not proof that the official full training recipe fails.

## Expanded data acquired; fresh final test remains unused

A total of 46 complete 400-step trajectories is now cached: 32 training, six validation and eight original test records. Cumulative transferred data including repeated Range headers is 323,510,820 bytes; the additional acquisition took 166.69 seconds after the initial 57.06-second acquisition. The compressed cache occupies 243,890,690 bytes. Every record passed TFRecord length/payload CRC32C and SHA256 checks; existing cache hashes were checked before reuse. No entire multi-GB TFRecord was downloaded.

`final-split-contract.json` freezes the next split before final model selection: train0000..0031, valid0000..0005, development source test0000..0001, and **fresh final source test0002..0007**. The old two test trajectories became development data after their results were inspected. The fresh six trajectories have not been used for prediction accuracy, model/epoch selection or algorithm ranking. A source-decoding integrity check is not a prediction evaluation. Future model size/training caps must be set using measured throughput and validation results before opening the fresh test performance.

At this point, two actual HybridMeshGraphNet models are trained; no model has been trained on the expanded 32-trajectory set yet. These records preserve an honest failed experiment and a reproducible path to a larger experiment, alongside the successful installation and native CUDA execution. Public data do not supply original COMSOL strain outputs; original solver-strain validation requires a separate generated FEM dataset and is not silently replaced by derived kinematics.

### Entry point and compatibility scope

The actual Plate entry point is the new lazy driver `train_plate.py`, which instantiates the installed, unmodified `physicsnemo.models.meshgraphnet.HybridMeshGraphNet`. The stock NVIDIA `train.py`/`inference.py` files were inspected and preserved, not directly launched for these Plate runs. The stock all-loading-step `.pt` preprocessing is replaced with on-demand CPU radius graphs and per-trajectory reads to bound RAM and storage. This preserves the documented graph/target mathematics but is not an unmodified stock-entry-point reproduction. Library implementation is PyPI PhysicsNeMo 2.2.2; example feature/target specification is pinned to the cited NVIDIA commit. Both versions are recorded rather than pretending the installed package is the Git commit.

Future resource commands can select the largest observed training graph: `--hidden 128 --processor 15 --probe 3 --probe-trajectory train_0011 --probe-frame 319 --train-count 32 --valid-count 6 --test-start 2 --test-stop 8`. This command is **prepared, not executed**. At the sampled state it has 1,488 nodes and 60,298 directed edges. Optional `--checkpoint-segments 15` selects activation checkpointing; any such resource test must be logged as a separate measurement. Training geometry sampling is recorded in `training-resource-geometries.json` and does not inspect fresh test accuracy.

The development stress hotspot check also fails: in source trajectory test0001, FEM peak von Mises stress is 494,882.47 at loading step 360. At that same point and step the noise-0.003 model predicts 27,162.76 (94.51% local error), and the noise-zero model predicts 31,294.06 (93.68% local error), all in unconfirmed dataset stress units. The file `development-stress-hotspots.json` identifies array indices, not solver node IDs. These actual localized misses matter for design decisions; a global average error alone would conceal them. No fresh-final-test hotspot has been inspected.
